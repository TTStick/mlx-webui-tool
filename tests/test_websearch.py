"""联网搜索：服务接入、安全边界、隐私约定。"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest
from conftest import wait_job

from mlxui import websearch
from mlxui.websearch import Source, clean_query, extract_text, failure_note, render_block


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def http(monkeypatch):
    calls = []

    def install(payload=None, error=None):
        def opener(request, timeout=None):
            calls.append(request)
            if error is not None:
                raise error
            return FakeResponse(json.dumps(payload).encode())

        monkeypatch.setattr(websearch.urllib.request, "urlopen", opener)
        return calls

    return install


def new_conv(client, model, **settings):
    base = {"max_tokens": 6, "temperature": 0.0, "context_length": 2048}
    return client.post("/api/conversations", json={"model": model, "settings": {**base, **settings}}).json()["id"]


def send(client, app, conv, text):
    res = client.post(f"/api/conversations/{conv}/send", json={"content": text}).json()
    return res, wait_job(app, res["job_id"])


# ------------------------------------------------------------------ 服务
def test_wikipedia_picks_language_and_cleans_snippets(http):
    calls = http({"query": {"search": [{"title": "苹果 公司", "snippet": '<span class="x">苹果</span>&amp;公司'}]}})
    results = websearch.search("苹果公司", {"provider": "wikipedia", "max_results": 3})
    assert calls[0].full_url.startswith("https://zh.wikipedia.org/w/api.php")
    assert results[0].snippet == "苹果&公司"
    assert results[0].url == "https://zh.wikipedia.org/wiki/" + websearch.urllib.parse.quote("苹果_公司")


def test_wikipedia_uses_english_for_latin_queries(http):
    calls = http({"query": {"search": [{"title": "MLX", "snippet": "x"}]}})
    websearch.search("apple mlx", {"provider": "wikipedia"})
    assert calls[0].full_url.startswith("https://en.wikipedia.org/")


def test_searxng(http):
    calls = http({"results": [{"title": "T", "url": "https://a.example/x", "content": "<b>S</b>"}]})
    results = websearch.search("q", {"provider": "searxng", "searxng_url": "http://127.0.0.1:8888/"})
    assert calls[0].full_url == "http://127.0.0.1:8888/search?q=q&format=json"
    assert results[0].snippet == "S"


def test_brave_sends_key_in_header(http):
    calls = http({"web": {"results": [{"title": "T", "url": "https://a.example", "description": "<strong>D</strong>"}]}})
    results = websearch.search("q", {"provider": "brave", "brave_api_key": "K1"})
    assert calls[0].get_header("X-subscription-token") == "K1"
    assert "K1" not in calls[0].full_url
    assert results[0].snippet == "D"


def test_tavily_sends_key_in_header(http):
    calls = http({"results": [{"title": "T", "url": "https://a.example", "content": "C"}]})
    results = websearch.search("q", {"provider": "tavily", "tavily_api_key": "K2"})
    assert calls[0].get_header("Authorization") == "Bearer K2"
    assert b"K2" not in calls[0].data
    assert results[0].snippet == "C"


@pytest.mark.parametrize(
    ("settings", "keyword"),
    [
        ({"provider": "searxng", "searxng_url": ""}, "SearXNG"),
        ({"provider": "brave", "brave_api_key": ""}, "Brave"),
        ({"provider": "tavily", "tavily_api_key": ""}, "Tavily"),
    ],
)
def test_missing_configuration_is_explained_without_network(http, settings, keyword):
    calls = http({})
    with pytest.raises(websearch.SearchError, match=keyword):
        websearch.search("q", settings)
    assert calls == []


def test_unauthorized_key_is_explained(http):
    http(error=urllib.error.HTTPError("u", 401, "x", {}, None))
    with pytest.raises(websearch.SearchError, match="API Key"):
        websearch.search("q", {"provider": "brave", "brave_api_key": "bad"})


def test_rate_limit_is_explained(http):
    http(error=urllib.error.HTTPError("u", 429, "x", {}, None))
    with pytest.raises(websearch.SearchError, match="额度"):
        websearch.search("q", {"provider": "brave", "brave_api_key": "k"})


def test_network_failure_is_explained(http):
    http(error=urllib.error.URLError("down"))
    with pytest.raises(websearch.SearchError, match="连接"):
        websearch.search("q", {"provider": "wikipedia"})


def test_empty_results_are_reported(http):
    http({"query": {"search": []}})
    with pytest.raises(websearch.SearchError, match="没有搜到"):
        websearch.search("q", {"provider": "wikipedia"})


def test_no_silent_fallback_to_another_provider(http):
    calls = http(error=urllib.error.URLError("down"))
    with pytest.raises(websearch.SearchError):
        websearch.search("q", {"provider": "searxng", "searxng_url": "http://127.0.0.1:1"})
    assert len(calls) == 1


def test_requests_carry_no_cookies_or_browser_disguise(http):
    calls = http({"query": {"search": [{"title": "T", "snippet": "s"}]}})
    websearch.search("q", {"provider": "wikipedia"})
    assert calls[0].get_header("Cookie") is None
    assert calls[0].get_header("User-agent").startswith("MLXUI/")


def test_non_http_results_are_dropped(http):
    http({"results": [{"title": "T", "url": "file:///etc/passwd"}, {"title": "U", "url": "https://ok.example"}]})
    results = websearch.search("q", {"provider": "searxng", "searxng_url": "http://x.example"})
    assert [r.url for r in results] == ["https://ok.example"]


# ------------------------------------------------------------------ 安全
def resolve_to(monkeypatch, address):
    monkeypatch.setattr(websearch.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (address, 443))])


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.8", "192.168.1.1", "172.16.0.1", "169.254.169.254", "100.100.1.1", "::1", "fe80::1"],
)
def test_private_addresses_are_refused(monkeypatch, address):
    resolve_to(monkeypatch, address)
    with pytest.raises(websearch.UnsafeUrl):
        websearch._assert_public("http://evil.example/")


def test_public_address_allowed(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    websearch._assert_public("https://example.com/")


def test_fake_ip_proxy_range_is_allowed(monkeypatch):
    resolve_to(monkeypatch, "198.18.0.42")
    websearch._assert_public("https://example.com/")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a.example/", "http://"])
def test_bad_schemes_refused(url):
    with pytest.raises(websearch.UnsafeUrl):
        websearch._assert_public(url)


def test_redirect_to_internal_address_is_refused(monkeypatch):
    resolve_to(monkeypatch, "127.0.0.1")
    handler = websearch._SafeRedirect()
    with pytest.raises(websearch.UnsafeUrl):
        handler.redirect_request(None, None, 302, "", {}, "http://internal.example/")


def test_fetch_page_never_touches_private_hosts(monkeypatch):
    resolve_to(monkeypatch, "127.0.0.1")
    def explode(*args, **kwargs):
        raise AssertionError("should not connect")

    monkeypatch.setattr(websearch._opener, "open", explode)
    assert websearch.fetch_page("http://127.0.0.1:8420/api/settings") == ""


# ---------------------------------------------------------------- 正文提取
def test_extract_prefers_article_and_drops_chrome():
    body = "正文内容" * 80
    page = f"<html><nav>导航</nav><script>var x=1</script><article><h1>标题</h1><p>{body}</p></article><footer>版权</footer></html>"
    text = extract_text(page)
    assert "标题" in text and body in text
    assert "导航" not in text and "var x" not in text and "版权" not in text


def test_extract_falls_back_to_body():
    assert "段落" in extract_text("<body><div>段落</div><article>短</article></body>")


def test_render_block_respects_budget():
    sources = [Source(f"标题{i}", f"https://a{i}.example", "摘要", "很长的正文" * 2000) for i in range(3)]
    block = render_block("q", sources, len, 1500)
    assert len(block) <= 1500
    assert "标题0" in block and "标题2" in block and "[3]" in block


def test_render_block_warns_against_instructions_in_pages():
    block = render_block("q", [Source("t", "https://a.example", "s")], len, 2000)
    assert "不是用户说的" in block and "今天是" in block


def test_failure_note_tells_model_to_be_honest():
    assert "告诉用户" in failure_note("q", "网络错误")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("苹果 新品", "苹果 新品"),
        ("关键词：苹果 新品\n解释", "苹果 新品"),
        ('"apple news"', "apple news"),
        ("", "原始 问题"),
    ],
)
def test_clean_query(raw, expected):
    assert clean_query(raw, "原始   问题") == expected


def test_clean_query_is_capped():
    assert len(clean_query("字" * 500, "")) == 100


# -------------------------------------------------------------- 接入对话
@pytest.fixture()
def stubbed(app_ctx, monkeypatch):
    client, app, model = app_ctx
    calls: list[str] = []

    def fake_search(query, settings):
        calls.append(query)
        return [Source("资料标题", "https://a.example/p", "摘要内容")]

    def fake_enrich(sources, count):
        for source in sources:
            source.text = "这是抓取到的网页正文NETTEXT"

    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(websearch, "enrich", fake_enrich)
    return client, app, model, calls


def event(job, kind):
    return [e for e in job.slice(0) if e["type"] == kind]


def test_search_is_off_by_default(stubbed):
    client, app, model, calls = stubbed
    send(client, app, new_conv(client, model), "你好")
    assert calls == []


def test_search_flow_records_sources(stubbed):
    client, app, model, calls = stubbed
    conv = new_conv(client, model, web_search=True)
    _, job = send(client, app, conv, "最近的新闻")
    assert job.state == "done" and len(calls) == 1
    assert [e["stage"] for e in event(job, "search")] == ["query", "done"]
    context = event(job, "context")[0]
    assert context["breakdown"]["search"] > 0
    stored = client.get(f"/api/conversations/{conv}").json()["messages"][-1]["stats"]["search"]
    assert stored["sources"][0]["url"] == "https://a.example/p"


def test_page_text_is_not_persisted(stubbed):
    client, app, model, _ = stubbed
    conv = new_conv(client, model, web_search=True)
    send(client, app, conv, "最近的新闻")
    assert "NETTEXT" not in json.dumps(client.get(f"/api/conversations/{conv}").json(), ensure_ascii=False)


def test_search_results_stay_out_of_later_turns(stubbed):
    client, app, model, _ = stubbed
    conv = new_conv(client, model, web_search=True)
    send(client, app, conv, "最近的新闻")
    preview = client.post(
        "/api/context/preview", json={"conversation_id": conv, "draft": "继续", "include_prompt": True}
    ).json()
    assert "NETTEXT" not in preview["prompt"]


def test_search_query_never_sees_memory(stubbed):
    client, app, model, calls = stubbed
    engine = app.state.engine
    client.patch("/api/settings", json={"memory": {"enabled": True}})
    app.state.memory.add_manual("profile", "用户的真实姓名叫王秘密")
    prompts = []
    client.post("/api/models/load", json={"id": model})
    original = engine.generate_once

    def spy(tokens, *args, **kwargs):
        prompts.append(engine.tokenizer.decode(tokens))
        return original(tokens, *args, **kwargs)

    engine.generate_once = spy
    conv = new_conv(client, model, web_search=True)
    _, job = send(client, app, conv, "最近的新闻")
    assert prompts and "王秘密" not in prompts[0] and "最近的新闻" in prompts[0]
    assert "王秘密" not in calls[0]
    assert event(job, "context")[0]["breakdown"]["memory"] > 0


def test_attachments_are_not_used_for_the_query(stubbed):
    client, app, model, _ = stubbed
    engine = app.state.engine
    prompts = []
    client.post("/api/models/load", json={"id": model})
    original = engine.generate_once

    def spy(tokens, *args, **kwargs):
        prompts.append(engine.tokenizer.decode(tokens))
        return original(tokens, *args, **kwargs)

    engine.generate_once = spy
    uploaded = client.post("/api/attachments", params={"name": "a.txt"}, content="附件机密".encode()).json()
    conv = new_conv(client, model, web_search=True)
    res = client.post(f"/api/conversations/{conv}/send", json={"content": "总结", "attachments": [uploaded]}).json()
    wait_job(app, res["job_id"])
    assert "机密" not in prompts[0]


def test_failed_search_still_answers_and_says_so(app_ctx, monkeypatch):
    client, app, model = app_ctx

    def fail(query, settings):
        raise websearch.SearchError("网络不通")

    monkeypatch.setattr(websearch, "search", fail)
    conv = new_conv(client, model, web_search=True)
    _, job = send(client, app, conv, "最近的新闻")
    assert job.state == "done"
    assert event(job, "search")[-1]["stage"] == "failed"
    assert event(job, "context")[0]["breakdown"]["search"] > 0
    stored = client.get(f"/api/conversations/{conv}").json()["messages"][-1]["stats"]["search"]
    assert stored["error"] == "网络不通" and "sources" not in stored


def test_continue_does_not_search_again(stubbed):
    client, app, model, calls = stubbed
    conv = new_conv(client, model, web_search=True)
    data, _ = send(client, app, conv, "最近的新闻")
    calls.clear()
    res = client.post(f"/api/conversations/{conv}/continue", json={"message_id": data["messages"][-1]["id"]}).json()
    wait_job(app, res["job_id"])
    assert calls == []


def test_stop_during_search_skips_network(app_ctx, monkeypatch):
    client, app, model = app_ctx
    called = []
    monkeypatch.setattr(websearch, "search", lambda q, s: called.append(q) or [])
    engine = app.state.engine
    client.post("/api/models/load", json={"id": model})
    original = engine.generate_once

    def cancel_then_generate(tokens, *args, **kwargs):
        for job in engine.jobs.values():
            job.cancelled.set()
        return original(tokens, *args, **kwargs)

    engine.generate_once = cancel_then_generate
    conv = new_conv(client, model, web_search=True)
    send(client, app, conv, "最近的新闻")
    assert called == []


# ------------------------------------------------------------------ 设置
def test_api_keys_are_write_only(app_ctx):
    client, app, _ = app_ctx
    updated = client.patch("/api/settings", json={"search": {"brave_api_key": "SECRET123"}}).json()
    assert updated["search"]["brave_key_set"] is True
    assert "SECRET123" not in json.dumps(updated)
    assert "SECRET123" not in client.get("/api/settings").text
    assert app.state.db.get_settings()["search"]["brave_api_key"] == "SECRET123"

    cleared = client.patch("/api/settings", json={"search": {"brave_api_key": ""}}).json()
    assert cleared["search"]["brave_key_set"] is False


def test_search_settings_are_validated(app_ctx):
    client, _, _ = app_ctx
    assert client.patch("/api/settings", json={"search": {"provider": "google"}}).status_code == 400
    assert client.patch("/api/settings", json={"search": {"searxng_url": "not a url"}}).status_code == 400
    updated = client.patch("/api/settings", json={"search": {"max_results": 99, "fetch_pages": -1}}).json()
    assert updated["search"]["max_results"] == 10 and updated["search"]["fetch_pages"] == 0


def test_web_search_param_is_per_conversation(app_ctx):
    client, _, model = app_ctx
    conv = new_conv(client, model)
    assert client.get(f"/api/conversations/{conv}").json()["params"]["web_search"] is False
    detail = client.patch(f"/api/conversations/{conv}", json={"settings": {"web_search": True}}).json()
    assert detail["params"]["web_search"] is True
