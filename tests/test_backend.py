from __future__ import annotations

import json
from types import SimpleNamespace

from conftest import MemoryKeyStore, wait_job

from mlxui.config import normalize_params
from mlxui.context import StreamFilter, TokenCounter, Turn, build_context, strip_thinking
from mlxui.service import clean_title, first_line_title

FAST = {"max_tokens": 12, "temperature": 0.0, "context_length": 2048}


def new_conv(client, model, **settings):
    return client.post(
        "/api/conversations", json={"model": model, "settings": {**FAST, **settings}}
    ).json()["id"]


def send(client, app, conv, text="你好"):
    res = client.post(f"/api/conversations/{conv}/send", json={"content": text})
    assert res.status_code == 200, res.text
    data = res.json()
    job = wait_job(app, data["job_id"])
    return data, job


def event_types(job):
    return [e["type"] for e in job.slice(0)]


# ------------------------------------------------------------- 纯函数
def test_stream_filter_holds_split_stop_word():
    f = StreamFilter(["STOP"], None, None)
    assert f.feed("abcS") == ("abc", False)
    assert f.feed("T") == ("", False)
    assert f.feed("OPxyz") == ("", True)


def test_stream_filter_releases_false_prefix():
    f = StreamFilter(["STOP"], None, None)
    assert f.feed("xS") == ("x", False)
    assert f.feed("ay") == ("Say", False)


def test_stream_filter_normalizes_reasoning_tags():
    f = StreamFilter([], "<|channel>thought", "<channel|>")
    out = ""
    for piece in ["<|chan", "nel>thought", "abc<chan", "nel|>done"]:
        out += f.feed(piece)[0]
    out += f.flush()
    assert out == "<think>abc</think>done"


def test_strip_thinking():
    assert strip_thinking("<think>a</think>\n\nb") == "b"
    assert strip_thinking("x<think>unfinished") == "x"


def test_normalize_params_clamps_and_drops():
    out = normalize_params({"temperature": 9, "top_k": "x", "seed": "", "bogus": 1, "context_length": 1})
    assert out == {"temperature": 2.0, "seed": None, "context_length": 256}


def test_titles():
    assert first_line_title("\n\n  hello\nworld") == "hello"
    assert clean_title('<think>x</think>\n"标题：今天的天气。"') == "今天的天气"


# ------------------------------------------------------------ 生成流程
def test_send_streams_and_persists(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    data, job = send(client, app, conv, "你好世界")
    assert job.state == "done"
    types = event_types(job)
    assert "context" in types and "done" in types and types[-1] == "end"

    detail = client.get(f"/api/conversations/{conv}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    user, assistant = detail["messages"]
    assert user["tokens"] == len("你好世界")
    assert assistant["status"] == "done"
    stats = assistant["stats"]
    assert stats["prompt_tokens"] > 0
    assert stats["completion_tokens"] > 0
    assert stats["peak_memory"] > 0
    assert stats["kv_bytes"] > 0
    assert detail["conversation"]["title"] == "你好世界"


def test_second_turn_reuses_kv_cache(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv, "first question here")
    data, job = send(client, app, conv, "second")
    assert job.result_stats["cached_tokens"] > 10
    assert job.result_stats["processed_tokens"] < job.result_stats["prompt_tokens"]


def _thinking_model(client):
    return next(m for m in client.get("/api/models").json()["models"] if m["name"] == "tiny-think")


def test_thinking_template_is_detected(app_ctx):
    client, _, _ = app_ctx
    info = _thinking_model(client)
    assert info["thinking_toggle"] is True and info["reasoning"] is True


def test_thinking_on_restores_open_tag_in_output(app_ctx):
    client, app, _ = app_ctx
    conv = new_conv(client, _thinking_model(client)["id"], max_tokens=6, thinking=True)
    _, job = send(client, app, conv, "hi")
    assert job.state == "done"
    assert job.result_text.startswith("<think>\n")


def test_thinking_off_renders_empty_block_in_prompt(app_ctx):
    client, app, _ = app_ctx
    conv = new_conv(client, _thinking_model(client)["id"], max_tokens=6, thinking=False)
    preview = client.post(
        "/api/context/preview", json={"conversation_id": conv, "draft": "hi", "include_prompt": True}
    ).json()
    assert preview["prompt"].endswith("<think>\n\n</think>\n\n")
    _, job = send(client, app, conv, "hi")
    assert not job.result_text.startswith("<think>")


def test_history_excludes_reasoning(app_ctx):
    client, app, _ = app_ctx
    conv = new_conv(client, _thinking_model(client)["id"], max_tokens=6)
    data, _ = send(client, app, conv, "hi")
    app.state.db.update_message(data["messages"][-1]["id"], content="<think>secret</think>visible answer")
    preview = client.post(
        "/api/context/preview", json={"conversation_id": conv, "draft": "next", "include_prompt": True}
    ).json()
    assert "visible answer" in preview["prompt"]
    assert "secret" not in preview["prompt"]


def test_monitor_reports_real_memory(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv)
    snap = client.get("/api/monitor").json()
    assert snap["engine"]["model_id"] == model
    assert snap["engine"]["weights_bytes"] > 0
    assert snap["engine"]["kv_bytes"] > 0
    assert snap["mlx"]["working_set_limit"] > 0
    assert snap["system"]["total"] > snap["system"]["available"] > 0


def test_regenerate_creates_branch_and_switch_restores(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    data, _ = send(client, app, conv)
    first = data["messages"][-1]["id"]

    res = client.post(f"/api/conversations/{conv}/regenerate", json={"message_id": first}).json()
    wait_job(app, res["job_id"])
    detail = client.get(f"/api/conversations/{conv}").json()
    assistant = detail["messages"][-1]
    assert assistant["id"] != first
    assert len(assistant["siblings"]) == 2 and assistant["sibling_index"] == 1

    restored = client.post(f"/api/conversations/{conv}/switch", json={"message_id": first}).json()
    assert restored["messages"][-1]["id"] == first


def test_edit_forks_user_message(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    data, _ = send(client, app, conv, "original")
    user_id = data["messages"][0]["id"]
    res = client.post(
        f"/api/conversations/{conv}/edit", json={"message_id": user_id, "content": "edited"}
    ).json()
    wait_job(app, res["job_id"])
    detail = client.get(f"/api/conversations/{conv}").json()
    assert detail["messages"][0]["content"] == "edited"
    assert len(detail["messages"][0]["siblings"]) == 2


def test_delete_message_removes_descendants(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv, "a")
    data, _ = send(client, app, conv, "b")
    second_user = data["messages"][2]["id"]
    detail = client.post(
        f"/api/conversations/{conv}/delete-message", json={"message_id": second_user}
    ).json()
    assert len(detail["messages"]) == 2


def test_stop_marks_message_stopped(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=4000)
    res = client.post(f"/api/conversations/{conv}/send", json={"content": "hi"}).json()
    assert client.post(f"/api/jobs/{res['job_id']}/stop").status_code == 200
    job = wait_job(app, res["job_id"])
    assert job.state == "stopped"
    detail = client.get(f"/api/conversations/{conv}").json()
    assert detail["messages"][-1]["status"] == "stopped"


def test_concurrent_send_same_conversation_rejected(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=4000)
    res = client.post(f"/api/conversations/{conv}/send", json={"content": "hi"}).json()
    again = client.post(f"/api/conversations/{conv}/send", json={"content": "x"})
    assert again.status_code == 409
    client.post(f"/api/jobs/{res['job_id']}/stop")
    wait_job(app, res["job_id"])


def test_two_conversations_queue_in_order(app_ctx):
    client, app, model = app_ctx
    a = new_conv(client, model)
    b = new_conv(client, model)
    first = client.post(f"/api/conversations/{a}/send", json={"content": "one"}).json()
    second = client.post(f"/api/conversations/{b}/send", json={"content": "two"}).json()
    job_a = wait_job(app, first["job_id"])
    job_b = wait_job(app, second["job_id"])
    assert job_a.state == "done" and job_b.state == "done"
    assert job_a.finished_at <= job_b.finished_at


def test_context_truncates_oldest_turns(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, context_length=300, max_tokens=8)
    for index in range(4):
        send(client, app, conv, f"{index}" * 60)
    data, job = send(client, app, conv, "latest")
    context = next(e for e in job.slice(0) if e["type"] == "context")
    assert context["excluded"], "oldest turns should have been dropped"
    assert context["total"] <= 300
    latest_user = data["messages"][-2]["id"]
    assert latest_user in context["included"]
    assert job.state == "done"


def test_oversized_message_reports_error(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, context_length=256)
    data, job = send(client, app, conv, "x" * 2000)
    assert job.state == "error"
    detail = client.get(f"/api/conversations/{conv}").json()
    assert detail["messages"][-1]["status"] == "error"
    assert "上下文窗口" in detail["messages"][-1]["error"]


def test_preview_counts_draft_and_budget(app_ctx):
    client, _, model = app_ctx
    conv = new_conv(client, model, system_prompt="sys")
    res = client.post(
        "/api/context/preview", json={"conversation_id": conv, "draft": "hello", "include_prompt": True}
    ).json()
    assert res["breakdown"]["system"] == 3
    assert res["breakdown"]["current"] == 5
    assert res["breakdown"]["template"] > 0
    assert res["estimated"] is False
    assert "<|im_start|>" in res["prompt"]
    assert res["total"] == sum(res["breakdown"].values())


def test_preview_with_empty_draft_matches_next_send(app_ctx):
    """最新一轮本身超过预算时，空输入框的预览不能误报放不下。"""
    client, app, model = app_ctx
    conv = new_conv(client, model, context_length=300, max_tokens=120)
    data, job = send(client, app, conv, "q" * 150)
    assert job.state == "done"
    preview = client.post("/api/context/preview", json={"conversation_id": conv}).json()
    assert preview["over_limit"] is False
    assert preview["total"] <= 300
    assert "__next__" not in preview["included"]


def test_preview_of_empty_conversation_counts_system_prompt(app_ctx):
    client, _, model = app_ctx
    res = client.post(
        "/api/context/preview", json={"model": model, "settings": {"system_prompt": "abc"}}
    ).json()
    assert res["estimated"] is False
    assert res["breakdown"]["system"] == 3
    assert res["included"] == []


def test_preview_without_model_estimates(app_ctx):
    client, _, _ = app_ctx
    res = client.post("/api/context/preview", json={"draft": "你好 hello world"}).json()
    assert res["estimated"] is True
    assert res["total"] > 0


def test_custom_stop_word_is_not_emitted(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=60)
    data, job = send(client, app, conv, "seed")
    baseline = job.result_text
    assert len(baseline) > 4
    stop = baseline[3:5]
    other = new_conv(client, model, max_tokens=60, stop=[stop])
    _, stopped = send(client, app, other, "seed")
    assert stop not in stopped.result_text
    assert stopped.result_stats["finish_reason"] == "stop_sequence"


def test_seed_makes_sampling_repeatable(app_ctx):
    client, app, model = app_ctx
    a = new_conv(client, model, temperature=1.0, seed=7, max_tokens=20)
    b = new_conv(client, model, temperature=1.0, seed=7, max_tokens=20)
    _, ja = send(client, app, a, "same prompt")
    _, jb = send(client, app, b, "same prompt")
    assert ja.result_text == jb.result_text


def test_continue_appends_to_message(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=6)
    data, job = send(client, app, conv)
    assistant_id = data["messages"][-1]["id"]
    before = client.get(f"/api/conversations/{conv}").json()["messages"][-1]["content"]
    res = client.post(f"/api/conversations/{conv}/continue", json={"message_id": assistant_id}).json()
    second = wait_job(app, res["job_id"])
    assert second.state == "done"
    after = client.get(f"/api/conversations/{conv}").json()["messages"][-1]
    assert after["id"] == assistant_id
    assert after["content"].startswith(before) and len(after["content"]) > len(before)


def test_compact_reports_empty_summary(app_ctx, monkeypatch):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=8)
    send(client, app, conv, "first")
    send(client, app, conv, "second")

    def eos_immediately(*args, **kwargs):
        yield SimpleNamespace(
            text="", token=0, generation_tokens=1, prompt_tps=1.0, generation_tps=1.0, finish_reason="stop"
        )

    monkeypatch.setattr("mlxui.engine.stream_generate", eos_immediately)
    res = client.post(f"/api/conversations/{conv}/compact", json={}).json()
    job = wait_job(app, res["job_id"])
    assert [e for e in job.slice(0) if e["type"] == "error"]
    assert client.get(f"/api/conversations/{conv}").json()["conversation"]["summary"] is None


def test_budget_never_looser_than_overflow_check():
    """回复上限很小时，落在 (limit-16, limit-reserve] 的 prompt 也要被截断而不是报错。"""

    class CharTokenizer:
        has_chat_template = False
        bos_token = None

        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf-8"))

    tokenizer = CharTokenizer()
    turns = [
        Turn("a", "user", "x" * 100),
        Turn("b", "assistant", "y" * 10),
        Turn("c", "user", "z" * 139),  # 完整 prompt 恰好 288 tokens
    ]
    result = build_context(
        tokenizer, TokenCounter(tokenizer), system="", summary=None, turns=turns, limit=300, max_tokens=8
    )
    assert result.excluded == ["a", "b"]
    assert result.total <= 284


def test_compact_summarizes_and_replaces_history(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=8, seed=3)
    send(client, app, conv, "first")
    data, _ = send(client, app, conv, "second")
    res = client.post(f"/api/conversations/{conv}/compact", json={}).json()
    job = wait_job(app, res["job_id"])
    assert job.state == "done"
    detail = client.get(f"/api/conversations/{conv}").json()
    assert detail["conversation"]["summary"]
    preview = client.post("/api/context/preview", json={"conversation_id": conv}).json()
    assert preview["breakdown"]["summary"] > 0
    assert data["messages"][0]["id"] not in preview["included"]

    client.post(f"/api/conversations/{conv}/clear-summary")
    preview = client.post("/api/context/preview", json={"conversation_id": conv}).json()
    assert preview["breakdown"]["summary"] == 0


def test_model_switch_reloads(app_ctx):
    client, app, model = app_ctx
    assert client.post("/api/models/load", json={"id": model}).json()["model_id"] == model
    assert client.post("/api/models/unload").json()["model_id"] is None
    assert client.get("/api/monitor").json()["engine"]["weights_bytes"] == 0


def test_missing_model_rejected(app_ctx):
    client, _, _ = app_ctx
    conv = client.post("/api/conversations", json={"model": "nope/nope"}).json()["id"]
    res = client.post(f"/api/conversations/{conv}/send", json={"content": "hi"})
    assert res.status_code == 400


# ---------------------------------------------------------- 对话管理
def test_archive_pin_rename_search_delete(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv, "独一无二的关键词 abc100%")

    client.patch(f"/api/conversations/{conv}", json={"title": "我的标题", "pinned": True})
    item = client.get("/api/conversations").json()["conversations"][0]
    assert item["title"] == "我的标题" and item["pinned"] is True and item["message_count"] == 2

    hits = client.get("/api/search", params={"q": "独一无二"}).json()["results"]
    assert hits[0]["id"] == conv and "独一无二" in hits[0]["snippet"]
    assert client.get("/api/search", params={"q": "abc100%"}).json()["results"]
    assert not client.get("/api/search", params={"q": "abc1000%"}).json()["results"]

    client.patch(f"/api/conversations/{conv}", json={"archived": True})
    assert client.get("/api/conversations").json()["conversations"] == []
    archived = client.get("/api/conversations", params={"archived": True}).json()["conversations"]
    assert archived[0]["id"] == conv and archived[0]["pinned"] is False

    client.patch(f"/api/conversations/{conv}", json={"archived": False})
    assert len(client.get("/api/conversations").json()["conversations"]) == 1

    assert client.delete(f"/api/conversations/{conv}").json()["ok"] is True
    assert client.get(f"/api/conversations/{conv}").status_code == 404


def test_bulk_actions(app_ctx):
    client, _, model = app_ctx
    ids = [new_conv(client, model) for _ in range(3)]
    assert client.post("/api/conversations/bulk", json={"ids": ids[:2], "action": "archive"}).json()["count"] == 2
    assert len(client.get("/api/conversations").json()["conversations"]) == 1
    assert client.post("/api/conversations/bulk", json={"ids": ids, "action": "bogus"}).status_code == 400


def test_title_rename_survives_user_edit_flag(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    client.patch(f"/api/conversations/{conv}", json={"title": "手动"})
    send(client, app, conv, "不应该覆盖标题")
    assert client.get(f"/api/conversations/{conv}").json()["conversation"]["title"] == "手动"


def test_export_import_roundtrip_with_branches(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    data, _ = send(client, app, conv, "往返测试")
    res = client.post(
        f"/api/conversations/{conv}/regenerate", json={"message_id": data["messages"][-1]["id"]}
    ).json()
    wait_job(app, res["job_id"])

    markdown = client.get(f"/api/conversations/{conv}/export", params={"format": "md"})
    assert "往返测试" in markdown.text
    assert "filename*=UTF-8''" in markdown.headers["content-disposition"]

    raw = client.get("/api/export").content
    assert client.post("/api/import", content=raw).json()["count"] == 1

    items = client.get("/api/conversations").json()["conversations"]
    assert len(items) == 2
    imported = next(i for i in items if i["id"] != conv)
    detail = client.get(f"/api/conversations/{imported['id']}").json()
    assert len(detail["messages"]) == 2
    assert len(detail["messages"][-1]["siblings"]) == 2

    bad = client.post("/api/import", content=b'{"a":1}')
    assert bad.status_code == 400


def test_prompts_crud(app_ctx):
    client, _, _ = app_ctx
    seeded = client.get("/api/prompts").json()["prompts"]
    assert seeded
    created = client.post("/api/prompts", json={"name": "n", "content": "c"}).json()
    client.post("/api/prompts", json={"id": created["id"], "name": "n2", "content": "c2"})
    listed = client.get("/api/prompts").json()["prompts"]
    assert any(p["name"] == "n2" for p in listed) and len(listed) == len(seeded) + 1
    client.delete(f"/api/prompts/{created['id']}")
    assert len(client.get("/api/prompts").json()["prompts"]) == len(seeded)


def test_settings_validation(app_ctx):
    client, _, _ = app_ctx
    assert client.patch("/api/settings", json={"hf_endpoint": "ftp nonsense"}).status_code == 400
    updated = client.patch(
        "/api/settings", json={"params": {"temperature": 5}, "kv_cache_limit_mb": -1, "ui": {"theme": "dark"}}
    ).json()
    assert updated["params"]["temperature"] == 2.0
    assert updated["params"]["top_p"] == 0.95
    assert updated["kv_cache_limit_mb"] == 0
    assert updated["ui"]["theme"] == "dark" and updated["ui"]["send_key"] == "enter"


def test_kv_cache_disabled_by_zero_limit(app_ctx):
    client, app, model = app_ctx
    client.patch("/api/settings", json={"kv_cache_limit_mb": 0})
    conv = new_conv(client, model)
    send(client, app, conv, "a")
    _, job = send(client, app, conv, "b")
    assert job.result_stats["cached_tokens"] == 0


def test_conversation_setting_can_be_reset(app_ctx):
    client, _, model = app_ctx
    conv = new_conv(client, model, temperature=1.5)
    detail = client.patch(f"/api/conversations/{conv}", json={"settings": {"temperature": None}}).json()
    assert detail["params"]["temperature"] == 0.7


# -------------------------------------------------------------- 附件
def test_attachment_goes_into_context(app_ctx):
    client, app, model = app_ctx
    uploaded = client.post("/api/attachments", params={"name": "a.txt"}, content="附件内容".encode()).json()
    assert uploaded["text"] == "附件内容"
    conv = new_conv(client, model)
    res = client.post(
        f"/api/conversations/{conv}/send", json={"content": "看附件", "attachments": [uploaded]}
    ).json()
    job = wait_job(app, res["job_id"])
    context = next(e for e in job.slice(0) if e["type"] == "context")
    assert context["breakdown"]["attachments"] > 0


def test_attachment_rejects_image_and_binary(app_ctx):
    client, _, _ = app_ctx
    assert client.post("/api/attachments", params={"name": "a.png"}, content=b"x").status_code == 415
    assert client.post("/api/attachments", params={"name": "a.bin"}, content=b"\x00\x01").status_code == 415


def test_attachment_decodes_gbk(app_ctx):
    client, _, _ = app_ctx
    res = client.post("/api/attachments", params={"name": "g.txt"}, content="中文内容".encode("gbk")).json()
    assert res["text"] == "中文内容"


# ------------------------------------------------------ OpenAI 兼容接口
def test_openai_non_stream(app_ctx):
    client, _, model = app_ctx
    res = client.post(
        "/v1/chat/completions",
        json={"model": model, "max_tokens": 6, "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert res["object"] == "chat.completion"
    assert res["usage"]["completion_tokens"] > 0


def test_openai_stream(app_ctx):
    client, _, model = app_ctx
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": model, "max_tokens": 6, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as res:
        lines = [line for line in res.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[-2][6:])["choices"][0]["finish_reason"] in ("stop", "length")


def test_openai_models_and_errors(app_ctx):
    client, _, model = app_ctx
    assert client.get("/v1/models").json()["data"][0]["id"] == model
    res = client.post("/v1/chat/completions", json={"model": model, "messages": []})
    assert res.status_code == 400


# -------------------------------------------------------------- 安全
def test_guard_rejects_cross_site_and_rebinding(tiny_model_dir, tmp_path):
    from fastapi.testclient import TestClient

    from mlxui.app import create_app

    app = create_app(allow_any_host=False, db_path=tmp_path / "guard.db", keystore=MemoryKeyStore())
    with TestClient(app, base_url="http://127.0.0.1:8420") as client:
        assert client.get("/api/settings").status_code == 200
        assert client.get("/api/settings", headers={"host": "evil.com"}).status_code == 403
        forged = client.post(
            "/api/conversations", json={}, headers={"origin": "https://evil.com"}
        )
        assert forged.status_code == 403
        same = client.post(
            "/api/conversations", json={}, headers={"origin": "http://127.0.0.1:8420"}
        )
        assert same.status_code == 200


def test_delete_model_refuses_external_directory(app_ctx):
    client, _, model = app_ctx
    assert client.post("/api/models/delete", json={"id": model}).status_code == 400
    assert client.post("/api/models/delete", json={"id": "../../etc"}).status_code == 404


def test_download_rejects_bad_repo_name(app_ctx):
    client, _, _ = app_ctx
    assert client.post("/api/downloads", json={"repo": "../evil"}).status_code == 400
