"""反思记忆：解析、注入、调度、来源追溯、管理接口。"""
from __future__ import annotations

import time

import pytest
from conftest import wait_job

from mlxui.engine import Preempted
from mlxui.memory import grams, looks_sensitive, parse_operations, similarity


def new_conv(client, model, **settings):
    base = {"max_tokens": 6, "temperature": 0.0, "context_length": 2048}
    return client.post("/api/conversations", json={"model": model, "settings": {**base, **settings}}).json()["id"]


def send(client, app, conv, text):
    res = client.post(f"/api/conversations/{conv}/send", json={"content": text}).json()
    return res, wait_job(app, res["job_id"])


@pytest.fixture()
def ctx(app_ctx):
    client, app, model = app_ctx
    client.patch("/api/settings", json={"memory": {"enabled": True, "idle_seconds": 10}})
    return client, app, model, app.state.memory, app.state.engine


def script(engine, *outputs):
    """让模型按脚本输出，并记录每次收到的提示。"""
    queue = list(outputs)
    seen: list[str] = []
    original_tokenizer = engine.tokenizer

    def fake(tokens, max_tokens, temperature=0.3, preemptible=False):
        tokenizer = engine.tokenizer or original_tokenizer
        seen.append(tokenizer.decode(tokens))
        return queue.pop(0) if queue else "NONE"

    engine.generate_once = fake
    return seen


def reflect(ctx, conv, **kwargs):
    _, _, _, memory, engine = ctx
    return engine.submit(lambda: memory.reflect_conversation(engine, conv, **kwargs)).result(timeout=30)


def consolidate(ctx):
    _, _, _, memory, engine = ctx
    return engine.submit(lambda: memory.consolidate(engine, preemptible=False)).result(timeout=30)


def active(memory):
    return {m["content"]: m for m in memory.list("active")}


LONG = "我最近在准备搬家，下周三要去面试，平时喜欢喝不加糖的美式咖啡"


# ---------------------------------------------------------------- 解析
def test_parse_standard_lines():
    ops = parse_operations("ADD preference 3: 用户喜欢美式\nUPDATE m2: 用户搬到了杭州\nDELETE m5\n乱七八糟的解释")
    assert [o.action for o in ops] == ["add", "update", "delete"]
    assert ops[0].kind == "preference" and ops[0].importance == 3
    assert ops[1].refs == ["m2"] and ops[2].refs == ["m5"]


def test_parse_tolerates_small_model_formatting():
    ops = parse_operations("<think>想想</think>\n- ADD 偏好 4：用户怕冷\n1. add event 2: 用户下周面试 | 10天\n`DELETE M3`")
    assert ops[0].kind == "preference" and ops[0].content == "用户怕冷"
    assert ops[1].kind == "event" and ops[1].expires_days == 10 and ops[1].content == "用户下周面试"
    assert ops[2].refs == ["m3"]


@pytest.mark.parametrize(
    "line",
    ["ADD nonsense 3: x", "ADD profile 9: x", "ADD profile 3: 内容", "NONE", "ADD insight 3: 模型不能直接写反思"],
)
def test_parse_rejects_invalid_lines(line):
    assert parse_operations(line) == []


def test_insight_needs_two_references():
    assert parse_operations("INSIGHT 4 (m1): 只有一条依据") == []
    assert parse_operations("INSIGHT 4: 没写依据") == []
    ops = parse_operations("INSIGHT 4 (m1,m3): 用户压力大")
    assert ops[0].refs == ["m1", "m3"] and ops[0].kind == "insight"


@pytest.mark.parametrize(
    "text",
    [
        "用户手机号是 13812345678",
        "用户的身份证 11010519900307123X",
        "用户银行卡 6222020200112233445",
        "用户邮箱 someone@example.com",
        "用户的 key 是 sk-abcdefghijklmnop",
        "用户的密码是 hunter2",
        "用户的 password: hunter2",
    ],
)
def test_sensitive_content_is_never_stored(text):
    assert looks_sensitive(text)
    assert parse_operations(f"ADD profile 3: {text}") == []


def test_ordinary_content_is_not_flagged():
    assert not looks_sensitive("用户在 2026 年 3 月搬到杭州，养了一只猫")


def test_similarity_handles_chinese_and_english():
    assert similarity("用户喜欢喝美式咖啡", "用户喜欢喝美式咖啡。") == 1.0
    assert similarity("用户喜欢喝美式咖啡", "用户养了一只橘猫") < 0.3
    assert "coffee" in grams("I like Coffee")


# ---------------------------------------------------------------- 提取
def test_reflection_creates_memory_with_provenance(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    prompts = script(engine, "ADD preference 3: 用户喜欢不加糖的美式\nADD event 3: 用户下周三要面试 | 10")
    assert reflect(ctx, conv) == {"added": 2, "updated": 0, "removed": 0}
    assert LONG in prompts[0]

    items = active(memory)
    assert set(items) == {"用户喜欢不加糖的美式", "用户下周三要面试"}
    event = items["用户下周三要面试"]
    assert event["expires_at"] > time.time() + 9 * 86400
    listed = client.get("/api/memory").json()["memories"][0]
    assert listed["sources"][0]["conversation_id"] == conv


def test_already_reflected_messages_are_not_reread(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    prompts = script(engine, "NONE", "NONE")
    reflect(ctx, conv)
    assert reflect(ctx, conv) is None
    assert len(prompts) == 1
    assert memory._due(ignore_idle=True) == []


def test_only_new_messages_are_sent_next_time(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "NONE")
    reflect(ctx, conv)
    prompts = script(engine, "NONE")
    send(client, app, conv, "今天带猫去看了医生，它叫豆豆，有点担心")
    reflect(ctx, conv)
    assert "豆豆" in prompts[0] and "美式" not in prompts[0]


def test_short_small_talk_is_skipped(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, "嗨")
    prompts = script(engine)
    assert reflect(ctx, conv) is None
    assert prompts == []


def test_reflection_ignores_attachment_content(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    uploaded = client.post("/api/attachments", params={"name": "a.txt"}, content="附件里的机密内容".encode()).json()
    res = client.post(f"/api/conversations/{conv}/send", json={"content": LONG, "attachments": [uploaded]}).json()
    wait_job(app, res["job_id"])
    prompts = script(engine, "NONE")
    reflect(ctx, conv)
    assert "机密" not in prompts[0]


def test_duplicate_becomes_extra_evidence(ctx):
    client, app, model, memory, engine = ctx
    a, b = new_conv(client, model), new_conv(client, model)
    send(client, app, a, LONG)
    send(client, app, b, LONG + "真的")
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡", "ADD preference 3: 用户喜欢喝美式咖啡。")
    reflect(ctx, a)
    reflect(ctx, b)
    items = client.get("/api/memory").json()["memories"]
    assert len(items) == 1 and len(items[0]["sources"]) == 2


def test_update_and_delete_operate_on_existing(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD profile 3: 用户住在北京\nADD preference 2: 用户喜欢下雨天")
    reflect(ctx, conv)
    send(client, app, conv, "我其实上个月搬到杭州了，而且我不喜欢下雨天")
    prompts = script(engine, "UPDATE m1: 用户住在杭州\nDELETE m2")
    reflect(ctx, conv)
    # 编号顺序由相关性决定，这里只验证结果里没有旧内容
    assert "m1" in prompts[0]
    assert len(active(memory)) == 1


def test_model_cannot_touch_user_edited_or_pinned(ctx):
    client, app, model, memory, engine = ctx
    locked = memory.add_manual("profile", "用户叫小明")
    pinned = memory.add_manual("preference", "用户喜欢猫")
    memory.update(pinned["id"], {"pinned": True})
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "UPDATE m1: 被模型改掉\nDELETE m2\nDELETE m1")
    reflect(ctx, conv)
    assert set(active(memory)) == {locked["content"], pinned["content"]}


def test_review_mode_holds_new_memories(ctx):
    client, app, model, memory, engine = ctx
    client.patch("/api/settings", json={"memory": {"review": True}})
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    assert active(memory) == {}
    pending = memory.list("pending")
    assert pending[0]["content"] == "用户喜欢喝美式咖啡"

    preview = client.post("/api/context/preview", json={"conversation_id": conv, "include_prompt": True}).json()
    assert "用户喜欢喝美式咖啡" not in preview["prompt"]

    client.post("/api/memory/bulk", json={"ids": [pending[0]["id"]], "action": "approve"})
    assert "用户喜欢喝美式咖啡" in active(memory)


def test_review_mode_blocks_model_edits(ctx):
    client, app, model, memory, engine = ctx
    existing = memory.add_manual("profile", "用户住在北京")
    with app.state.db.conn as c:
        c.execute("UPDATE memories SET locked=0 WHERE id=?", (existing["id"],))
    client.patch("/api/settings", json={"memory": {"review": True}})
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "DELETE m1\nUPDATE m1: 改掉")
    reflect(ctx, conv)
    assert "用户住在北京" in active(memory)


def test_explicit_request_skips_idle_wait(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, "帮我记住我对花生过敏")
    assert memory._due() == [conv]
    other = new_conv(client, model)
    send(client, app, other, "随便聊聊今天的天气")
    assert other not in memory._due(limit=5)


def test_due_after_idle_period(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    assert memory._due() == []
    app.state.db.update_conversation(conv, updated_at=time.time() - 3600)
    assert memory._due() == [conv]


def test_conversation_can_opt_out(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model, memory=False)
    send(client, app, conv, LONG)
    app.state.db.update_conversation(conv, updated_at=time.time() - 3600)
    assert memory._due() == []


def test_idle_tick_reflects_only_when_model_is_loaded(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    app.state.db.update_conversation(conv, updated_at=time.time() - 3600)
    script(engine, "ADD preference 3: 用户喜欢喝美式")
    engine.submit(lambda: memory.idle_tick(engine)).result(timeout=30)
    assert len(active(memory)) == 1

    client.post("/api/models/unload")
    other = new_conv(client, model)
    app.state.db.add_message(other, None, "user", LONG)
    app.state.db.update_conversation(other, updated_at=time.time() - 3600)
    calls = script(engine)
    engine.submit(lambda: memory.idle_tick(engine)).result(timeout=30)
    assert calls == []


def test_failed_reflection_backs_off(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    app.state.db.update_conversation(conv, updated_at=time.time() - 3600)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    engine.generate_once = explode
    engine.submit(lambda: memory.idle_tick(engine)).result(timeout=30)
    assert memory._due() == []


def test_generate_once_yields_to_user_requests(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, "x")

    def task():
        engine._queue.put(lambda: None)  # 模拟用户在反思途中发来新消息
        return engine.generate_once([1, 2, 3, 4], 20, preemptible=True)

    with pytest.raises(Preempted):
        engine.submit(task).result(timeout=30)


def test_preempted_reflection_is_retried(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    app.state.db.update_conversation(conv, updated_at=time.time() - 3600)

    def interrupted(*args, **kwargs):
        raise Preempted()

    engine.generate_once = interrupted
    engine.submit(lambda: memory.idle_tick(engine)).result(timeout=30)
    assert memory._due() == [conv]
    assert engine.state == "idle"


def test_manual_reflect_endpoint(ctx):
    client, app, model, memory, engine = ctx
    client.patch("/api/settings", json={"default_model": model})
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式")
    assert client.post("/api/memory/reflect").json()["added"] == 1


# ---------------------------------------------------------------- 注入
def prompt_of(client, conv, draft="今天聊点什么"):
    return client.post(
        "/api/context/preview", json={"conversation_id": conv, "draft": draft, "include_prompt": True}
    ).json()


def test_core_memory_goes_into_system(ctx):
    client, _, model, memory, _ = ctx
    memory.add_manual("profile", "用户叫小明")
    conv = new_conv(client, model)
    result = prompt_of(client, conv)
    system_part = result["prompt"].split("<|im_end|>")[0]
    assert "用户叫小明" in system_part
    assert result["breakdown"]["memory"] > 0
    assert result["total"] == sum(result["breakdown"].values())


def test_relevant_memory_is_recalled_near_the_question(ctx):
    client, _, model, memory, _ = ctx
    memory.add_manual("person", "用户的妹妹叫小雨，在成都读大学")
    memory.add_manual("preference", "用户喜欢爬山和露营")
    conv = new_conv(client, model)
    result = prompt_of(client, conv, "我妹妹小雨最近要考试")
    system_part, rest = result["prompt"].split("<|im_end|>", 1) if "system" in result["prompt"] else ("", result["prompt"])
    assert "小雨，在成都" in rest and "小雨，在成都" not in system_part
    assert "爬山" not in result["prompt"]


def test_budget_is_respected(ctx):
    client, _, model, memory, _ = ctx
    client.patch("/api/settings", json={"memory": {"budget_tokens": 100}})
    for index in range(40):
        memory.add_manual("profile", f"用户的第{index}件事情，内容比较长一些用来占位置")
    conv = new_conv(client, model)
    assert prompt_of(client, conv)["breakdown"]["memory"] <= 100


def test_budget_scales_with_small_context(ctx):
    client, _, model, memory, _ = ctx
    for index in range(40):
        memory.add_manual("profile", f"用户的第{index}件事情，内容比较长一些用来占位置")
    conv = new_conv(client, model, context_length=512, max_tokens=64)
    assert prompt_of(client, conv)["breakdown"]["memory"] <= 512 // 5


def test_memory_can_be_disabled_globally_and_per_conversation(ctx):
    client, _, model, memory, _ = ctx
    memory.add_manual("profile", "用户叫小明")
    off = new_conv(client, model, memory=False)
    assert "小明" not in prompt_of(client, off)["prompt"]
    on = new_conv(client, model)
    assert "小明" in prompt_of(client, on)["prompt"]
    client.patch("/api/settings", json={"memory": {"enabled": False}})
    assert "小明" not in prompt_of(client, on)["prompt"]


def test_expired_and_archived_are_not_used(ctx):
    client, app, model, memory, _ = ctx
    gone = memory.add_manual("event", "用户明天要去体检")
    archived = memory.add_manual("profile", "用户曾经住在深圳")
    memory.update(archived["id"], {"status": "archived"})
    with app.state.db.conn as c:
        c.execute("UPDATE memories SET expires_at=? WHERE id=?", (time.time() - 1, gone["id"]))
    conv = new_conv(client, model)
    prompt = prompt_of(client, conv, "体检 深圳")["prompt"]
    assert "体检" not in prompt.replace("体检 深圳", "") and "曾经住在" not in prompt


def test_expiry_moves_status(ctx):
    _, app, _, memory, _ = ctx
    gone = memory.add_manual("event", "用户明天要去体检")
    with app.state.db.conn as c:
        c.execute("UPDATE memories SET expires_at=? WHERE id=?", (time.time() - 1, gone["id"]))
    memory._last_expiry_check = 0
    memory._expire()
    assert memory.get(gone["id"])["status"] == "expired"


def test_unused_trivia_fades_but_not_pinned_or_manual(ctx):
    _, app, _, memory, _ = ctx
    now = time.time()
    old = now - 90 * 86400
    with app.state.db.conn as c:
        for mid, importance, pinned, locked in [("a", 1, 0, 0), ("b", 5, 0, 0), ("c", 1, 1, 0), ("d", 1, 0, 1)]:
            c.execute(
                "INSERT INTO memories(id,kind,content,importance,pinned,locked,created_at,updated_at) "
                "VALUES(?,'preference',?,?,?,?,?,?)",
                (mid, f"用户{mid}", importance, pinned, locked, old, old),
            )
    memory._last_expiry_check = 0
    memory._expire()
    assert memory.get("a")["status"] == "archived"
    assert {memory.get(x)["status"] for x in "bcd"} == {"active"}


def test_core_block_is_stable_as_memories_get_used(ctx):
    client, app, model, memory, _ = ctx
    a = memory.add_manual("profile", "用户叫小明")
    memory.add_manual("profile", "用户在杭州工作")
    conv = new_conv(client, model)
    first = prompt_of(client, conv)["prompt"].split("<|im_end|>")[0]
    memory.update(a["id"], {"importance": 5})
    memory.touch_used([a["id"]])
    assert prompt_of(client, conv)["prompt"].split("<|im_end|>")[0] == first


def test_kv_cache_still_hits_with_memory(ctx):
    client, app, model, memory, _ = ctx
    memory.add_manual("profile", "用户叫小明")
    memory.add_manual("person", "用户的妹妹叫小雨")
    conv = new_conv(client, model)
    send(client, app, conv, "第一个问题，内容长一点方便观察缓存命中的情况")
    _, job = send(client, app, conv, "小雨最近怎么样")
    assert job.result_stats["cached_tokens"] > 50


def test_recall_is_not_persisted_in_history(ctx):
    client, app, model, memory, _ = ctx
    memory.add_manual("person", "用户的妹妹叫小雨，在成都读大学")
    conv = new_conv(client, model)
    send(client, app, conv, "我妹妹小雨最近要考试")
    detail = client.get(f"/api/conversations/{conv}").json()
    assert detail["messages"][0]["content"] == "我妹妹小雨最近要考试"
    prompt = prompt_of(client, conv, "完全无关的话题")["prompt"]
    assert prompt.count("在成都") == 1, "联想内容只出现在最后一轮，不会累积进历史"
    assert prompt.index("在成都") > prompt.rindex("我妹妹小雨最近要考试")


def test_used_memories_are_tracked_by_id_only(ctx):
    client, app, model, memory, _ = ctx
    item = memory.add_manual("profile", "用户叫小明")
    conv = new_conv(client, model)
    data, job = send(client, app, conv, "你好啊")
    assert job.result_stats["memory"] == [item["id"]]
    assert memory.get(item["id"])["use_count"] == 1
    stored = client.get(f"/api/conversations/{conv}").json()["messages"][-1]["stats"]
    assert stored["memory"] == [item["id"]]
    assert "小明" not in str(stored)


def test_openai_api_never_sees_memory(ctx):
    client, _, model, memory, engine = ctx
    memory.add_manual("profile", "用户叫小明")
    res = client.post(
        "/v1/chat/completions",
        json={"model": model, "max_tokens": 3, "messages": [{"role": "user", "content": "你好"}]},
    )
    assert res.status_code == 200
    assert memory.get(memory.list()[0]["id"])["use_count"] == 0


# ---------------------------------------------------------------- 反思
def seed(ctx, conv, lines):
    _, _, _, _, engine = ctx
    script(engine, "\n".join(f"ADD preference 3: {line}" for line in lines))
    reflect(ctx, conv)


def test_insight_inherits_sources_and_is_derived(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    seed(ctx, conv, ["用户最近睡得很晚", "用户工作压力大", "用户周末也在加班", "用户想换工作"])
    script(engine, "INSIGHT 4 (m1,m2): 用户最近过得挺累，聊天时多一点关心")
    assert consolidate(ctx)["insights"] == 1
    insight = memory.list("active", "insight")[0]
    assert insight["derived"] is True and insight["sources"][0]["conversation_id"] == conv


def test_insight_with_unknown_references_is_dropped(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    seed(ctx, conv, ["用户最近睡得很晚", "用户工作压力大", "用户周末也在加班", "用户想换工作"])
    script(engine, "INSIGHT 4 (m98,m99): 凭空编造的结论")
    consolidate(ctx)
    assert memory.list("active", "insight") == []


def test_manual_only_memories_cannot_ground_an_insight(ctx):
    """没有对话来源就无法追溯，这样的反思不该产生。"""
    client, _, model, memory, engine = ctx
    client.post("/api/models/load", json={"id": model})
    for text in ["用户喜欢猫", "用户喜欢狗", "用户喜欢鸟", "用户喜欢鱼"]:
        memory.add_manual("preference", text)
    script(engine, "INSIGHT 4 (m1,m2): 用户喜欢动物")
    consolidate(ctx)
    assert memory.list("active", "insight") == []


def test_merge_replaces_originals(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    seed(ctx, conv, ["用户喜欢喝咖啡", "用户每天早上一杯美式", "用户工作压力大", "用户想换工作"])
    script(engine, "MERGE (m1,m2): 用户每天早上喝一杯美式咖啡")
    assert consolidate(ctx)["merged"] == 1
    assert len(active(memory)) == 3


def test_consolidation_triggers_after_enough_new_memories(ctx):
    client, app, model, memory, engine = ctx
    assert not memory._needs_consolidation()
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    seed(ctx, conv, ["用户喜欢爬山", "用户养了一只橘猫", "用户在杭州上班", "用户每天跑步五公里", "用户害怕坐飞机", "用户最爱吃火锅"])
    assert memory._needs_consolidation()
    script(engine, "NONE")
    consolidate(ctx)
    assert not memory._needs_consolidation()


# ------------------------------------------------------------- 来源追溯
def test_deleting_only_source_removes_memory(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    response = client.delete(f"/api/conversations/{conv}").json()
    assert response["memories_removed"] == 1
    assert active(memory) == {}
    assert app.state.db.conn.execute("SELECT COUNT(*) n FROM memory_sources").fetchone()["n"] == 0
    assert app.state.db.conn.execute("SELECT COUNT(*) n FROM reflections").fetchone()["n"] == 0


def test_memory_with_another_source_survives(ctx):
    client, app, model, memory, engine = ctx
    a, b = new_conv(client, model), new_conv(client, model)
    send(client, app, a, LONG)
    send(client, app, b, LONG + "嗯")
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡", "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, a)
    reflect(ctx, b)
    client.delete(f"/api/conversations/{a}")
    items = client.get("/api/memory").json()["memories"]
    assert len(items) == 1 and [s["conversation_id"] for s in items[0]["sources"]] == [b]


def test_insight_is_removed_when_any_source_goes(ctx):
    client, app, model, memory, engine = ctx
    a, b = new_conv(client, model), new_conv(client, model)
    send(client, app, a, LONG)
    send(client, app, b, "我最近工作压力特别大，经常睡不好觉")
    seed(ctx, a, ["用户最近在搬家", "用户下周要面试"])
    seed(ctx, b, ["用户工作压力大", "用户经常失眠"])
    ids = sorted(active(memory))
    assert len(ids) == 4

    def all_refs(tokens, max_tokens, temperature=0.3, preemptible=False):
        return "INSIGHT 4 (m1,m2,m3,m4): 用户这阵子过得很累"

    engine.generate_once = all_refs
    consolidate(ctx)
    assert len(memory.list("active", "insight")) == 1

    client.delete(f"/api/conversations/{a}")
    assert memory.list("active", "insight") == []
    assert set(active(memory)) == {"用户工作压力大", "用户经常失眠"}


def test_manual_memory_is_unaffected_by_deletion(ctx):
    client, app, model, memory, engine = ctx
    memory.add_manual("profile", "用户叫小明")
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    client.delete(f"/api/conversations/{conv}")
    assert "用户叫小明" in active(memory)


def test_edited_memory_still_goes_with_its_source(ctx):
    """用户改过的提炼记忆，来源对话被删时同样要清除，否则就留下了痕迹。"""
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    item = memory.list("active")[0]
    memory.update(item["id"], {"content": "用户只喝美式"})
    client.delete(f"/api/conversations/{conv}")
    assert active(memory) == {}


def test_deleting_message_in_batch_removes_derived_memory(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    data, _ = send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    client.post(f"/api/conversations/{conv}/delete-message", json={"message_id": data["messages"][0]["id"]})
    assert active(memory) == {}


def test_deleting_later_message_keeps_earlier_memory(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    data, _ = send(client, app, conv, "另外一件没有被反思过的事")
    client.post(f"/api/conversations/{conv}/delete-message", json={"message_id": data["messages"][2]["id"]})
    assert "用户喜欢喝美式咖啡" in active(memory)


def test_clear_all_removes_everything(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    client.post("/api/memory/clear")
    conn = app.state.db.conn
    for table in ("memories", "memory_sources", "reflections"):
        assert conn.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"] == 0


# ---------------------------------------------------------------- 管理
def test_crud_api(ctx):
    client, _, _, memory, _ = ctx
    created = client.post("/api/memory", json={"kind": "profile", "content": "用户叫小明", "importance": 4}).json()
    assert created["origin"] == "manual" and created["locked"] is True
    assert client.post("/api/memory", json={"kind": "bogus", "content": "x"}).status_code == 400
    assert client.post("/api/memory", json={"kind": "profile", "content": "  "}).status_code == 400

    patched = client.patch(f"/api/memory/{created['id']}", json={"content": "用户叫大明", "pinned": True}).json()
    assert patched["content"] == "用户叫大明" and patched["pinned"] is True
    assert client.patch("/api/memory/none", json={"pinned": True}).status_code == 404

    client.patch(f"/api/memory/{created['id']}", json={"status": "archived"})
    data = client.get("/api/memory", params={"status": "archived"}).json()
    assert len(data["memories"]) == 1 and data["counts"]["status"]["archived"] == 1

    assert client.delete(f"/api/memory/{created['id']}").json() == {"ok": True}
    assert client.get("/api/memory", params={"status": "archived"}).json()["memories"] == []


def test_filter_by_kind_and_text(ctx):
    client, _, _, memory, _ = ctx
    memory.add_manual("profile", "用户叫小明")
    memory.add_manual("preference", "用户喜欢100%的黑巧")
    assert len(client.get("/api/memory", params={"kind": "profile"}).json()["memories"]) == 1
    assert len(client.get("/api/memory", params={"q": "100%"}).json()["memories"]) == 1
    assert client.get("/api/memory", params={"q": "1000%"}).json()["memories"] == []


def test_user_edit_locks_memory(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    item = memory.list("active")[0]
    assert item["locked"] is False
    assert memory.update(item["id"], {"importance": 5})["locked"] is False
    assert memory.update(item["id"], {"content": "用户只喝美式"})["locked"] is True


def test_seen_flag_and_status(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    snapshot = client.get("/api/monitor").json()["memory"]
    assert snapshot["unseen"] == 1 and snapshot["version"] >= 1
    assert client.post("/api/memory/seen").json()["unseen"] == 0


def test_settings_validation(ctx):
    client, _, _, _, _ = ctx
    updated = client.patch("/api/settings", json={"memory": {"budget_tokens": 1, "idle_seconds": 1}}).json()
    assert updated["memory"]["budget_tokens"] == 50 and updated["memory"]["idle_seconds"] == 10


def test_backup_roundtrip_keeps_memory_links(ctx):
    client, app, model, memory, engine = ctx
    conv = new_conv(client, model)
    send(client, app, conv, LONG)
    script(engine, "ADD preference 3: 用户喜欢喝美式咖啡")
    reflect(ctx, conv)
    memory.add_manual("profile", "用户叫小明")

    backup = client.get("/api/export").content
    client.post("/api/memory/clear")
    client.delete(f"/api/conversations/{conv}")
    assert client.post("/api/import", content=backup).json()["count"] == 1

    items = {m["content"]: m for m in client.get("/api/memory").json()["memories"]}
    assert set(items) == {"用户喜欢喝美式咖啡", "用户叫小明"}
    new_conv_id = items["用户喜欢喝美式咖啡"]["sources"][0]["conversation_id"]
    assert new_conv_id != conv
    client.delete(f"/api/conversations/{new_conv_id}")
    assert "用户喜欢喝美式咖啡" not in active(memory)


def test_single_conversation_export_excludes_memory(ctx):
    client, _, model, memory, _ = ctx
    memory.add_manual("profile", "用户叫小明")
    conv = new_conv(client, model)
    exported = client.get(f"/api/conversations/{conv}/export", params={"format": "json"}).text
    assert "小明" not in exported
