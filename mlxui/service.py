"""对话业务逻辑：发送、重新生成、编辑、继续、压缩、标题生成。"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from typing import Any

from . import websearch
from .config import normalize_params
from .context import (
    ContextResult,
    TokenCounter,
    Turn,
    build_context,
    encode_prompt,
    estimate_context,
    estimate_tokens,
    render_prompt,
    strip_thinking,
)
from .db import DB
from .engine import Engine, Job
from .memory import Composed, MemoryService
from .models import ModelRegistry

FIRST_LINE_LIMIT = 30
NEXT_TURN = "__next__"
SEARCH_BUDGET_TOKENS = 2500
SEARCH_QUERY_INSTRUCTION = (
    "今天是 {today}。根据下面的对话，写出最适合发给搜索引擎的一行关键词，"
    "用来回答用户最后的问题。只输出关键词，不要解释，不要写任何用户的个人信息。\n\n{dialog}"
)
TITLE_INSTRUCTION = (
    "根据下面这段对话，起一个不超过 12 个字的简短标题。"
    "使用对话本身的语言，只输出标题，不要标点、引号或解释。"
)
COMPACT_INSTRUCTION = (
    "请把下面的对话压缩成一份摘要，供后续对话继续使用。"
    "保留：用户的目标与偏好、已确认的事实和决定、重要的代码片段与数字、尚未解决的问题。"
    "省略寒暄和重复内容。使用对话本身的语言，只输出摘要。\n\n"
)


class Busy(Exception):
    pass


class BadRequest(Exception):
    pass


class NotFound(Exception):
    pass


def first_line_title(text: str) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return "新对话"
    return line[:FIRST_LINE_LIMIT] + ("…" if len(line) > FIRST_LINE_LIMIT else "")


def clean_title(raw: str) -> str:
    text = strip_thinking(raw).strip()
    text = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    decoration = " \t\"'“”‘’《》「」*#`.。!！?？:："
    text = re.sub(r"^(标题|Title)\s*[:：]\s*", "", text.strip(decoration), flags=re.I)
    return text.strip(decoration)[:FIRST_LINE_LIMIT]


class MessageSink:
    """把生成进度写回某条 assistant 消息。"""

    def __init__(self, db: DB, conversation_id: str, message_id: str, prefix: str = ""):
        self.db = db
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.prefix = prefix

    def on_context(self, context: ContextResult) -> None:
        # 只有 user 消息的 token 在这里确定，assistant 的来自生成统计
        known = {k: v for k, v in context.counts.items() if k != "__continue__"}
        self.db.set_token_counts(known)
        self.db.update_message(self.message_id, status="streaming")

    def on_progress(self, text: str) -> None:
        self.db.update_message(self.message_id, content=self.prefix + text)

    def on_finish(self, text: str, stats: dict[str, Any], status: str) -> None:
        self.db.update_message(
            self.message_id,
            content=self.prefix + text,
            status=status,
            stats=stats,
            tokens=stats.get("completion_tokens"),
        )
        self.db.touch(self.conversation_id)
        if stats.get("completion_tokens") is not None:
            self.db.add_usage(
                stats.get("model", ""),
                stats.get("processed_tokens", 0),
                stats.get("cached_tokens", 0),
                stats.get("completion_tokens", 0),
                stats.get("duration", 0.0),
            )

    def on_error(self, message: str, code: str | None) -> None:
        self.db.update_message(self.message_id, status="error", error=message, error_code=code)
        self.db.touch(self.conversation_id)


class SummarySink:
    """压缩任务的结果写到对话上，而不是某条消息。"""

    def __init__(self, db: DB, conversation_id: str, anchor_id: str):
        self.db = db
        self.conversation_id = conversation_id
        self.anchor_id = anchor_id

    def on_context(self, context: ContextResult) -> None:
        pass

    def on_progress(self, text: str) -> None:
        pass

    def on_finish(self, text: str, stats: dict[str, Any], status: str) -> None:
        summary = strip_thinking(text).strip()
        if status == "done" and summary:
            self.db.update_conversation(
                self.conversation_id, summary=summary, summary_upto=self.anchor_id
            )
        if stats.get("completion_tokens") is not None:
            self.db.add_usage(
                stats.get("model", ""),
                stats.get("processed_tokens", 0),
                stats.get("cached_tokens", 0),
                stats.get("completion_tokens", 0),
                stats.get("duration", 0.0),
            )

    def on_error(self, message: str, code: str | None) -> None:
        pass


class UsageSink:
    """OpenAI 兼容接口只记录用量，不落对话。"""

    def __init__(self, db: DB):
        self.db = db

    def on_context(self, context: ContextResult) -> None:
        pass

    def on_progress(self, text: str) -> None:
        pass

    def on_finish(self, text: str, stats: dict[str, Any], status: str) -> None:
        if stats.get("completion_tokens") is not None:
            self.db.add_usage(
                stats.get("model", ""),
                stats.get("processed_tokens", 0),
                stats.get("cached_tokens", 0),
                stats.get("completion_tokens", 0),
                stats.get("duration", 0.0),
            )

    def on_error(self, message: str, code: str | None) -> None:
        pass


def turns_from_path(
    path: list[dict[str, Any]], conversation: dict[str, Any] | None
) -> tuple[list[Turn], str | None]:
    """把消息链转成发送给模型的轮次，并应用压缩摘要。"""
    summary: str | None = None
    start = 0
    if conversation and conversation.get("summary_upto"):
        ids = [m["id"] for m in path]
        if conversation["summary_upto"] in ids:
            start = ids.index(conversation["summary_upto"]) + 1
            summary = conversation.get("summary")
    turns: list[Turn] = []
    for message in path[start:]:
        if message["role"] == "user":
            turns.append(Turn(message["id"], "user", message["content"], message["attachments"]))
        elif message["role"] == "assistant":
            if message["status"] in ("pending", "streaming", "error"):
                continue
            content = strip_thinking(message["content"]).strip()
            if content:
                turns.append(Turn(message["id"], "assistant", content))
    return turns, summary


def _recall_query(turns: list[Turn]) -> str:
    users = [t.content for t in turns if t.role == "user" and t.content]
    return " ".join(users[-2:])


class ChatService:
    def __init__(self, db: DB, engine: Engine, registry: ModelRegistry, memory: MemoryService):
        self.db = db
        self.engine = engine
        self.registry = registry
        self.memory = memory
        self._preview_lock = threading.Lock()

    # ---------------------------------------------------------- 工具
    def params_for(self, conversation: dict[str, Any] | None) -> dict[str, Any]:
        params = dict(self.db.get_settings()["params"])
        if conversation:
            params.update(normalize_params(conversation.get("settings")))
        return params

    def _conversation(self, conv_id: str) -> dict[str, Any]:
        conversation = self.db.get_conversation(conv_id)
        if conversation is None:
            raise NotFound("对话不存在")
        return conversation

    def _ensure_idle(self, conv_id: str) -> None:
        if self.engine.active_for(conv_id):
            raise Busy("这个对话正在生成，请等待完成或先停止")

    def _resolve_model(self, conversation: dict[str, Any]) -> str:
        model = conversation.get("model") or self.db.get_settings().get("default_model")
        if not model:
            raise BadRequest("请先选择一个模型")
        if self.registry.get(model) is None:
            raise BadRequest(f"模型 {model} 不在本地，请重新选择")
        return model

    def _owned_message(self, conv_id: str, message_id: str) -> dict[str, Any]:
        message = self.db.get_message(message_id)
        if message is None or message["conversation_id"] != conv_id:
            raise NotFound("消息不存在")
        return message

    def detail(self, conv_id: str) -> dict[str, Any]:
        conversation = self._conversation(conv_id)
        job = self.engine.active_for(conv_id)
        return {
            "conversation": conversation,
            "messages": self.db.path_to(conv_id, conversation["leaf_id"]),
            # prefix 让重新进入页面的客户端能正确回放“继续生成”的任务
            "job": {**job.summary(), "prefix": getattr(job.sink, "prefix", "")} if job else None,
            "params": self.params_for(conversation),
        }

    def _start(
        self,
        conversation: dict[str, Any],
        assistant: dict[str, Any],
        anchor_id: str,
        *,
        force: bool = False,
        title_from: str | None = None,
        continue_turn: Turn | None = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        conv_id = conversation["id"]
        params = self.params_for(conversation)
        model = assistant["model"]
        search_state: dict[str, Any] = {}

        def build(tokenizer: Any, counter: TokenCounter) -> ContextResult:
            current = self._conversation(conv_id)
            path = self.db.path_to(conv_id, anchor_id)
            turns, summary = turns_from_path(path, current)
            if continue_turn is not None:
                turns.append(continue_turn)
            limit = int(params["context_length"])
            composed = Composed()
            if params.get("memory", True):
                composed = self.memory.compose(_recall_query(turns), counter, limit)
                if composed.ids:
                    job.extra_stats["memory"] = composed.ids
            web = ""
            if search_state.get("sources"):
                web = websearch.render_block(
                    search_state["query"],
                    search_state["sources"],
                    counter,
                    min(limit // 3, SEARCH_BUDGET_TOKENS),
                )
            elif search_state.get("error"):
                web = websearch.failure_note(search_state["query"], search_state["error"])
            return build_context(
                tokenizer,
                counter,
                system=params["system_prompt"],
                summary=summary,
                turns=turns,
                limit=limit,
                max_tokens=int(params["max_tokens"]),
                thinking=bool(params["thinking"]),
                continue_final=continue_turn is not None,
                memory=composed.core,
                recall=composed.recall,
                web=web,
            )

        job = Job(
            kind="chat",
            model_id=model,
            params=params,
            build_context=build,
            sink=MessageSink(self.db, conv_id, assistant["id"], prefix),
            cache_key=conv_id,
            conversation_id=conv_id,
            message_id=assistant["id"],
            force=force,
        )
        if params.get("web_search") and continue_turn is None:
            job.prepare = self._search_step(conv_id, anchor_id, search_state)
        title_task = self._title_task(conv_id, title_from) if title_from is not None else None

        def post(engine: Engine, finished: Job) -> None:
            if finished.state == "done":
                self.memory.touch_used(finished.extra_stats.get("memory", []))
            if title_task is not None:
                title_task(engine, finished)

        job.post = post
        self.engine.enqueue(job)
        return {**self.detail(conv_id), "job_id": job.id}

    def _search_step(self, conv_id: str, anchor_id: str, state: dict[str, Any]):
        """在工作线程里先搜索，结果只进入本轮提示。"""

        def prepare(engine: Engine, job: Job) -> None:
            settings = self.db.get_settings()["search"]
            # 这里特意不带记忆：检索词要发到外部，不能混入用户的长期记忆
            turns, _ = turns_from_path(self.db.path_to(conv_id, anchor_id), None)
            job.state = "searching"
            engine.state = "searching"
            job.emit("status", state="searching")
            query = self._search_query(engine, turns)
            record: dict[str, Any] = {"query": query, "provider": settings["provider"]}
            state["query"] = query
            job.extra_stats["search"] = record
            job.emit("search", stage="query", query=query, provider=settings["provider"])
            if job.cancelled.is_set():
                return
            try:
                sources = websearch.search(query, settings)
            except websearch.SearchError as exc:
                state["error"] = str(exc)
                record["error"] = str(exc)
                job.emit("search", stage="failed", query=query, error=str(exc))
                return
            job.emit("status", state="reading", count=min(len(sources), settings["fetch_pages"]))
            websearch.enrich(sources, int(settings["fetch_pages"]))
            state["sources"] = sources
            record["sources"] = [s.public() for s in sources]
            job.emit("search", stage="done", query=query, sources=record["sources"])

        return prepare

    def _search_query(self, engine: Engine, turns: list[Turn]) -> str:
        names = {"user": "用户", "assistant": "助手"}
        recent = [f"{names[t.role]}：{t.content[:400]}" for t in turns[-5:]]
        prompt = SEARCH_QUERY_INSTRUCTION.format(
            today=datetime.now().strftime("%Y年%m月%d日"), dialog="\n".join(recent)
        )
        text = render_prompt(engine.tokenizer, [{"role": "user", "content": prompt}], thinking=False)
        raw = engine.generate_once(encode_prompt(engine.tokenizer, text), 32, temperature=0.1)
        fallback = next((t.content for t in reversed(turns) if t.role == "user"), "")
        return websearch.clean_query(strip_thinking(raw), fallback)

    def _title_task(self, conv_id: str, user_text: str):
        def post(engine: Engine, job: Job) -> None:
            if job.state != "done" or not self.db.get_settings().get("auto_title"):
                return
            conversation = self.db.get_conversation(conv_id)
            if conversation is None or conversation["title_source"] != "first_line":
                return
            reply = strip_thinking(job.result_text).strip()[:600]
            messages = [
                {"role": "system", "content": TITLE_INSTRUCTION},
                {"role": "user", "content": f"用户：{user_text[:600]}\n\n助手：{reply}"},
            ]
            text = render_prompt(engine.tokenizer, messages, thinking=False)
            title = clean_title(engine.generate_once(encode_prompt(engine.tokenizer, text), 32))
            if title:
                self.db.update_conversation(conv_id, title=title, title_source="auto")
                job.emit("title", title=title)

        return post

    # ---------------------------------------------------------- 动作
    def send(
        self,
        conv_id: str,
        content: str,
        attachments: list[dict[str, Any]],
        force: bool = False,
    ) -> dict[str, Any]:
        conversation = self._conversation(conv_id)
        self._ensure_idle(conv_id)
        if not content.strip() and not attachments:
            raise BadRequest("消息不能为空")
        model = self._resolve_model(conversation)

        user = self.db.add_message(
            conv_id, conversation["leaf_id"], "user", content, attachments=attachments
        )
        assistant = self.db.add_message(conv_id, user["id"], "assistant", status="pending", model=model)
        updates: dict[str, Any] = {"leaf_id": assistant["id"], "updated_at": time.time()}
        title_from = None
        if conversation["title_source"] == "default":
            updates["title"] = first_line_title(content or attachments[0].get("name", ""))
            updates["title_source"] = "first_line"
            title_from = content or attachments[0].get("name", "")
        conversation = self.db.update_conversation(conv_id, **updates)
        self.memory.request(conv_id, content)
        return self._start(conversation, assistant, user["id"], force=force, title_from=title_from)  # type: ignore[arg-type]

    def regenerate(self, conv_id: str, message_id: str, force: bool = False) -> dict[str, Any]:
        conversation = self._conversation(conv_id)
        self._ensure_idle(conv_id)
        message = self._owned_message(conv_id, message_id)
        model = self._resolve_model(conversation)
        anchor_id = message["id"] if message["role"] == "user" else message["parent_id"]
        if anchor_id is None:
            raise BadRequest("这条消息没有对应的用户输入")
        assistant = self.db.add_message(conv_id, anchor_id, "assistant", status="pending", model=model)
        conversation = self.db.update_conversation(
            conv_id, leaf_id=assistant["id"], updated_at=time.time()
        )
        return self._start(conversation, assistant, anchor_id, force=force)  # type: ignore[arg-type]

    def edit(self, conv_id: str, message_id: str, content: str, force: bool = False) -> dict[str, Any]:
        conversation = self._conversation(conv_id)
        self._ensure_idle(conv_id)
        message = self._owned_message(conv_id, message_id)
        if message["role"] != "user":
            raise BadRequest("只能编辑用户消息")
        if not content.strip() and not message["attachments"]:
            raise BadRequest("消息不能为空")
        model = self._resolve_model(conversation)
        user = self.db.add_message(
            conv_id, message["parent_id"], "user", content, attachments=message["attachments"]
        )
        assistant = self.db.add_message(conv_id, user["id"], "assistant", status="pending", model=model)
        conversation = self.db.update_conversation(
            conv_id, leaf_id=assistant["id"], updated_at=time.time()
        )
        return self._start(conversation, assistant, user["id"], force=force)  # type: ignore[arg-type]

    def continue_reply(self, conv_id: str, message_id: str, force: bool = False) -> dict[str, Any]:
        conversation = self._conversation(conv_id)
        self._ensure_idle(conv_id)
        message = self._owned_message(conv_id, message_id)
        if message["role"] != "assistant" or not message["content"]:
            raise BadRequest("只能继续生成助手的回复")
        model = self._resolve_model(conversation)
        raw = message["content"]
        # 思考已结束就只续写正文；还在思考中就把半截推理原样交给模型接着写
        visible = strip_thinking(raw) if "</think>" in raw else raw
        assistant = {**message, "model": model}
        self.db.update_message(message["id"], status="pending", error=None, error_code=None)
        conversation = self.db.update_conversation(
            conv_id, leaf_id=message["id"], updated_at=time.time()
        )
        return self._start(
            conversation,  # type: ignore[arg-type]
            assistant,
            message["parent_id"],
            force=force,
            continue_turn=Turn("__continue__", "assistant", visible),
            prefix=raw,
        )

    def switch_branch(self, conv_id: str, message_id: str) -> dict[str, Any]:
        self._conversation(conv_id)
        self._ensure_idle(conv_id)
        self._owned_message(conv_id, message_id)
        leaf = self.db.deepest_descendant(conv_id, message_id)
        self.db.update_conversation(conv_id, leaf_id=leaf)
        return self.detail(conv_id)

    def delete_message(self, conv_id: str, message_id: str) -> dict[str, Any]:
        self._conversation(conv_id)
        self._ensure_idle(conv_id)
        self._owned_message(conv_id, message_id)
        leaf = self.db.delete_subtree(conv_id, message_id)
        self.db.update_conversation(conv_id, leaf_id=leaf)
        self.memory.version += 1
        self.engine.forget_conversation(conv_id)  # KV 缓存里还有被删消息的内容
        return self.detail(conv_id)

    def delete_conversation(self, conv_id: str) -> int:
        """彻底删除对话和它留下的一切，返回同时移除的记忆条数。"""
        job = self.engine.active_for(conv_id)
        if job:
            self.engine.stop(job)
        removed = self.db.delete_conversation(conv_id)
        if removed:
            self.memory.version += 1
        self.engine.forget_conversation(conv_id)
        return removed

    def compact(self, conv_id: str, force: bool = False) -> dict[str, Any]:
        conversation = self._conversation(conv_id)
        self._ensure_idle(conv_id)
        model = self._resolve_model(conversation)
        path = self.db.path_to(conv_id, conversation["leaf_id"])
        usable = [m for m in path if m["status"] not in ("pending", "streaming", "error")]
        user_positions = [i for i, m in enumerate(usable) if m["role"] == "user"]
        if len(user_positions) < 2:
            raise BadRequest("对话至少要有两轮才能压缩")
        to_compact = usable[: user_positions[-1]]  # 最后一轮保留原文
        anchor_id = to_compact[-1]["id"]

        names = {"user": "用户", "assistant": "助手"}
        lines = []
        previous, _ = None, None
        turns, previous = turns_from_path(to_compact, conversation)
        if previous:
            lines.append(f"（更早的摘要）\n{previous}")
        for turn in turns:
            lines.append(f"{names[turn.role]}：{turn.text}")
        transcript = "\n\n".join(lines)

        params = self.params_for(conversation)
        params.update(temperature=0.3, thinking=False, max_tokens=1024, stop=[], system_prompt="")

        def build(tokenizer: Any, counter: TokenCounter) -> ContextResult:
            limit = int(params["context_length"])
            allowed = max(limit * 3 // 4 - 256, 128)
            encoded = tokenizer.encode(transcript, add_special_tokens=False)
            text = tokenizer.decode(encoded[-allowed:]) if len(encoded) > allowed else transcript
            return build_context(
                tokenizer,
                counter,
                system="",
                summary=None,
                turns=[Turn("__compact__", "user", COMPACT_INSTRUCTION + text)],
                limit=limit,
                max_tokens=int(params["max_tokens"]),
                thinking=False,
            )

        job = Job(
            kind="compact",
            model_id=model,
            params=params,
            build_context=build,
            sink=SummarySink(self.db, conv_id, anchor_id),
            cache_key=None,
            conversation_id=conv_id,
            force=force,
        )

        def verify(_: Engine, finished: Job) -> None:
            current = self._conversation(conv_id)
            if finished.state == "done" and current.get("summary_upto") != anchor_id:
                finished.emit("error", message="模型没有生成有效的摘要，上下文保持不变", code=None)

        job.post = verify
        self.engine.enqueue(job)
        return {"job_id": job.id}

    def clear_summary(self, conv_id: str) -> dict[str, Any]:
        self._conversation(conv_id)
        self.db.update_conversation(conv_id, summary=None, summary_upto=None)
        return self.detail(conv_id)

    # ---------------------------------------------------------- 预览
    def preview(
        self,
        *,
        conversation_id: str | None,
        draft: str,
        attachments: list[dict[str, Any]],
        model: str | None,
        settings: dict[str, Any] | None,
        include_prompt: bool = False,
    ) -> dict[str, Any]:
        conversation = self.db.get_conversation(conversation_id) if conversation_id else None
        params = self.params_for(conversation)
        params.update(normalize_params(settings))
        model_id = model or (conversation or {}).get("model") or self.db.get_settings().get("default_model")

        turns: list[Turn] = []
        summary = None
        if conversation:
            turns, summary = turns_from_path(
                self.db.path_to(conversation["id"], conversation["leaf_id"]), conversation
            )
        if draft or attachments:
            turns.append(Turn("__draft__", "user", draft, attachments))
        elif not turns or turns[-1].role == "assistant":
            # 输入框为空时按“下一条消息发送时”统计，这样最新一轮也能被截断，
            # 显示的数字和实际发送时一致
            turns.append(Turn(NEXT_TURN, "user", ""))

        limit = int(params["context_length"])
        arguments = {
            "system": params["system_prompt"],
            "summary": summary,
            "turns": turns,
            "limit": limit,
            "max_tokens": int(params["max_tokens"]),
        }
        with self._preview_lock:
            pair = self.engine.tokenizer_for(model_id) if model_id else None
            if params.get("memory", True):
                composed = self.memory.compose(
                    _recall_query(turns), pair[1] if pair else estimate_tokens, limit
                )
                arguments.update(memory=composed.core, recall=composed.recall)
            if pair is None:
                result = estimate_context(**arguments)
            else:
                tokenizer, counter = pair
                try:
                    result = build_context(
                        tokenizer, counter, thinking=bool(params["thinking"]), strict=False, **arguments
                    )
                except Exception:  # 模板异常时退回估算，界面不能因此报错
                    result = estimate_context(**arguments)
        payload = result.public()
        payload["included"] = [i for i in payload["included"] if i != NEXT_TURN]
        if include_prompt:
            payload["prompt"] = result.text
        return payload

    def count_tokens(self, model: str | None, text: str) -> int | None:
        if not model:
            return None
        with self._preview_lock:
            pair = self.engine.tokenizer_for(model)
            return pair[1](text) if pair else None

    # ---------------------------------------------------------- 导出
    def export_markdown(self, conv_id: str) -> str:
        conversation = self._conversation(conv_id)
        path = self.db.path_to(conv_id, conversation["leaf_id"])
        lines = [f"# {conversation['title']}", ""]
        if conversation.get("model"):
            lines += [f"> 模型：{conversation['model']}", ""]
        names = {"user": "用户", "assistant": "助手"}
        for message in path:
            lines += [f"## {names.get(message['role'], message['role'])}", ""]
            for attachment in message["attachments"]:
                lines += [f"📎 {attachment.get('name', '附件')}", ""]
            content = re.sub(
                r"<think>(.*?)</think>\s*",
                lambda m: f"<details><summary>思考过程</summary>\n\n{m.group(1).strip()}\n\n</details>\n\n",
                message["content"],
                flags=re.S,
            )
            lines += [content, ""]
        return "\n".join(lines)

    def export_json(self, conv_ids: list[str] | None = None) -> str:
        ids = conv_ids
        if ids is None:
            ids = [c["id"] for c in self.db.list_conversations(False)]
            ids += [c["id"] for c in self.db.list_conversations(True)]
        data = [d for d in (self.db.dump_conversation(i) for i in ids) if d]
        backup: dict[str, Any] = {"format": "mlxui", "version": 1, "conversations": data}
        if conv_ids is None:  # 单个对话的导出不带记忆，避免把长期记忆带出去
            backup["memories"] = self.db.dump_memories()
        return json.dumps(backup, ensure_ascii=False, indent=1)

    def import_json(self, payload: dict[str, Any]) -> int:
        items = payload.get("conversations")
        if payload.get("format") != "mlxui" or not isinstance(items, list):
            raise BadRequest("文件格式不正确，只支持本应用导出的 JSON")
        mapping: dict[str, str] = {}
        for item in items:
            restored = self.db.restore_conversation(item)
            if item.get("id"):
                mapping[item["id"]] = restored["id"]
        self.db.restore_memories(payload.get("memories") or [], mapping)
        self.memory.version += 1
        return len(items)
