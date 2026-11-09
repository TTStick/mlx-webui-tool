"""把消息链组装成 prompt，并精确统计每部分占用的 token。

上下文窗口语义：context_length = prompt + 回复 的 token 总量上限。
超出时按完整轮次丢弃最早的历史（system 与最新一条用户消息始终保留）。
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)
SUMMARY_HEADER = "【此前对话的摘要】"
_MIN_REPLY_ROOM = 16


class ContextOverflow(Exception):
    """最新一条消息本身就放不进上下文窗口。"""

    def __init__(self, needed: int, limit: int):
        super().__init__(
            f"这条消息需要 {needed:,} tokens，超过了当前上下文窗口 {limit:,}。"
            "请调大上下文窗口或缩短内容。"
        )
        self.needed = needed
        self.limit = limit


def strip_thinking(text: str) -> str:
    """去掉推理内容，历史消息只把正文发回给模型。"""
    text = THINK_BLOCK.sub("", text)
    unclosed = text.find("<think>")
    return text[:unclosed] if unclosed >= 0 else text


@dataclass
class Turn:
    id: str
    role: str
    content: str
    attachments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attachment_text(self) -> str:
        parts = []
        for att in self.attachments:
            parts.append(f'<file name="{att.get("name", "file")}">\n{att.get("text", "")}\n</file>')
        return "\n\n".join(parts)

    @property
    def text(self) -> str:
        files = self.attachment_text
        return f"{files}\n\n{self.content}" if files else self.content


@dataclass
class ContextResult:
    tokens: list[int]
    text: str
    total: int
    limit: int
    reserve: int
    max_reply: int
    included: list[str]
    excluded: list[str]
    breakdown: dict[str, int]
    counts: dict[str, int]
    over_limit: bool = False
    estimated: bool = False
    opens_thinking: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "limit": self.limit,
            "reserve": self.reserve,
            "max_reply": self.max_reply,
            "included": self.included,
            "excluded": self.excluded,
            "breakdown": self.breakdown,
            "counts": self.counts,
            "over_limit": self.over_limit,
            "estimated": self.estimated,
        }


class TokenCounter:
    """带 LRU 的计数器；输入框实时统计时不必反复编码整段历史。"""

    def __init__(self, tokenizer: Any, capacity: int = 4096):
        self.tokenizer = tokenizer
        self.capacity = capacity
        self._memo: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            cached = self._memo.get(text)
            if cached is not None:
                self._memo.move_to_end(text)
                return cached
        count = len(self.tokenizer.encode(text, add_special_tokens=False))
        with self._lock:
            self._memo[text] = count
            if len(self._memo) > self.capacity:
                self._memo.popitem(last=False)
        return count

    def clear(self) -> None:
        """缓存的键是原文，删除对话时要一起清掉。"""
        with self._lock:
            self._memo.clear()


def estimate_tokens(text: str) -> int:
    """没有分词器时的粗略估算：CJK 约 1 字 1 token，其余约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "⺀" <= ch <= "鿿" or "豈" <= ch <= "﫿")
    return cjk + max((len(text) - cjk) // 4, 1 if len(text) > cjk else 0)


def _merge_adjacent(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """部分模板要求 user/assistant 严格交替，合并相邻同角色消息。"""
    merged: list[dict[str, str]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1] = {"role": msg["role"], "content": f'{merged[-1]["content"]}\n\n{msg["content"]}'}
        else:
            merged.append(dict(msg))
    return merged


def _fold_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """模板不支持 system 角色时，把 system 并入第一条用户消息。"""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = f'{system}\n\n{rest[0]["content"]}'
    elif system:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def _plain_prompt(messages: list[dict[str, str]], generation: bool) -> str:
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    text = "\n\n".join(f'{names[m["role"]]}: {m["content"]}' for m in messages)
    return f"{text}\n\nAssistant:" if generation else text


def render_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    thinking: bool = True,
    continue_final: bool = False,
) -> str:
    """用模型自带的 chat template 渲染；不支持 system 角色或没有模板时自动兜底。"""
    messages = _merge_adjacent(messages)
    generation = not continue_final
    if not getattr(tokenizer, "has_chat_template", False):
        return _plain_prompt(messages, generation)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": generation,
        "enable_thinking": bool(thinking),
    }
    if continue_final:
        kwargs["continue_final_message"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        if not any(m["role"] == "system" for m in messages):
            raise
        return tokenizer.apply_chat_template(_fold_system(messages), **kwargs)


def encode_prompt(tokenizer: Any, text: str) -> list[int]:
    """与 mlx_lm.stream_generate 一致：模板已带 BOS 时不再重复添加。"""
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not text.startswith(bos)
    return list(tokenizer.encode(text, add_special_tokens=add_special))


def build_context(
    tokenizer: Any,
    counter: TokenCounter,
    *,
    system: str,
    summary: str | None,
    turns: list[Turn],
    limit: int,
    max_tokens: int,
    thinking: bool = True,
    continue_final: bool = False,
    strict: bool = True,
    memory: str = "",
    recall: str = "",
    web: str = "",
) -> ContextResult:
    """组装 prompt。strict=False 用于预览，放不下时返回 over_limit 而不是抛异常。

    memory 放进 system，保持稳定以便复用 KV 缓存；recall 和 web 每轮都会变，
    贴在最后一条用户消息前面，只影响最后一轮的缓存，也不会写进历史。
    """
    reserve = min(max_tokens, max(limit // 4, 1))
    # 预算不能比下面的溢出判定宽松，否则截断后仍可能被判为放不下
    budget = limit - max(reserve, _MIN_REPLY_ROOM)

    head: list[dict[str, str]] = []
    summary_block = f"{SUMMARY_HEADER}\n{summary}" if summary else ""
    system_text = "\n\n".join(part for part in (system.strip(), memory, summary_block) if part)
    if system_text:
        head.append({"role": "system", "content": system_text})

    starts = [i for i, t in enumerate(turns) if t.role == "user"] or [0]
    last_user_index = max((i for i, t in enumerate(turns) if t.role == "user"), default=-1)
    transient = "\n\n".join(part for part in (web, recall) if part)

    def attempt(start: int) -> tuple[str, list[int]]:
        body = []
        for index, turn in enumerate(turns[start:], start):
            content = turn.text
            if index == last_user_index and transient:
                content = f"{transient}\n\n{content}" if content else transient
            body.append({"role": turn.role, "content": content})
        text = render_prompt(tokenizer, head + body, thinking=thinking, continue_final=continue_final)
        return text, encode_prompt(tokenizer, text)

    try:
        text, tokens = attempt(starts[0])
        chosen = starts[0]
        if len(tokens) > budget and len(starts) > 1:
            # 丢弃轮次数与 token 数单调相关，用二分找到能放下的最长后缀
            low, high = 1, len(starts) - 1
            best: tuple[int, str, list[int]] | None = None
            while low <= high:
                mid = (low + high) // 2
                t_text, t_tokens = attempt(starts[mid])
                if len(t_tokens) <= budget:
                    best = (starts[mid], t_text, t_tokens)
                    high = mid - 1
                else:
                    low = mid + 1
            if best is None:
                chosen = starts[-1]
                text, tokens = attempt(chosen)
            else:
                chosen, text, tokens = best
    except Exception:
        if strict:
            raise
        # 预览阶段模板可能拒绝空会话（例如 Qwen3.5 要求至少一条用户消息）
        chosen = starts[0]
        text = ""
        tokens = []

    included_turns = turns[chosen:]
    excluded_turns = turns[:chosen]

    counts = {t.id: counter(t.text) for t in turns}
    last_user = next((t for t in reversed(included_turns) if t.role == "user"), None)

    breakdown = {
        "system": counter(system.strip()),
        "summary": counter(summary) if summary else 0,
        "attachments": 0,
        "history": 0,
        "current": 0,
        "template": 0,
        "memory": counter(memory) + counter(recall),
        "search": counter(web),
    }
    for turn in included_turns:
        files = counter(turn.attachment_text)
        breakdown["attachments"] += files
        body = max(counts[turn.id] - files, 0)
        if turn is last_user:
            breakdown["current"] += body
        else:
            breakdown["history"] += body

    content_sum = sum(v for k, v in breakdown.items() if k != "template")
    total = len(tokens) if tokens else content_sum
    breakdown["template"] = max(total - content_sum, 0)

    over_limit = total > limit - _MIN_REPLY_ROOM
    if over_limit and strict:
        raise ContextOverflow(total + _MIN_REPLY_ROOM, limit)

    think_start = getattr(tokenizer, "think_start", None)
    # 以渲染结果为准：有些模板不管开关都会把开标签写进提示
    opens_thinking = bool(think_start and tokens and text.rstrip().endswith(think_start))
    return ContextResult(
        tokens=tokens,
        text=text,
        total=total,
        limit=limit,
        reserve=reserve,
        max_reply=max(min(max_tokens, limit - total), 0),
        included=[t.id for t in included_turns],
        excluded=[t.id for t in excluded_turns],
        breakdown=breakdown,
        counts=counts,
        over_limit=over_limit,
        opens_thinking=opens_thinking,
    )


def estimate_context(
    *,
    system: str,
    summary: str | None,
    turns: list[Turn],
    limit: int,
    max_tokens: int,
    memory: str = "",
    recall: str = "",
    web: str = "",
) -> ContextResult:
    """未选择模型或分词器不可用时的估算结果，仅用于界面展示。"""
    reserve = min(max_tokens, max(limit // 4, 1))
    counts = {t.id: estimate_tokens(t.text) for t in turns}
    last_user = next((t for t in reversed(turns) if t.role == "user"), None)
    breakdown = {
        "system": estimate_tokens(system),
        "summary": estimate_tokens(summary or ""),
        "attachments": sum(estimate_tokens(t.attachment_text) for t in turns),
        "history": 0,
        "current": 0,
        "template": 8 * len(turns),
        "memory": estimate_tokens(memory) + estimate_tokens(recall),
        "search": estimate_tokens(web),
    }
    for turn in turns:
        body = max(counts[turn.id] - estimate_tokens(turn.attachment_text), 0)
        breakdown["current" if turn is last_user else "history"] += body
    total = sum(breakdown.values())
    return ContextResult(
        tokens=[],
        text="",
        total=total,
        limit=limit,
        reserve=reserve,
        max_reply=max(min(max_tokens, limit - total), 0),
        included=[t.id for t in turns],
        excluded=[],
        breakdown=breakdown,
        counts=counts,
        over_limit=total > limit - _MIN_REPLY_ROOM,
        estimated=True,
    )


class StreamFilter:
    """流式文本过滤：处理自定义停止词，并把各家推理标签统一成 <think>。

    结尾若可能是某个标记的前缀就先扣住，等后续 token 到达再决定，
    这样跨 token 拆开的标记也不会漏到界面上。
    """

    def __init__(self, stops: list[str], think_start: str | None, think_end: str | None):
        self.stops = [s for s in stops if s]
        self.replace: dict[str, str] = {}
        if think_start and think_start != "<think>":
            self.replace[think_start] = "<think>"
        if think_end and think_end != "</think>":
            self.replace[think_end] = "</think>"
        self.needles = self.stops + list(self.replace)
        self.buffer = ""

    def _apply(self, text: str) -> str:
        for source, target in self.replace.items():
            text = text.replace(source, target)
        return text

    def _held_suffix(self) -> int:
        longest = 0
        for needle in self.needles:
            for size in range(min(len(needle) - 1, len(self.buffer)), longest, -1):
                if self.buffer.endswith(needle[:size]):
                    longest = size
                    break
        return longest

    def feed(self, text: str) -> tuple[str, bool]:
        """返回 (可以输出的文本, 是否命中停止词)。"""
        self.buffer += text
        earliest = -1
        for stop in self.stops:
            pos = self.buffer.find(stop)
            if pos >= 0 and (earliest < 0 or pos < earliest):
                earliest = pos
        if earliest >= 0:
            out = self._apply(self.buffer[:earliest])
            self.buffer = ""
            return out, True
        if not self.needles:
            out, self.buffer = self.buffer, ""
            return out, False
        hold = self._held_suffix()
        cut = len(self.buffer) - hold
        out = self._apply(self.buffer[:cut])
        self.buffer = self.buffer[cut:]
        return out, False

    def flush(self) -> str:
        out = self._apply(self.buffer)
        self.buffer = ""
        return out
