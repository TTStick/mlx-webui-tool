"""MLX 推理引擎。

所有 MLX 调用都放在同一个工作线程里串行执行（Apple Silicon 上一次只跑一个推理），
HTTP 层只负责提交任务和订阅事件，所以刷新页面不会打断正在进行的生成。
"""
from __future__ import annotations

import asyncio
import gc
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import psutil
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load_tokenizer

from .context import ContextOverflow, ContextResult, StreamFilter, TokenCounter
from .db import DB
from .models import ModelRegistry

log = logging.getLogger("mlxui.engine")

LOW_MEMORY_BYTES = 800 * 1024 * 1024


class ModelTooLarge(Exception):
    code = "model_too_large"


class ModelMissing(Exception):
    code = "model_missing"


class Preempted(Exception):
    """后台任务让路给用户的请求。"""


def device_info() -> dict[str, Any]:
    return dict(mx.device_info())


class Job:
    """一次生成任务。事件进入内存列表，订阅者可以从任意序号开始读取。"""

    def __init__(
        self,
        *,
        kind: str,
        model_id: str,
        params: dict[str, Any],
        build_context: Callable[[Any, TokenCounter], ContextResult],
        sink: Any = None,
        cache_key: str | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        force: bool = False,
    ):
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.model_id = model_id
        self.params = params
        self.build_context = build_context
        self.sink = sink
        self.cache_key = cache_key
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.force = force
        self.state = "queued"
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.result_text = ""
        self.result_stats: dict[str, Any] = {}
        self.error: str | None = None
        self.post: Callable[["Engine", "Job"], None] | None = None
        self.prepare: Callable[["Engine", "Job"], None] | None = None
        self.extra_stats: dict[str, Any] = {}
        self.cancelled = threading.Event()
        self.done_event = threading.Event()
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    # ------------------------------------------------------- 事件总线
    def emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            data["type"] = kind
            data["seq"] = len(self._events)
            self._events.append(data)
            subscribers = list(self._subscribers)
        for loop, event in subscribers:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass  # 订阅方的事件循环已关闭

    def slice(self, start: int) -> list[dict[str, Any]]:
        with self._lock:
            return self._events[start:]

    def subscribe(self) -> asyncio.Event:
        event = asyncio.Event()
        with self._lock:
            self._subscribers.append((asyncio.get_running_loop(), event))
        return event

    def unsubscribe(self, event: asyncio.Event) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s[1] is not event]

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    def close(self) -> None:
        self.finished_at = time.time()
        self.emit("end")
        self.done_event.set()

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "model": self.model_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
        }


@dataclass
class CacheEntry:
    model_id: str
    cache: list[Any]
    tokens: list[int]
    nbytes: int
    used_at: float


def _cache_nbytes(cache: list[Any]) -> int:
    try:
        return int(sum(c.nbytes for c in cache))
    except (NotImplementedError, AttributeError, TypeError):
        return 0


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "Insufficient Memory" in text or "out of memory" in text.lower():
        return "显存不足，生成已中止。可以调小上下文窗口、卸载其他程序，或换一个更小的模型。"
    return text or exc.__class__.__name__


class Engine:
    def __init__(self, db: DB, registry: ModelRegistry):
        self.db = db
        self.registry = registry
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_id: str | None = None
        self.weights_bytes = 0
        self.load_seconds = 0.0
        self.loaded_at: float | None = None
        self.state = "idle"
        self.current: Job | None = None
        self.last_active = time.time()
        self.caches: OrderedDict[str, CacheEntry] = OrderedDict()
        self.kv_bytes_per_token: dict[str, int] = {}
        self.jobs: dict[str, Job] = {}
        self._pending: list[Job] = []
        self._pending_lock = threading.Lock()
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._tokenizers: OrderedDict[str, tuple[Any, TokenCounter]] = OrderedDict()
        self._tokenizer_lock = threading.Lock()
        self._counter: TokenCounter | None = None
        self.idle_hook: Callable[["Engine"], None] | None = None
        self._thread = threading.Thread(target=self._loop, name="mlx-worker", daemon=True)
        self._thread.start()

    # -------------------------------------------------------- 工作线程
    def _loop(self) -> None:
        while True:
            try:
                task = self._queue.get(timeout=3)
            except queue.Empty:
                self._on_idle()
                continue
            if task is None:
                return
            try:
                task()
            except Exception as exc:  # 单个任务失败不能拖垮整个工作线程
                log.error("worker task failed: %s", type(exc).__name__)  # 只记类型，避免内容进日志
            self._prune_jobs()

    def _on_idle(self) -> None:
        if not self.db.is_open:
            return
        self._maybe_idle_unload()
        if self.db.needs_checkpoint:
            self.db.checkpoint()
        if self.idle_hook is not None and self.current is None:
            try:
                self.idle_hook(self)
            except Exception as exc:  # noqa: BLE001
                log.error("idle task failed: %s", type(exc).__name__)

    def submit(self, fn: Callable[[], Any]) -> Future:
        future: Future = Future()

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn())
            except Exception as exc:  # noqa: BLE001 - 原样转交调用方
                future.set_exception(exc)

        self._queue.put(run)
        return future

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _prune_jobs(self) -> None:
        cutoff = time.time() - 120
        for job_id in [j.id for j in self.jobs.values() if j.finished_at and j.finished_at < cutoff]:
            self.jobs.pop(job_id, None)

    def _maybe_idle_unload(self) -> None:
        minutes = self.db.get_settings().get("idle_unload_minutes") or 0
        if minutes and self.model is not None and self.current is None:
            if time.time() - self.last_active > minutes * 60:
                self._unload()

    # ---------------------------------------------------------- 模型
    def _trust_remote_code(self) -> bool:
        return bool(self.db.get_settings().get("trust_remote_code"))

    def _unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.weights_bytes = 0
        self.load_seconds = 0.0
        self.loaded_at = None
        self._counter = None
        self.caches.clear()
        gc.collect()
        mx.clear_cache()

    def _ensure_model(self, model_id: str, force: bool = False) -> None:
        if self.model_id == model_id and self.model is not None:
            return
        info = self.registry.get(model_id)
        if info is None:
            raise ModelMissing(f"找不到模型 {model_id}，它可能已被删除。")

        limit = int(device_info().get("max_recommended_working_set_size", 0))
        if limit and info["weights_bytes"] > limit * 0.95 and not force:
            raise ModelTooLarge(
                f"模型权重 {info['weights_bytes'] / 1024**3:.1f} GB，"
                f"超过本机建议的显存上限 {limit / 1024**3:.1f} GB，加载后可能非常慢甚至卡死。"
            )

        self._unload()
        self.state = "loading"
        started = time.perf_counter()
        trust = self._trust_remote_code()
        model, tokenizer = mlx_load(
            info["path"],
            tokenizer_config={"trust_remote_code": trust},
            trust_remote_code=trust,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._counter = TokenCounter(tokenizer)
        self.weights_bytes = int(sum(v.nbytes for _, v in tree_flatten(model.parameters())))
        self.load_seconds = time.perf_counter() - started
        self.loaded_at = time.time()
        self.last_active = time.time()

    def load_model(self, model_id: str, force: bool = False) -> Future:
        def task() -> dict[str, Any]:
            try:
                self._ensure_model(model_id, force=force)
            finally:
                self.state = "idle"
            return self.status()

        return self.submit(task)

    def unload_model(self) -> Future:
        def task() -> dict[str, Any]:
            self._unload()
            return self.status()

        return self.submit(task)

    def clear_kv_cache(self) -> Future:
        def task() -> dict[str, Any]:
            self.caches.clear()
            gc.collect()
            mx.clear_cache()
            return self.status()

        return self.submit(task)

    def tokenizer_for(self, model_id: str) -> tuple[Any, TokenCounter] | None:
        """取分词器用于统计 token。模型没加载时也能工作，且不占显存。

        特意不共用工作线程那份实例：Rust 分词器并发编码可能报 Already borrowed。
        """
        with self._tokenizer_lock:
            if model_id in self._tokenizers:
                self._tokenizers.move_to_end(model_id)
                return self._tokenizers[model_id]
            info = self.registry.get(model_id)
            if info is None:
                return None
            try:
                tokenizer = load_tokenizer(
                    info["path"],
                    tokenizer_config_extra={"trust_remote_code": self._trust_remote_code()},
                )
            except Exception as exc:  # noqa: BLE001 - 分词器不可用时退回估算
                log.error("tokenizer load failed: %s", type(exc).__name__)
                return None
            pair = (tokenizer, TokenCounter(tokenizer))
            self._tokenizers[model_id] = pair
            while len(self._tokenizers) > 4:
                self._tokenizers.popitem(last=False)
            return pair

    # -------------------------------------------------------- KV 缓存
    def _acquire_cache(self, key: str | None, tokens: list[int]) -> tuple[list[Any], int]:
        """尽量复用上一轮的 KV 缓存，返回 (cache, 已缓存 token 数)。"""
        fresh = make_prompt_cache(self.model)
        if key is None:
            return fresh, 0
        entry = self.caches.pop(key, None)
        if entry is None or entry.model_id != self.model_id:
            return fresh, 0

        shared = 0
        for a, b in zip(entry.tokens, tokens):
            if a != b:
                break
            shared += 1
        shared = min(shared, len(tokens) - 1)  # 至少留一个 token 触发模型计算
        if shared <= 0:
            return fresh, 0

        surplus = len(entry.tokens) - shared
        if surplus > 0:
            if not can_trim_prompt_cache(entry.cache):
                return fresh, 0
            if trim_prompt_cache(entry.cache, surplus) != surplus:
                return fresh, 0
        return entry.cache, shared

    def _store_cache(self, key: str | None, cache: list[Any], tokens: list[int]) -> None:
        limit = int(self.db.get_settings().get("kv_cache_limit_mb", 2048)) * 1024 * 1024
        if key is None or limit <= 0:
            return
        offset = getattr(cache[0], "offset", None) if cache else None
        if not isinstance(offset, int) or offset <= 0:
            return
        mx.eval([c.state for c in cache])
        nbytes = _cache_nbytes(cache)
        if offset > 0 and nbytes:
            keys = getattr(cache[0], "keys", None)
            allocated = keys.shape[2] if keys is not None and keys.ndim >= 3 else offset
            self.kv_bytes_per_token[self.model_id] = int(nbytes / max(allocated, 1))  # type: ignore[index]
        self.caches[key] = CacheEntry(
            model_id=self.model_id,  # type: ignore[arg-type]
            cache=cache,
            tokens=tokens[:offset],
            nbytes=nbytes,
            used_at=time.time(),
        )
        self.caches.move_to_end(key)
        self._evict(limit)

    def _evict(self, limit: int) -> None:
        total = sum(e.nbytes for e in self.caches.values())
        low_memory = psutil.virtual_memory().available < LOW_MEMORY_BYTES
        evicted = False
        while len(self.caches) > 1 and (total > limit or low_memory):
            _, oldest = self.caches.popitem(last=False)
            total -= oldest.nbytes
            evicted = True
        if evicted:
            gc.collect()
        mx.clear_cache()

    # ---------------------------------------------------------- 生成
    def enqueue(self, job: Job) -> Job:
        self.jobs[job.id] = job
        with self._pending_lock:
            self._pending.append(job)
            position = len(self._pending)
        busy = self.current is not None or position > 1
        job.emit("status", state="queued", position=position if busy else 0)
        self._queue.put(lambda: self._run_job(job))
        return job

    def stop(self, job: Job) -> None:
        job.cancelled.set()

    def active_for(self, conversation_id: str) -> Job | None:
        for job in self.jobs.values():
            if job.conversation_id == conversation_id and not job.finished:
                return job
        return None

    def _finish_failed(self, job: Job, exc: Exception) -> None:
        message = _friendly_error(exc)
        code = getattr(exc, "code", None)
        job.state = "error"
        job.error = message
        if job.sink is not None:
            job.sink.on_error(message, code)
        job.emit("error", message=message, code=code)
        job.close()

    def _run_job(self, job: Job) -> None:
        with self._pending_lock:
            if job in self._pending:
                self._pending.remove(job)
        if job.cancelled.is_set():
            job.state = "stopped"
            if job.sink is not None:
                job.sink.on_finish("", {"finish_reason": "stopped"}, "stopped")
            job.emit("done", status="stopped", stats={"finish_reason": "stopped"})
            job.close()
            return

        self.current = job
        self.last_active = time.time()
        try:
            if self.model_id != job.model_id or self.model is None:
                job.state = "loading"
                job.emit("status", state="loading", model=job.model_id)
            self._ensure_model(job.model_id, force=job.force)

            if job.prepare is not None:
                job.prepare(self, job)

            job.state = "preparing"
            self.state = "preparing"
            job.emit("status", state="preparing")
            context = job.build_context(self.tokenizer, self._counter)  # type: ignore[arg-type]
            job.emit("context", **context.public())
            if job.sink is not None:
                job.sink.on_context(context)

            self._generate(job, context)
        except ContextOverflow as exc:
            self._finish_failed(job, exc)
        except Exception as exc:  # noqa: BLE001 - 任何推理错误都要回给界面
            log.error("generation failed: %s", type(exc).__name__)
            if job.cache_key:
                self.caches.pop(job.cache_key, None)
            self._finish_failed(job, exc)
        else:
            if job.post is not None:
                try:
                    job.post(self, job)
                except Exception as exc:  # noqa: BLE001
                    log.error("post task failed: %s", type(exc).__name__)
            job.close()
        finally:
            self.current = None
            self.state = "idle"
            self.last_active = time.time()

    def _generate(self, job: Job, context: ContextResult) -> None:
        params = job.params
        tokens = context.tokens
        cache, cached = self._acquire_cache(job.cache_key, tokens)
        remaining = tokens[cached:]

        job.emit("status", state="prompt", cached=cached, total=len(tokens))
        job.state = "generating"
        self.state = "generating"
        mx.reset_peak_memory()

        if params.get("seed") is not None:
            mx.random.seed(int(params["seed"]))
        sampler = make_sampler(
            temp=float(params["temperature"]),
            top_p=float(params["top_p"]),
            min_p=float(params["min_p"]),
            top_k=int(params["top_k"]),
        )
        processors = None
        if float(params["repetition_penalty"]) != 1.0:
            processors = make_logits_processors(
                repetition_penalty=float(params["repetition_penalty"]),
                repetition_context_size=int(params["repetition_context_size"]),
            )

        tokenizer = self.tokenizer
        think_start = getattr(tokenizer, "think_start", None)
        think_end = getattr(tokenizer, "think_end", None)
        output_filter = StreamFilter(list(params.get("stop") or []), think_start, think_end)

        def on_prompt_progress(done: int, total: int) -> None:
            job.emit("progress", done=done + cached, total=total + cached)

        full = ""
        pending = ""
        if context.opens_thinking:  # 模板把 <think> 写进了提示末尾，输出要补回开标签
            full = pending = "<think>\n"

        generated: list[int] = []
        finish_reason = "length"
        completion = 0
        prompt_tps = 0.0
        generation_tps = 0.0
        ttft = 0.0
        last_delta = last_tick = last_save = started = time.perf_counter()

        stream = stream_generate(
            self.model,
            tokenizer,
            mx.array(remaining),
            max_tokens=max(context.max_reply, 1),
            sampler=sampler,
            logits_processors=processors,
            prompt_cache=cache,
            prompt_progress_callback=on_prompt_progress,
        )
        try:
            for response in stream:
                now = time.perf_counter()
                if not generated:
                    ttft = now - started
                    job.emit("status", state="generating", ttft=ttft)
                generated.append(response.token)
                completion = response.generation_tokens
                prompt_tps = response.prompt_tps
                generation_tps = response.generation_tps

                out, stopped = output_filter.feed(response.text)
                full += out
                pending += out

                if job.cancelled.is_set():
                    finish_reason = "stopped"
                    break
                if stopped:
                    finish_reason = "stop_sequence"
                    break
                if response.finish_reason:
                    finish_reason = response.finish_reason

                if pending and (now - last_delta > 0.03 or response.finish_reason):
                    job.emit("delta", text=pending)
                    pending = ""
                    last_delta = now
                if now - last_tick > 0.25:
                    job.emit(
                        "tick",
                        tokens=completion,
                        tps=generation_tps,
                        active_memory=int(mx.get_active_memory()),
                        peak_memory=int(mx.get_peak_memory()),
                    )
                    last_tick = now
                if job.sink is not None and now - last_save > 1.0:
                    job.sink.on_progress(full)
                    last_save = now
        finally:
            stream.close()  # 立刻退出 wired_limit 上下文

        tail = output_filter.flush()
        full += tail
        pending += tail
        if pending:
            job.emit("delta", text=pending)

        self._store_cache(job.cache_key, cache, tokens + generated)
        entry = self.caches.get(job.cache_key) if job.cache_key else None
        duration = time.perf_counter() - started
        stats = {
            "model": job.model_id,
            "finish_reason": finish_reason,
            "prompt_tokens": len(tokens),
            "cached_tokens": cached,
            "processed_tokens": len(remaining),
            "completion_tokens": completion,
            "ttft": round(ttft, 4),
            "prompt_tps": round(prompt_tps, 2),
            "generation_tps": round(generation_tps, 2) if completion >= 3 else None,
            "duration": round(duration, 3),
            "peak_memory": int(mx.get_peak_memory()),
            "kv_bytes": entry.nbytes if entry else _cache_nbytes(cache),
            "context_limit": context.limit,
            "excluded_messages": len(context.excluded),
            **job.extra_stats,
        }
        status = "stopped" if finish_reason == "stopped" else "done"
        job.state = status
        job.result_text = full
        job.result_stats = stats
        if job.sink is not None:
            job.sink.on_finish(full, stats, status)
        job.emit("done", status=status, stats=stats)

    def generate_once(
        self, tokens: list[int], max_tokens: int, temperature: float = 0.3, preemptible: bool = False
    ) -> str:
        """工作线程内的一次性生成，不动对话的 KV 缓存。

        preemptible=True 时，一旦有用户请求进入队列就抛 Preempted，后台任务立刻让路。
        """

        def check(*_: Any) -> None:
            if preemptible and not self._queue.empty():
                raise Preempted()

        parts: list[str] = []
        stream = stream_generate(
            self.model,
            self.tokenizer,
            mx.array(tokens),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            prompt_progress_callback=check,
        )
        try:
            for response in stream:
                check()
                parts.append(response.text)
        finally:
            stream.close()
            mx.clear_cache()
        return "".join(parts)

    # -------------------------------------------------------- 清除痕迹
    def _clear_text_caches(self) -> None:
        if self._counter is not None:
            self._counter.clear()
        with self._tokenizer_lock:
            for _, counter in self._tokenizers.values():
                counter.clear()

    def forget_conversation(self, conversation_id: str) -> Future:
        """排在当前任务之后执行：正在生成的任务结束时可能刚把缓存写回去。"""

        def task() -> None:
            self.caches.pop(conversation_id, None)
            for job in [j for j in self.jobs.values() if j.conversation_id == conversation_id and j.finished]:
                self.jobs.pop(job.id, None)
            self._clear_text_caches()
            gc.collect()
            mx.clear_cache()

        return self.submit(task)

    def purge_sensitive(self) -> None:
        """锁定或抹除时调用，必须在工作线程内执行。模型权重不含用户内容，可以保留。"""
        self.caches.clear()
        for job in [j for j in self.jobs.values() if j.finished]:
            self.jobs.pop(job.id, None)
        self._clear_text_caches()
        gc.collect()
        mx.clear_cache()

    def cancel_all(self) -> None:
        for job in list(self.jobs.values()):
            job.cancelled.set()

    # ---------------------------------------------------------- 状态
    def status(self) -> dict[str, Any]:
        entries = [
            {"key": key, "tokens": len(e.tokens), "bytes": e.nbytes, "model": e.model_id}
            for key, e in list(self.caches.items())
        ]
        with self._pending_lock:
            pending = [j.summary() for j in self._pending]
        return {
            "state": self.state,
            "model_id": self.model_id,
            "weights_bytes": self.weights_bytes,
            "load_seconds": round(self.load_seconds, 2),
            "loaded_at": self.loaded_at,
            "kv_entries": entries,
            "kv_bytes": sum(e["bytes"] for e in entries),
            "kv_bytes_per_token": dict(self.kv_bytes_per_token),
            "current_job": self.current.summary() if self.current else None,
            "queue": pending,
        }
