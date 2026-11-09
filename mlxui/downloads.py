"""模型下载管理与 Hugging Face 搜索。"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

from .db import DB
from .models import REPO_ID, ModelRegistry, _architecture_supported
from .monitor import disk_info

_DISK_MARGIN = 1024**3  # 下载后至少给系统留 1 GB
_META_TTL = 600.0

# 尺寸取自 Hugging Face（2026-11 核对），界面展示时再结合本机显存判断是否合适
RECOMMENDED = [
    ("mlx-community/Qwen3-0.6B-4bit", 0.35, "qwen3", "最轻量的中文模型，适合先试用，支持思考模式"),
    ("mlx-community/Qwen3-1.7B-4bit", 0.98, "qwen3", "轻量但够用，日常问答响应很快"),
    ("mlx-community/gemma-3-1b-it-qat-4bit", 0.77, "gemma3_text", "Google 出品的小模型，英文表现好"),
    ("mlx-community/SmolLM3-3B-4bit", 1.75, "smollm3", "3B 级别，支持思考模式"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", 1.82, "llama", "经典的通用小模型"),
    ("mlx-community/Qwen3-4B-Instruct-2507-4bit", 2.28, "qwen3", "16GB 内存的主力选择，中文好、速度快"),
    ("mlx-community/Qwen3.5-4B-OptiQ-4bit", 4.04, "qwen3_5", "更新一代的 4B 模型，质量更高"),
    ("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", 4.30, "qwen2", "偏编程场景"),
    ("mlx-community/Qwen3-8B-4bit", 4.62, "qwen3", "质量明显提升，长上下文需注意显存"),
    ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", 4.62, "qwen3", "擅长推理，带完整思考过程"),
    ("mlx-community/Qwen3.5-9B-OptiQ-4bit", 8.22, "qwen3_5", "接近 16GB 机器的上限，上下文要开小一些"),
]


def _fit(weights_bytes: int, working_set: int) -> str:
    """ok：放得下且有余量；tight：能跑但上下文要小；too_large：超过显存建议上限。"""
    if not working_set:
        return "ok"
    if weights_bytes > working_set * 0.95:
        return "too_large"
    # 显存建议上限并不等于真正可用的量，系统和其他应用还要占一大块
    if weights_bytes + 2.5 * 1024**3 > working_set * 0.7:
        return "tight"
    return "ok"


def _http_json(url: str, timeout: float = 15) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "mlxui/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _matches_allow(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in DEFAULT_ALLOW_PATTERNS)


def repo_cache_path(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


class Download:
    def __init__(self, repo: str, total: int, cache: Path, weights: int):
        self.id = uuid.uuid4().hex
        self.repo = repo
        self.total = total
        self.weights = weights
        self.cache = cache
        self.status = "running"
        self.error: str | None = None
        self.done = 0
        self.speed = 0.0
        self.started_at = time.time()
        self.proc: subprocess.Popen[str] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo": self.repo,
            "total": self.total,
            "done": min(self.done, self.total) if self.total else self.done,
            "speed": self.speed,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
        }


class DownloadManager:
    def __init__(self, db: DB, registry: ModelRegistry, working_set: int):
        self.db = db
        self.registry = registry
        self.working_set = working_set
        self.tasks: dict[str, Download] = {}
        self._lock = threading.Lock()
        self._meta: dict[str, tuple[float, dict[str, Any]]] = {}

    @property
    def endpoint(self) -> str:
        value = (self.db.get_settings().get("hf_endpoint") or "https://huggingface.co").rstrip("/")
        return value

    # ------------------------------------------------------------ 元数据
    def repo_meta(self, repo: str) -> dict[str, Any]:
        cached = self._meta.get(repo)
        if cached and time.time() - cached[0] < _META_TTL:
            return cached[1]
        url = f"{self.endpoint}/api/models/{urllib.parse.quote(repo, safe='/')}?blobs=true"
        data = _http_json(url)
        files = [
            f for f in data.get("siblings", [])
            if isinstance(f.get("size"), int) and _matches_allow(f["rfilename"])
        ]
        config = data.get("config") or {}
        meta = {
            "id": repo,
            "size_bytes": sum(f["size"] for f in files),
            "weights_bytes": sum(f["size"] for f in files if f["rfilename"].endswith(".safetensors")),
            "model_type": config.get("model_type"),
            "gated": bool(data.get("gated")),
            "downloads": data.get("downloads"),
            "supported": _architecture_supported(config) if config else None,
        }
        self._meta[repo] = (time.time(), meta)
        return meta

    def _decorate(self, item: dict[str, Any], free: int, installed: set[str]) -> dict[str, Any]:
        item["fit"] = _fit(item["weights_bytes"], self.working_set)
        item["disk_ok"] = item["size_bytes"] + _DISK_MARGIN <= free
        item["installed"] = item["id"] in installed
        return item

    def recommended(self) -> list[dict[str, Any]]:
        free = disk_info(self.registry.download_cache())["free"]
        installed = {m["id"] for m in self.registry.scan()}
        items = []
        for repo, gigabytes, model_type, note in RECOMMENDED:
            size = int(gigabytes * 1e9)
            item = {
                "id": repo,
                "name": repo.rsplit("/", 1)[-1],
                "size_bytes": size,
                "weights_bytes": size,
                "model_type": model_type,
                "note": note,
                "supported": _architecture_supported({"model_type": model_type}),
            }
            if item["supported"]:
                items.append(self._decorate(item, free, installed))
        return items

    def search(self, query: str, limit: int = 18) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "search": query,
                "filter": "mlx",
                "pipeline_tag": "text-generation",
                "sort": "downloads",
                "direction": "-1",
                "limit": str(limit),
            }
        )
        data = _http_json(f"{self.endpoint}/api/models?{params}")
        repos = [m["id"] for m in data if REPO_ID.match(m.get("id", ""))]

        def fetch(repo: str) -> dict[str, Any] | None:
            try:
                return self.repo_meta(repo)
            except (urllib.error.URLError, OSError, ValueError):
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            metas = list(pool.map(fetch, repos))

        free = disk_info(self.registry.download_cache())["free"]
        installed = {m["id"] for m in self.registry.scan()}
        results = []
        for meta in metas:
            if not meta or not meta["weights_bytes"] or meta["gated"] or meta["supported"] is False:
                continue
            results.append(
                self._decorate(
                    {**meta, "name": meta["id"].rsplit("/", 1)[-1], "note": ""}, free, installed
                )
            )
        return results

    # -------------------------------------------------------------- 任务
    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [t.public() for t in self.tasks.values()]

    def start(self, repo: str) -> dict[str, Any]:
        if not REPO_ID.match(repo):
            raise ValueError("仓库名称不合法，应形如 mlx-community/模型名")
        with self._lock:
            for task in self.tasks.values():
                if task.repo == repo and task.status == "running":
                    return task.public()

        try:
            meta = self.repo_meta(repo)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise ValueError(f"找不到仓库 {repo}，或该仓库需要授权") from exc
            raise ValueError(f"无法访问模型仓库（HTTP {exc.code}）") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ValueError(
                f"连接 {self.endpoint} 失败。如果网络受限，可以在设置里切换到 hf-mirror.com 镜像。"
            ) from exc
        if not meta["weights_bytes"]:
            raise ValueError("这个仓库里没有 safetensors 权重文件，不是可用的 MLX 模型")
        if meta["supported"] is False:
            raise ValueError(f"mlx-lm 暂不支持这个模型架构：{meta['model_type']}")

        cache = self.registry.download_cache()
        cache.mkdir(parents=True, exist_ok=True)
        repo_dir = repo_cache_path(cache, repo)
        existing = _blob_bytes(repo_dir)
        needed = max(meta["size_bytes"] - existing, 0)
        free = disk_info(cache)["free"]
        if needed + _DISK_MARGIN > free:
            raise ValueError(
                f"磁盘空间不足：还需要 {needed / 1e9:.1f} GB，当前只剩 {free / 1e9:.1f} GB"
                "（下载后至少要保留 1 GB）。"
            )

        task = Download(repo, meta["size_bytes"], cache, meta["weights_bytes"])
        with self._lock:
            self.tasks[task.id] = task
        self.registry.hidden.add(repo)
        self.registry.invalidate()

        env = dict(os.environ, HF_ENDPOINT=self.endpoint)
        task.proc = subprocess.Popen(
            [
                sys.executable, "-m", "mlxui.download_worker", repo,
                "--cache-dir", str(cache), "--endpoint", self.endpoint,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=Path(__file__).resolve().parent.parent,  # 无论从哪里启动都能找到 mlxui 包
        )
        threading.Thread(target=self._watch, args=(task,), daemon=True).start()
        return task.public()

    def _watch(self, task: Download) -> None:
        repo_dir = repo_cache_path(task.cache, task.repo)
        last_bytes = _blob_bytes(repo_dir)
        last_time = time.time()
        task.done = last_bytes
        proc = task.proc
        assert proc is not None
        while proc.poll() is None:
            time.sleep(0.7)
            current = _blob_bytes(repo_dir)
            now = time.time()
            elapsed = max(now - last_time, 1e-3)
            rate = max(current - last_bytes, 0) / elapsed
            task.speed = rate if task.speed == 0 else task.speed * 0.7 + rate * 0.3
            task.done = current
            last_bytes, last_time = current, now

        stderr = proc.stderr.read().strip() if proc.stderr else ""
        task.speed = 0.0
        self.registry.hidden.discard(task.repo)
        self.registry.invalidate()
        if task.status == "cancelled":
            return
        if proc.returncode == 0:
            task.done = task.total
            task.status = "done"
        else:
            task.status = "failed"
            lines = [line for line in stderr.splitlines() if line.strip()]
            task.error = lines[-1][:300] if lines else "下载失败"

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task.status == "running" and task.proc is not None:
            task.status = "cancelled"
            task.proc.terminate()
            try:
                task.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                task.proc.kill()
            self.registry.hidden.discard(task.repo)
        return task.public()

    def discard(self, task_id: str) -> None:
        """移除任务记录并清理未完成的文件，把磁盘空间还给用户。"""
        task = self.tasks.pop(task_id, None)
        if task is None or task.status == "done":
            return
        repo_dir = repo_cache_path(task.cache, task.repo)
        if repo_dir.parent == task.cache and repo_dir.is_dir():
            shutil.rmtree(repo_dir, ignore_errors=True)
        self.registry.invalidate()


def _blob_bytes(repo_dir: Path) -> int:
    blobs = repo_dir / "blobs"
    total = 0
    try:
        with os.scandir(blobs) as entries:
            for entry in entries:
                try:
                    # 下载器若用 truncate 预分配文件，st_size 会虚高，取实际占用的较小值
                    stat = entry.stat(follow_symlinks=False)
                    total += min(stat.st_size, stat.st_blocks * 512)
                except OSError:
                    continue
    except OSError:
        return 0
    return total
