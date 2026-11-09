"""本地模型扫描：Hugging Face 缓存、用户指定目录、LM Studio 目录。"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub.constants import HF_HUB_CACHE

from .db import DB

REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")
_CACHE_TTL = 4.0
_MAX_DEPTH = 3
_MAX_PROBLEMS = 8
_SKIPPED_DIRS = {"node_modules", "__pycache__"}
_PICK_PROMPT = "选择模型文件夹，或存放多个模型的上级文件夹"
_picker_lock = threading.Lock()


class FolderPickerError(Exception):
    pass


def choose_folder() -> str | None:
    """弹出 macOS 的文件夹选择窗口，返回 POSIX 路径；用户取消时返回 None。

    网页拿不到本机文件夹的真实路径，但服务就运行在这台 Mac 上，可以由后端来弹窗。
    """
    if not _picker_lock.acquire(blocking=False):
        raise FolderPickerError("文件夹选择窗口已经打开了，可能被浏览器窗口挡住")
    try:
        result = subprocess.run(
            [
                "/usr/bin/osascript",
                "-e", "tell me to activate",
                "-e", f'POSIX path of (choose folder with prompt "{_PICK_PROMPT}")',
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FolderPickerError("无法打开系统的文件夹窗口，请直接粘贴路径") from exc
    finally:
        _picker_lock.release()
    if result.returncode == 0:
        return result.stdout.strip() or None
    if "-128" in result.stderr:  # 用户点了取消
        return None
    raise FolderPickerError("无法打开系统的文件夹窗口，请直接粘贴路径")


def _walk(root: Path) -> Iterator[tuple[str, list[str], list[str]]]:
    """限定深度遍历，跟随符号链接并跳过隐藏目录；深度有上限，循环链接也不会卡死。"""
    for current, dirs, files in os.walk(root, followlinks=True):
        depth = len(Path(current).relative_to(root).parts)
        if depth >= _MAX_DEPTH:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIPPED_DIRS]
        yield current, dirs, files


def _resolved(path: str) -> str:
    return str(Path(path).resolve())


def _normalize_folder(raw: str) -> Path:
    """容忍常见的粘贴形式：带引号、file:// 链接、终端拖拽产生的转义空格。"""
    text = raw.strip().strip("'\"")
    if text.startswith("file://"):
        text = urllib.parse.unquote(text[len("file://") :])
    return Path(text.replace("\\ ", " ")).expanduser()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _architecture_supported(config: dict[str, Any]) -> bool:
    if "model_type" not in config:
        return False
    try:
        from mlx_lm.utils import _get_classes

        _get_classes(config)
        return True
    except (ValueError, ImportError, KeyError, AttributeError):
        return False


def _kv_bytes_per_token(config: dict[str, Any]) -> int | None:
    """由结构参数估算 KV 缓存每 token 的字节数：层数×2×(KV头数×头维度)×float16。"""
    tc = config.get("text_config") or config
    layers = tc.get("num_hidden_layers") or tc.get("n_layer") or tc.get("num_layers")
    hidden = tc.get("hidden_size") or tc.get("n_embd") or tc.get("d_model")
    heads = tc.get("num_attention_heads") or tc.get("n_head")
    if isinstance(layers, list):
        layers = len(layers)
    if not (layers and hidden and heads):
        return None
    kv_heads = tc.get("num_key_value_heads") or heads
    head_dim = tc.get("head_dim") or hidden // heads
    return int(layers) * 2 * int(kv_heads) * int(head_dim) * 2


def _max_context(config: dict[str, Any], tokenizer_config: dict[str, Any]) -> int | None:
    tc = config.get("text_config") or config
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        value = tc.get(key)
        if isinstance(value, int) and value > 0:
            return value
    value = tokenizer_config.get("model_max_length")
    if isinstance(value, (int, float)) and 0 < value < 10_000_000:
        return int(value)
    return None


def _quantization(config: dict[str, Any], name: str) -> str:
    quant = config.get("quantization") or config.get("quantization_config") or {}
    bits = quant.get("bits") if isinstance(quant, dict) else None
    if bits:
        return f"{bits}-bit"
    match = re.search(r"(\d+)[-_]?bit", name, re.I)
    if match:
        return f"{match.group(1)}-bit"
    dtype = config.get("torch_dtype") or config.get("dtype")
    return str(dtype).replace("torch.", "") if dtype else "原始精度"


def _chat_template_text(path: Path, tokenizer_config: dict[str, Any]) -> str:
    template = tokenizer_config.get("chat_template") or ""
    if isinstance(template, list):
        template = " ".join(str(t.get("template", "")) for t in template if isinstance(t, dict))
    jinja = path / "chat_template.jinja"
    if jinja.is_file():
        try:
            template = f"{template}\n{jinja.read_text(encoding='utf-8')}"
        except OSError:
            pass
    return str(template)


def _is_complete(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(_read_json(index).get("weight_map", {}).values())
        return bool(shards) and all((path / s).exists() for s in shards)
    return any(f.exists() for f in path.glob("*.safetensors"))


def _missing_shards(path: Path) -> list[str]:
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return []
    shards = set(_read_json(index).get("weight_map", {}).values())
    return sorted(s for s in shards if not (path / s).exists())


def _has_tokenizer(path: Path) -> bool:
    for entry in path.iterdir():
        name = entry.name
        if name == "tokenizer_config.json":  # 只有配置、没有词表时加载不了
            continue
        if (
            name.startswith("tokenizer")
            or name.endswith((".model", ".tiktoken"))
            or name in ("vocab.json", "vocab.txt", "tekken.json")
        ):
            return True
    return False


def _display_name(model_id: str, path: Path) -> str:
    # HF 缓存结构被直接选中时，最后一级是提交哈希，要取回仓库名
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name.rsplit("--", 1)[-1]
    return model_id.rsplit("/", 1)[-1]


def _folder_problem(path: Path, files: list[str]) -> str | None:
    """解释疑似模型的文件夹为什么不能用；看起来不像模型的文件夹返回 None。"""
    has_weights = any(f.endswith(".safetensors") for f in files)
    config = _read_json(path / "config.json") if "config.json" in files else None
    if not has_weights:
        if any(f.lower().endswith(".gguf") for f in files):
            return "这是 GGUF 格式（llama.cpp 使用），MLX 无法加载。请下载同一模型的 MLX 版本。"
        if any(re.fullmatch(r"(pytorch_model|model).*\.bin", f) for f in files):
            return "这是 PyTorch 的 .bin 权重，MLX 需要 .safetensors 格式。请下载 MLX 版本的模型。"
        if config and "model_type" in config:
            return "有 config.json，但没有 .safetensors 权重文件。"
        return None
    if config is None:
        return "缺少 config.json。"
    missing = _missing_shards(path)
    if missing:
        return f"权重分片不完整，缺少 {len(missing)} 个文件（例如 {missing[0]}）。"
    if not _architecture_supported(config):
        return f"mlx-lm 暂不支持这个模型架构：{config.get('model_type') or '未知'}。"
    if not _has_tokenizer(path):
        return "缺少分词器文件（tokenizer.json 等）。"
    return None


def _describe(model_id: str, path: Path, source: str, deletable: bool) -> dict[str, Any] | None:
    config_path = path / "config.json"
    if not config_path.is_file() or not _is_complete(path):
        return None
    config = _read_json(config_path)
    if not _architecture_supported(config) or not _has_tokenizer(path):
        return None
    tokenizer_config = _read_json(path / "tokenizer_config.json")
    template = _chat_template_text(path, tokenizer_config)

    weights = 0
    total = 0
    modified = 0.0
    for entry in path.iterdir():
        try:
            stat = entry.stat()  # 跟随 HF 缓存的符号链接
        except OSError:
            continue
        if not entry.is_file():
            continue
        total += stat.st_size
        modified = max(modified, stat.st_mtime)
        if entry.suffix == ".safetensors":
            weights += stat.st_size

    return {
        "id": model_id,
        "name": _display_name(model_id, path),
        "path": str(path),
        "source": source,
        "deletable": deletable,
        "model_type": config.get("model_type"),
        "size_bytes": total,
        "weights_bytes": weights,
        "quantization": _quantization(config, model_id),
        "max_context": _max_context(config, tokenizer_config),
        "kv_bytes_per_token": _kv_bytes_per_token(config),
        "thinking_toggle": "enable_thinking" in template,
        "reasoning": "<think>" in template or "enable_thinking" in template,
        "has_template": bool(template.strip()),
        "modified": modified,
    }


class ModelRegistry:
    def __init__(self, db: DB):
        self.db = db
        self._lock = threading.Lock()
        self._cache: list[dict[str, Any]] = []
        self._stamp = 0.0
        self.hidden: set[str] = set()  # 正在下载的仓库不算已安装

    def hub_caches(self) -> list[Path]:
        caches = [Path(HF_HUB_CACHE)]
        custom = self.db.get_settings().get("download_dir") or ""
        if custom:
            caches.append(Path(custom).expanduser())
        return caches

    def download_cache(self) -> Path:
        return self.hub_caches()[-1]

    def _scan_hub(self, cache: Path) -> list[dict[str, Any]]:
        found = []
        if not cache.is_dir():
            return found
        for repo in sorted(cache.glob("models--*")):
            repo_id = repo.name[len("models--") :].replace("--", "/", 1)
            if repo_id in self.hidden:
                continue
            snapshots = repo / "snapshots"
            if not snapshots.is_dir():
                continue
            revision = None
            ref = repo / "refs" / "main"
            if ref.is_file():
                revision = ref.read_text().strip()
            candidates = (
                [snapshots / revision]
                if revision and (snapshots / revision).is_dir()
                else sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            )
            for snapshot in candidates:
                info = _describe(repo_id, snapshot, "huggingface", True)
                if info:
                    info["repo_path"] = str(repo)
                    found.append(info)
                    break
        return found

    def _scan_dir(self, root: Path) -> list[dict[str, Any]]:
        found = []
        if not root.is_dir():
            return found
        for current, dirs, files in _walk(root):
            if "config.json" in files and any(f.endswith(".safetensors") for f in files):
                info = _describe(current, Path(current), "local", False)
                if info:
                    found.append(info)
                dirs[:] = []
        return found

    def scan(self, refresh: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if not refresh and self._cache and time.time() - self._stamp < _CACHE_TTL:
                return self._cache
            models: dict[str, dict[str, Any]] = {}
            for cache in self.hub_caches():
                for info in self._scan_hub(cache):
                    models.setdefault(info["id"], info)
            roots = [Path(p).expanduser() for p in self.db.get_settings().get("model_dirs", [])]
            lmstudio = Path.home() / ".lmstudio" / "models"
            if lmstudio.is_dir():
                roots.append(lmstudio)
            seen = {_resolved(m["path"]) for m in models.values()}
            for root in roots:
                for info in self._scan_dir(root):
                    resolved = _resolved(info["path"])
                    if resolved not in seen:  # 同一个模型只登记一次
                        seen.add(resolved)
                        models[info["id"]] = info
            self._cache = sorted(models.values(), key=lambda m: m["name"].lower())
            self._stamp = time.time()
            return self._cache

    def get(self, model_id: str | None) -> dict[str, Any] | None:
        if not model_id:
            return None
        for info in self.scan():
            if info["id"] == model_id:
                return info
        for info in self.scan(refresh=True):
            if info["id"] == model_id:
                return info
        return None

    def invalidate(self) -> None:
        with self._lock:
            self._stamp = 0.0

    # ------------------------------------------------------ 手动添加的文件夹
    def inspect_folder(self, raw: str) -> dict[str, Any]:
        """找出文件夹里可用的模型，并说明其余疑似模型为什么不能用。"""
        root = _normalize_folder(raw)

        def failed(message: str) -> dict[str, Any]:
            return {"path": str(root), "models": [], "problems": [{"path": str(root), "message": message}]}

        if not root.is_absolute():
            return failed("请填写完整路径，例如 /Users/你的用户名/models/Qwen3-4B。")
        try:
            if root.is_file():
                root = root.parent
            if not root.is_dir():
                return failed("这个文件夹不存在。")
            os.listdir(root)  # os.walk 会吞掉权限错误，这里提前暴露
        except PermissionError:
            return failed("没有权限读取这个文件夹。请在“系统设置 → 隐私与安全性 → 文件和文件夹”里允许终端访问。")
        except OSError as exc:
            return failed(f"无法读取这个文件夹：{exc.strerror or exc}")

        models = self._scan_dir(root)
        usable = {m["path"] for m in models}
        problems: list[dict[str, str]] = []
        for current, dirs, files in _walk(root):
            if current in usable:
                dirs[:] = []
                continue
            message = _folder_problem(Path(current), files)
            if message:
                problems.append({"path": current, "message": message})
                dirs[:] = []
                if len(problems) >= _MAX_PROBLEMS:
                    break
        if not models and not problems:
            problems.append({"path": str(root), "message": f"这里没有找到模型（最多往下找 {_MAX_DEPTH} 层子文件夹）。"})
        return {"path": str(root), "models": models, "problems": problems}

    def add_folder(self, raw: str) -> dict[str, Any]:
        result = self.inspect_folder(raw)
        dirs = list(self.db.get_settings().get("model_dirs", []))
        known = {_resolved(m["path"]) for m in self.scan(refresh=True)}
        fresh = [m for m in result["models"] if _resolved(m["path"]) not in known]
        if fresh:
            dirs.append(result["path"])
            self.db.update_settings({"model_dirs": dirs})
        # 以注册表为准：同一模型可能已经以 Hugging Face 仓库名登记过
        registered = {_resolved(m["path"]): m for m in self.scan(refresh=True)}
        result["models"] = [
            registered[_resolved(m["path"])] for m in result["models"] if _resolved(m["path"]) in registered
        ]
        result.update(added=len(fresh), model_dirs=dirs)
        return result

    def remove_folder(self, path: str) -> list[str]:
        """只从列表移除，不会动磁盘上的文件。"""
        dirs = [d for d in self.db.get_settings().get("model_dirs", []) if d != path]
        self.db.update_settings({"model_dirs": dirs})
        self.invalidate()
        return dirs

    def local_dirs(self) -> list[dict[str, Any]]:
        models = [m for m in self.scan() if m["source"] == "local"]

        def row(raw: str, removable: bool) -> dict[str, Any]:
            root = Path(raw).expanduser()
            return {
                "path": raw,
                "count": sum(1 for m in models if Path(m["path"]).is_relative_to(root)),
                "exists": root.is_dir(),
                "removable": removable,
            }

        rows = [row(d, True) for d in self.db.get_settings().get("model_dirs", [])]
        lmstudio = Path.home() / ".lmstudio" / "models"
        if lmstudio.is_dir():
            rows.append(row(str(lmstudio), False))
        return rows
