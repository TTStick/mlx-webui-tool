"""路径、默认参数与参数校验。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DATA_DIR = Path(
    os.environ.get("MLXUI_DATA_DIR")
    or Path.home() / "Library" / "Application Support" / "mlxui"
)
DB_PATH = DATA_DIR / "mlxui.db"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_ATTACHMENT_CHARS = 400_000

DEFAULT_PARAMS: dict[str, Any] = {
    "system_prompt": "",
    "context_length": 8192,
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "repetition_context_size": 64,
    "seed": None,
    "stop": [],
    "thinking": True,
    "memory": True,
    "web_search": False,
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "default_model": None,
    "params": DEFAULT_PARAMS,
    "hf_endpoint": "https://huggingface.co",
    "download_dir": "",
    "model_dirs": [],
    "auto_title": True,
    "kv_cache_limit_mb": 2048,
    "idle_unload_minutes": 0,
    "trust_remote_code": False,
    "memory": {
        "enabled": True,
        "review": False,
        "idle_seconds": 60,
        "budget_tokens": 400,
    },
    "search": {
        "provider": "wikipedia",
        "searxng_url": "",
        "brave_api_key": "",
        "tavily_api_key": "",
        "max_results": 5,
        "fetch_pages": 3,
    },
    "privacy": {
        "usage_stats": False,
        "auto_lock_minutes": 0,
    },
    "ui": {
        "theme": "system",
        "font_size": 15,
        "send_key": "enter",
        "show_stats": True,
        "inspector_open": True,
        "sidebar_open": True,
    },
}

_RANGES: dict[str, tuple[type, float, float]] = {
    "context_length": (int, 256, 4_000_000),
    "max_tokens": (int, 1, 4_000_000),
    "temperature": (float, 0.0, 2.0),
    "top_p": (float, 0.0, 1.0),
    "top_k": (int, 0, 100_000),
    "min_p": (float, 0.0, 1.0),
    "repetition_penalty": (float, 0.5, 2.0),
    "repetition_context_size": (int, 1, 4096),
}


def normalize_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    """只保留已知字段并裁剪到合法范围；非法值直接丢弃，回落到默认。"""
    out: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if key in _RANGES:
            kind, low, high = _RANGES[key]
            try:
                number = kind(value)
            except (TypeError, ValueError):
                continue
            out[key] = kind(min(max(number, low), high))
        elif key == "system_prompt":
            out[key] = str(value or "")
        elif key == "seed":
            if value in (None, ""):
                out[key] = None
            else:
                try:
                    out[key] = int(value)
                except (TypeError, ValueError):
                    continue
        elif key == "stop":
            if isinstance(value, str):
                value = [value]
            if isinstance(value, (list, tuple)):
                out[key] = [str(s) for s in value if str(s)][:8]
        elif key in ("thinking", "memory", "web_search"):
            out[key] = bool(value)
    return out


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
