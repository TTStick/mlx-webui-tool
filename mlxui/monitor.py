"""系统资源监控：统一内存、MLX 显存、GPU 利用率、内存压力。"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import psutil

_PRESSURE = {1: "normal", 2: "warning", 4: "critical"}
_GPU_FIELDS = {
    "utilization": "Device Utilization %",
    "renderer": "Renderer Utilization %",
    "tiler": "Tiler Utilization %",
    "alloc_memory": "Alloc system memory",
    "in_use_memory": "In use system memory",
}
_TTL = 0.9


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=2
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _gpu_stats() -> dict[str, int]:
    """ioreg 不需要 sudo 就能读到 GPU 性能计数器。"""
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-w", "0"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    stats: dict[str, int] = {}
    for key, label in _GPU_FIELDS.items():
        match = re.search(rf'"{re.escape(label)}"=(\d+)', out)
        if match:
            stats[key] = int(match.group(1))
    return stats


class Monitor:
    def __init__(self, engine: Any):
        self.engine = engine
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] | None = None
        self._stamp = 0.0
        self._process = psutil.Process()
        psutil.cpu_percent(interval=None)  # 首次调用恒为 0，先垫一次

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._snapshot is not None and time.time() - self._stamp < _TTL:
                # 引擎状态变化比系统指标快，单独刷新
                return {**self._snapshot, "engine": self.engine.status()}
            self._snapshot = self._collect()
            self._stamp = time.time()
            return self._snapshot

    def _collect(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        info = mx.device_info()
        pressure = _sysctl_int("kern.memorystatus_vm_pressure_level")
        return {
            "time": time.time(),
            "system": {
                "total": vm.total,
                "available": vm.available,
                "used": vm.total - vm.available,
                "wired": getattr(vm, "wired", 0),
                "swap_used": swap.used,
                "swap_total": swap.total,
                "pressure": _PRESSURE.get(pressure or 1, "normal"),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "process_rss": self._process.memory_info().rss,
            },
            "gpu": _gpu_stats(),
            "mlx": {
                "active": int(mx.get_active_memory()),
                "peak": int(mx.get_peak_memory()),
                "cache": int(mx.get_cache_memory()),
                "working_set_limit": int(info.get("max_recommended_working_set_size", 0)),
                "device_name": info.get("device_name", ""),
            },
            "engine": self.engine.status(),
        }


def disk_info(path: Path) -> dict[str, int]:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {"total": usage.total, "free": usage.free, "used": usage.used}
