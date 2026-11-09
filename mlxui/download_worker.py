"""独立进程执行模型下载，这样取消时可以直接终止，也不会占用主进程的 GIL。"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download
from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--endpoint", default="")
    args = parser.parse_args()
    try:
        snapshot_download(
            args.repo,
            cache_dir=args.cache_dir or None,
            endpoint=args.endpoint or None,
            allow_patterns=DEFAULT_ALLOW_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001 - 错误信息经 stderr 交给父进程展示
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
