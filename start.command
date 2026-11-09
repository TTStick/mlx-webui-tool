#!/bin/bash
# 双击即可启动。首次运行会自动创建 Python 环境并安装依赖（约 450 MB），需要联网。
set -euo pipefail
cd "$(dirname "$0")"

# 双击启动时不会加载 shell 配置，手动补上 Homebrew 的路径
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
# Python 解释器放在项目目录内，不依赖系统环境，也避免 ~/.local 权限问题
export UV_PYTHON_INSTALL_DIR="$PWD/.python"

if [ "$(uname -m)" != "arm64" ]; then
  echo "MLX 只能运行在 Apple Silicon（M 系列芯片）的 Mac 上。"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv，请先运行：brew install uv"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "首次运行：正在创建 Python 环境…"
  uv venv --python 3.12 .venv
fi

if ! .venv/bin/python -c "import mlx_lm, fastapi, uvicorn, psutil, huggingface_hub, pypdf, sqlcipher3" >/dev/null 2>&1; then
  echo "正在安装依赖…"
  uv pip install --python .venv/bin/python -r pyproject.toml
fi

exec .venv/bin/python -m mlxui "$@"
