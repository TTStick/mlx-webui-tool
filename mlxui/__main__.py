"""启动入口：python -m mlxui [--port 8420] [--no-browser]"""
from __future__ import annotations

import argparse
import logging
import os
import resource
import socket
import threading
import time
import urllib.request
import webbrowser


def _harden_process() -> None:
    """尽量不在磁盘上留下痕迹：私有文件权限、关闭遥测、禁止核心转储。"""
    os.umask(0o077)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    # 只输出到终端且只有错误类型，不会把对话内容写进任何日志
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        if _port_free(host, port):
            return port
    raise SystemExit(f"端口 {preferred} 起的 50 个端口都被占用了，请用 --port 指定")


def _open_when_ready(url: str) -> None:
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"{url}/api/vault/status", timeout=1):
                break
        except OSError:
            time.sleep(0.2)
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlxui", description="本地 MLX 大模型聊天界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    _harden_process()
    port = _pick_port(args.host, args.port)
    exposed = args.host not in ("127.0.0.1", "localhost", "::1")
    if exposed:
        print("⚠️  已监听非本机地址，局域网内任何人都能使用这个服务，并且没有登录验证。")

    # 延迟导入：MLX 和 transformers 加载要几秒，端口问题应该先报出来
    import uvicorn

    from .app import create_app

    app = create_app(allow_any_host=exposed)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{display_host}:{port}"
    print(f"MLX UI 已启动：{url}")
    print(f"OpenAI 兼容接口：{url}/v1")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="critical",  # 访问日志和未处理异常的详情都不输出
        access_log=False,
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
