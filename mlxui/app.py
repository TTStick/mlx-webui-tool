"""HTTP 接口：Web 界面用的 /api 与 OpenAI 兼容的 /v1。"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
import shutil
import time
import urllib.error
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlx.core as mx
import mlx_lm
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, websearch
from .attachments import AttachmentError, extract_text
from .config import DB_PATH, MAX_UPLOAD_BYTES, normalize_params
from .context import ContextResult, TokenCounter, Turn, build_context
from .db import DB, DatabaseLocked, WrongKey
from .downloads import DownloadManager, _fit
from .engine import Engine, Job, ModelTooLarge, device_info
from .memory import KINDS, MemoryService
from .models import FolderPickerError, ModelRegistry, choose_folder
from .monitor import Monitor, disk_info
from .service import BadRequest, Busy, ChatService, NotFound, UsageSink
from .vault import MIN_PASSWORD_LENGTH, KeyStore, KeyStoreUnavailable, Vault, filevault_enabled

log = logging.getLogger("mlxui.app")

STATIC_DIR = Path(__file__).parent / "static"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_BACKGROUND_POLLS = {"/api/monitor", "/api/downloads"}  # 轮询不算用户在操作，否则永远不会自动锁定
_MAX_IMPORT_BYTES = 512 * 1024 * 1024


class NoCacheStatic(StaticFiles):
    """前端没有构建步骤，强制浏览器每次校验，升级后不会读到旧脚本。"""

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class LocalGuard:
    """拒绝跨站请求和 DNS 重绑定，防止恶意网页操作本机服务。"""

    def __init__(self, app: Any, allow_any_host: bool, vault: Vault):
        self.app = app
        self.allow_any_host = allow_any_host
        self.vault = vault

    @staticmethod
    def _host(value: str) -> str:
        if value.startswith("["):
            return value[1:].split("]", 1)[0].lower()
        return value.split(":", 1)[0].lower()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            host = headers.get("host", "")
            origin = headers.get("origin")
            allowed = self.allow_any_host or self._host(host) in _LOCAL_HOSTS
            if allowed and origin and origin != "null":
                allowed = urlparse(origin).netloc.lower() == host.lower()
            if not allowed:
                response = JSONResponse({"detail": "forbidden"}, status_code=403)
                await response(scope, receive, send)
                return

            path = scope["path"]
            is_api = path.startswith(("/api/", "/v1/"))
            if is_api and not path.startswith("/api/vault/") and not self.vault.unlocked:
                response = JSONResponse({"detail": "已锁定", "code": "locked"}, status_code=423)
                await response(scope, receive, send)
                return
            if is_api and path not in _BACKGROUND_POLLS:
                self.vault.touch()

            async def send_private(message: Any) -> None:
                # 接口响应里都是对话内容，不能留在浏览器的磁盘缓存里
                if is_api and message["type"] == "http.response.start":
                    kept = [(k, v) for k, v in message["headers"] if k.lower() != b"cache-control"]
                    message = {**message, "headers": [*kept, (b"cache-control", b"no-store")]}
                await send(message)

            await self.app(scope, receive, send_private)
            return
        await self.app(scope, receive, send)


def _text_of(content: Any) -> str:
    """OpenAI 的 content 可能是字符串，也可能是多段内容的列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def create_app(
    allow_any_host: bool = False, db_path: Path = DB_PATH, keystore: KeyStore | None = None
) -> FastAPI:
    db = DB(db_path)
    vault = Vault(db_path.parent, keystore)
    registry = ModelRegistry(db)
    engine = Engine(db, registry)
    monitor = Monitor(engine)
    working_set = int(device_info().get("max_recommended_working_set_size", 0))
    downloads = DownloadManager(db, registry, working_set)
    memory = MemoryService(db)
    service = ChatService(db, engine, registry, memory)
    filevault: dict[str, bool | None] = {}

    def auto_open() -> None:
        """钥匙串模式自动解锁；取不到密钥时停在对应的状态，由界面引导。"""
        try:
            key = vault.auto_key()
        except KeyStoreUnavailable:
            return
        if key is None:
            return
        try:
            db.open(key)
        except WrongKey:
            vault.forget()
            return
        db.mark_interrupted()

    def seal() -> None:
        """清掉内存里的敏感内容并关闭数据库，必须在工作线程内执行。"""
        engine.purge_sensitive()
        db.close()
        vault.forget()

    def idle_tick(worker: Engine) -> None:
        minutes = db.get_settings()["privacy"]["auto_lock_minutes"]
        if minutes and vault.mode == "password" and time.time() - vault.last_activity > minutes * 60:
            seal()
            return
        memory.idle_tick(worker)

    engine.idle_hook = idle_tick
    auto_open()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for task in list(downloads.tasks.values()):
            downloads.cancel(task.id)
        engine.shutdown()

    app = FastAPI(title="MLX UI", version=__version__, lifespan=lifespan)
    app.state.db = db
    app.state.engine = engine
    app.state.registry = registry
    app.state.service = service
    app.state.downloads = downloads
    app.state.memory = memory
    app.state.vault = vault
    app.add_middleware(LocalGuard, allow_any_host=allow_any_host, vault=vault)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.error("request failed: %s", type(exc).__name__)  # 只记类型，异常信息里可能有用户内容
        return JSONResponse({"detail": "服务内部错误"}, status_code=500)

    @app.exception_handler(DatabaseLocked)
    async def _locked(_: Request, exc: DatabaseLocked) -> JSONResponse:
        return JSONResponse({"detail": "已锁定", "code": "locked"}, status_code=423)

    @app.exception_handler(Busy)
    async def _busy(_: Request, exc: Busy) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(BadRequest)
    async def _bad(_: Request, exc: BadRequest) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(NotFound)
    async def _missing(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    # ------------------------------------------------------ 状态与监控
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        info = device_info()
        return {
            "version": __version__,
            "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__,
            "python": platform.python_version(),
            "device": info.get("device_name"),
            "memory_size": info.get("memory_size"),
            "working_set_limit": info.get("max_recommended_working_set_size"),
            "disk": disk_info(registry.download_cache()),
            "data_dir": str(db_path.parent),
            "download_cache": str(registry.download_cache()),
        }

    @app.get("/api/monitor")
    def monitor_snapshot() -> dict[str, Any]:
        return {**monitor.snapshot(), "memory": memory.status()}

    # ---------------------------------------------------------- 加密与隐私
    def vault_status() -> dict[str, Any]:
        if "value" not in filevault:
            filevault["value"] = filevault_enabled()
        return {
            "state": vault.state,
            "mode": vault.mode,
            "filevault": filevault["value"],
            "min_password": MIN_PASSWORD_LENGTH,
        }

    def require_password(password: str) -> str:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise HTTPException(400, f"密码至少需要 {MIN_PASSWORD_LENGTH} 位")
        return password

    @app.get("/api/vault/status")
    def get_vault_status() -> dict[str, Any]:
        return vault_status()

    @app.post("/api/vault/unlock")
    async def unlock_vault(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if vault.state != "locked":
            return vault_status()
        key = await asyncio.to_thread(vault.derive, str(body.get("password", "")))
        try:
            await asyncio.to_thread(db.open, key)
        except WrongKey as exc:
            await asyncio.sleep(0.5)
            raise HTTPException(401, "密码不正确") from exc
        vault.key = key
        vault.touch()
        db.mark_interrupted()
        return vault_status()

    @app.post("/api/vault/retry")
    def retry_vault() -> dict[str, Any]:
        """钥匙串暂时读不到时（比如还没登录完成）重新试一次，不需要重启。"""
        if vault.state == "key_missing":
            auto_open()
        return vault_status()

    @app.post("/api/vault/setup")
    async def setup_vault(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """钥匙串不可用时的首次设置。"""
        if vault.state != "setup":
            raise HTTPException(409, "已经设置过了")
        key, meta = await asyncio.to_thread(vault.prepare_password, require_password(str(body.get("password", ""))))
        await asyncio.to_thread(db.open, key)
        vault.commit_password(key, meta)
        db.mark_interrupted()
        return vault_status()

    def _verify_current(password: str) -> None:
        if vault.derive(password) != vault.key:
            raise HTTPException(401, "当前密码不正确")

    @app.post("/api/vault/password")
    async def set_password(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """开启或修改启动密码。整个数据库会用新密钥重新加密。"""
        if not vault.unlocked:
            raise HTTPException(423, "已锁定")
        if vault.mode == "password":
            await asyncio.to_thread(_verify_current, str(body.get("current", "")))
        key, meta = await asyncio.to_thread(vault.prepare_password, require_password(str(body.get("password", ""))))
        await asyncio.wrap_future(engine.submit(lambda: db.rekey(key)))
        vault.commit_password(key, meta)
        return vault_status()

    @app.post("/api/vault/password/remove")
    async def remove_password(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """取消启动密码，改用钥匙串自动解锁。"""
        if vault.mode != "password" or not vault.unlocked:
            raise HTTPException(409, "当前没有启动密码")
        await asyncio.to_thread(_verify_current, str(body.get("current", "")))
        key = vault.new_random_key()
        await asyncio.wrap_future(engine.submit(lambda: db.rekey(key)))
        try:
            vault.commit_keychain(key)
        except KeyStoreUnavailable as exc:
            raise HTTPException(503, "无法写入钥匙串") from exc
        return vault_status()

    @app.post("/api/vault/rotate")
    async def rotate_key() -> dict[str, Any]:
        """换一把新密钥重写数据库，之后磁盘上残留的旧密文就无法再解开。"""
        if vault.mode != "keychain" or not vault.unlocked:
            raise HTTPException(409, "启动密码模式下，请通过修改密码来更换密钥")
        key = vault.new_random_key()
        await asyncio.wrap_future(engine.submit(lambda: db.rekey(key)))
        try:
            vault.commit_keychain(key)
        except KeyStoreUnavailable as exc:
            raise HTTPException(503, "无法写入钥匙串") from exc
        return vault_status()

    @app.post("/api/vault/lock")
    async def lock_vault() -> dict[str, Any]:
        if vault.mode != "password":
            raise HTTPException(409, "只有设置了启动密码才能锁定")
        engine.cancel_all()
        await asyncio.wrap_future(engine.submit(seal))
        return vault_status()

    @app.post("/api/vault/wipe")
    async def wipe_vault(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """销毁密钥并删除数据库。这是不可恢复的。"""
        if body.get("confirm") != "抹除":
            raise HTTPException(400, "需要确认才能抹除")
        engine.cancel_all()

        def wipe() -> None:
            engine.purge_sensitive()
            db.wipe()

        await asyncio.wrap_future(engine.submit(wipe))
        vault.destroy()
        memory.version += 1
        auto_open()
        return vault_status()

    # -------------------------------------------------------------- 记忆
    @app.get("/api/memory")
    def list_memories(status: str = "active", kind: str = "", q: str = "") -> dict[str, Any]:
        return {
            "memories": memory.list(status, kind, q.strip()),
            "counts": memory.counts(),
            "kinds": KINDS,
            "status": memory.status(),
        }

    @app.post("/api/memory")
    def add_memory(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return memory.add_manual(
                str(body.get("kind", "profile")), str(body.get("content", "")), int(body.get("importance", 3))
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.patch("/api/memory/{memory_id}")
    def patch_memory(memory_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            updated = memory.update(memory_id, body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if updated is None:
            raise HTTPException(404, "记忆不存在")
        return updated

    @app.delete("/api/memory/{memory_id}")
    def delete_memory(memory_id: str) -> dict[str, Any]:
        memory.delete(memory_id)
        return {"ok": True}

    @app.post("/api/memory/bulk")
    def bulk_memory(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return {"count": memory.bulk([str(i) for i in body.get("ids", [])], str(body.get("action", "")))}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/memory/clear")
    def clear_memory() -> dict[str, Any]:
        memory.clear()
        return {"ok": True}

    @app.post("/api/memory/seen")
    def memory_seen() -> dict[str, Any]:
        memory.mark_seen()
        return memory.status()

    @app.post("/api/memory/reflect")
    async def reflect_memory() -> dict[str, Any]:
        model = db.get_settings().get("default_model")
        try:
            return await asyncio.wrap_future(engine.submit(lambda: memory.reflect_now(engine, model)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ---------------------------------------------------------- 模型
    def _present(model: dict[str, Any]) -> dict[str, Any]:
        return {
            **model,
            "fit": _fit(model["weights_bytes"], working_set),
            "loaded": model["id"] == engine.model_id,
            "kv_bytes_per_token_measured": engine.kv_bytes_per_token.get(model["id"]),
        }

    @app.get("/api/models")
    def list_models(refresh: bool = False) -> dict[str, Any]:
        return {
            "models": [_present(m) for m in registry.scan(refresh=refresh)],
            "local_dirs": registry.local_dirs(),
            "working_set_limit": working_set,
            "disk": disk_info(registry.download_cache()),
        }

    @app.post("/api/models/local")
    def add_local_model(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        path = str(body.get("path", "")).strip()
        if not path:
            raise HTTPException(400, "请选择或填写模型文件夹的路径")
        result = registry.add_folder(path)
        result["models"] = [_present(m) for m in result["models"]]
        return result

    @app.post("/api/models/local/remove")
    def remove_local_folder(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        dirs = registry.remove_folder(str(body.get("path", "")))
        default = db.get_settings().get("default_model")
        if default and registry.get(default) is None:
            db.update_settings({"default_model": None})
        return {"model_dirs": dirs, "default_model": db.get_settings().get("default_model")}

    @app.post("/api/system/pick-folder")
    async def pick_folder(request: Request) -> dict[str, Any]:
        # 弹窗出现在运行服务的 Mac 上，开放局域网访问时不能让别的设备触发
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(403, "只能在运行服务的这台 Mac 上使用文件夹选择窗口")
        try:
            return {"path": await asyncio.to_thread(choose_folder)}
        except FolderPickerError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/models/load")
    async def load_model(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return await asyncio.wrap_future(
                engine.load_model(str(body.get("id", "")), force=bool(body.get("force")))
            )
        except ModelTooLarge as exc:
            return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=409)  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001 - 加载失败原因直接展示给用户
            raise HTTPException(500, f"加载失败：{exc}") from exc

    @app.post("/api/models/unload")
    async def unload_model() -> dict[str, Any]:
        return await asyncio.wrap_future(engine.unload_model())

    @app.post("/api/models/clear-cache")
    async def clear_cache() -> dict[str, Any]:
        return await asyncio.wrap_future(engine.clear_kv_cache())

    @app.post("/api/models/delete")
    async def delete_model(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        model_id = str(body.get("id", ""))
        info = registry.get(model_id)
        if info is None:
            raise HTTPException(404, "模型不存在")
        if not info["deletable"]:
            raise HTTPException(400, "这是外部目录里的模型，请在访达中自行管理")
        target = Path(info["repo_path"])
        if target.parent not in registry.hub_caches() or not target.name.startswith("models--"):
            raise HTTPException(400, "模型路径不在受管理的缓存目录内")
        if engine.model_id == model_id:
            await asyncio.wrap_future(engine.unload_model())
        await asyncio.to_thread(shutil.rmtree, target)
        registry.invalidate()
        if db.get_settings().get("default_model") == model_id:
            db.update_settings({"default_model": None})
        return {"ok": True, "disk": disk_info(registry.download_cache())}

    @app.get("/api/models/recommended")
    def recommended_models() -> dict[str, Any]:
        return {"models": downloads.recommended()}

    @app.get("/api/models/search")
    def search_models(q: str) -> dict[str, Any]:
        try:
            return {"models": downloads.search(q)}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise HTTPException(
                502, f"无法连接 {downloads.endpoint}。如果网络受限，可在设置中切换到 hf-mirror.com。"
            ) from exc

    @app.get("/api/downloads")
    def list_downloads() -> dict[str, Any]:
        return {"tasks": downloads.list()}

    @app.post("/api/downloads")
    def start_download(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return downloads.start(str(body.get("repo", "")).strip())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/downloads/{task_id}/cancel")
    def cancel_download(task_id: str) -> dict[str, Any]:
        result = downloads.cancel(task_id)
        if result is None:
            raise HTTPException(404, "下载任务不存在")
        downloads.discard(task_id)
        return result

    @app.delete("/api/downloads/{task_id}")
    def dismiss_download(task_id: str) -> dict[str, Any]:
        downloads.discard(task_id)
        return {"ok": True}

    # ---------------------------------------------------------- 对话
    @app.get("/api/conversations")
    def list_conversations(archived: bool = False) -> dict[str, Any]:
        active = {j.conversation_id for j in engine.jobs.values() if not j.finished}
        items = db.list_conversations(archived)
        for item in items:
            item["generating"] = item["id"] in active
        return {"conversations": items}

    @app.post("/api/conversations")
    def create_conversation(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        settings = normalize_params(body.get("settings"))
        model = body.get("model") or db.get_settings().get("default_model")
        return db.create_conversation(model=model, settings=settings)

    @app.get("/api/conversations/{conv_id}")
    def get_conversation(conv_id: str) -> dict[str, Any]:
        return service.detail(conv_id)

    @app.patch("/api/conversations/{conv_id}")
    def patch_conversation(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        conversation = db.get_conversation(conv_id)
        if conversation is None:
            raise HTTPException(404, "对话不存在")
        updates: dict[str, Any] = {}
        if "title" in body:
            title = str(body["title"]).strip()
            if not title:
                raise HTTPException(400, "标题不能为空")
            updates["title"] = title[:120]
            updates["title_source"] = "user"
        if "pinned" in body:
            updates["pinned"] = bool(body["pinned"])
        if "model" in body:
            updates["model"] = body["model"] or None
        if "settings" in body:
            # null 表示清除这个字段的覆盖，恢复全局默认
            merged = {**conversation["settings"], **normalize_params(body["settings"])}
            for key, value in body["settings"].items():
                if value is None and key != "seed":
                    merged.pop(key, None)
            updates["settings"] = merged
        if updates:
            db.update_conversation(conv_id, **updates)
        if "archived" in body:
            db.set_archived(conv_id, bool(body["archived"]))
        return service.detail(conv_id)

    @app.delete("/api/conversations/{conv_id}")
    def delete_conversation(conv_id: str) -> dict[str, Any]:
        return {"ok": True, "memories_removed": service.delete_conversation(conv_id)}

    @app.post("/api/conversations/bulk")
    def bulk_conversations(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        ids = [str(i) for i in body.get("ids", [])]
        action = str(body.get("action", ""))
        if action not in ("archive", "unarchive", "delete", "pin", "unpin"):
            raise HTTPException(400, "不支持的操作")
        if action == "delete":
            removed = sum(service.delete_conversation(conv_id) for conv_id in ids)
            return {"count": len(ids), "memories_removed": removed}
        return {"count": db.bulk(ids, action)}

    @app.get("/api/search")
    def search(q: str) -> dict[str, Any]:
        return {"results": db.search(q.strip()) if q.strip() else []}

    @app.post("/api/conversations/{conv_id}/send")
    def send_message(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.send(
            conv_id,
            str(body.get("content", "")),
            list(body.get("attachments") or []),
            force=bool(body.get("force")),
        )

    @app.post("/api/conversations/{conv_id}/regenerate")
    def regenerate(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.regenerate(conv_id, str(body.get("message_id")), force=bool(body.get("force")))

    @app.post("/api/conversations/{conv_id}/edit")
    def edit(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.edit(
            conv_id, str(body.get("message_id")), str(body.get("content", "")), force=bool(body.get("force"))
        )

    @app.post("/api/conversations/{conv_id}/continue")
    def continue_reply(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.continue_reply(conv_id, str(body.get("message_id")), force=bool(body.get("force")))

    @app.post("/api/conversations/{conv_id}/switch")
    def switch_branch(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.switch_branch(conv_id, str(body.get("message_id")))

    @app.post("/api/conversations/{conv_id}/delete-message")
    def delete_message(conv_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.delete_message(conv_id, str(body.get("message_id")))

    @app.post("/api/conversations/{conv_id}/compact")
    def compact(conv_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return service.compact(conv_id, force=bool(body.get("force")))

    @app.post("/api/conversations/{conv_id}/clear-summary")
    def clear_summary(conv_id: str) -> dict[str, Any]:
        return service.clear_summary(conv_id)

    @app.post("/api/context/preview")
    def context_preview(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return service.preview(
            conversation_id=body.get("conversation_id"),
            draft=str(body.get("draft", "")),
            attachments=list(body.get("attachments") or []),
            model=body.get("model"),
            settings=body.get("settings"),
            include_prompt=bool(body.get("include_prompt")),
        )

    # ---------------------------------------------------------- 任务流
    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request, after: int = -1) -> StreamingResponse:
        job = engine.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在或已结束")

        async def stream():
            cursor = after + 1
            wake = job.subscribe()
            try:
                while True:
                    batch = job.slice(cursor)
                    for event in batch:
                        yield f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                        cursor = event["seq"] + 1
                    if job.finished and not job.slice(cursor):
                        return
                    wake.clear()
                    if job.slice(cursor):
                        continue
                    try:
                        await asyncio.wait_for(wake.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                    if await request.is_disconnected():
                        return
            finally:
                job.unsubscribe(wake)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        job = engine.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在或已结束")
        engine.stop(job)
        return {"ok": True}

    # ---------------------------------------------------------- 附件
    async def read_body(request: Request, limit: int) -> bytes:
        """整体读进内存。不用 multipart：它会把大文件以明文暂存到系统临时目录。"""
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > limit:
                raise HTTPException(413, f"文件过大，上限 {limit // 1024 // 1024} MB")
        return bytes(data)

    @app.post("/api/attachments")
    async def upload_attachment(request: Request, name: str = "file") -> dict[str, Any]:
        data = await read_body(request, MAX_UPLOAD_BYTES)
        try:
            text, truncated = await asyncio.to_thread(extract_text, name, data)
        except AttachmentError as exc:
            raise HTTPException(415, str(exc)) from exc
        return {
            "id": uuid.uuid4().hex,
            "name": name,
            "size": len(data),
            "text": text,
            "truncated": truncated,
        }

    # -------------------------------------------------------- 提示词库
    @app.get("/api/prompts")
    def list_prompts() -> dict[str, Any]:
        return {"prompts": db.list_prompts()}

    @app.post("/api/prompts")
    def save_prompt(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        name = str(body.get("name", "")).strip()
        content = str(body.get("content", "")).strip()
        if not name or not content:
            raise HTTPException(400, "名称和内容都不能为空")
        return db.save_prompt(body.get("id"), name, content)

    @app.delete("/api/prompts/{prompt_id}")
    def delete_prompt(prompt_id: str) -> dict[str, Any]:
        db.delete_prompt(prompt_id)
        return {"ok": True}

    # ---------------------------------------------------------- 设置
    def public_settings(settings: dict[str, Any]) -> dict[str, Any]:
        """API Key 只写不读，页面只需要知道有没有设置过。"""
        search = dict(settings["search"])
        for name in ("brave", "tavily"):
            search[f"{name}_key_set"] = bool(search.pop(f"{name}_api_key", ""))
        return {**settings, "search": search}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return public_settings(db.get_settings())

    @app.patch("/api/settings")
    def patch_settings(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        if "params" in body:
            patch["params"] = normalize_params(body["params"])
        if "default_model" in body:
            patch["default_model"] = body["default_model"] or None
        if "hf_endpoint" in body:
            endpoint = str(body["hf_endpoint"]).strip().rstrip("/")
            if not re.match(r"^https?://[^\s/]+", endpoint):
                raise HTTPException(400, "下载源地址不合法")
            patch["hf_endpoint"] = endpoint
        if "download_dir" in body:
            patch["download_dir"] = str(body["download_dir"]).strip()
        if "model_dirs" in body:
            patch["model_dirs"] = [str(p).strip() for p in body["model_dirs"] if str(p).strip()]
        for key in ("auto_title", "trust_remote_code"):
            if key in body:
                patch[key] = bool(body[key])
        for key in ("kv_cache_limit_mb", "idle_unload_minutes"):
            if key in body:
                try:
                    patch[key] = max(int(body[key]), 0)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(400, f"{key} 必须是数字") from exc
        if isinstance(body.get("ui"), dict):
            patch["ui"] = body["ui"]
        if isinstance(body.get("memory"), dict):
            incoming = body["memory"]
            clean: dict[str, Any] = {k: bool(incoming[k]) for k in ("enabled", "review") if k in incoming}
            if "idle_seconds" in incoming:
                clean["idle_seconds"] = max(min(int(incoming["idle_seconds"]), 86400), 10)
            if "budget_tokens" in incoming:
                clean["budget_tokens"] = max(min(int(incoming["budget_tokens"]), 4000), 50)
            patch["memory"] = clean
        if isinstance(body.get("privacy"), dict):
            incoming = body["privacy"]
            clean = {}
            if "usage_stats" in incoming:
                clean["usage_stats"] = bool(incoming["usage_stats"])
            if "auto_lock_minutes" in incoming:
                clean["auto_lock_minutes"] = max(min(int(incoming["auto_lock_minutes"]), 1440), 0)
            patch["privacy"] = clean
        if isinstance(body.get("search"), dict):
            incoming = body["search"]
            clean = {}
            if "provider" in incoming:
                if incoming["provider"] not in websearch.PROVIDERS:
                    raise HTTPException(400, "不支持的搜索服务")
                clean["provider"] = incoming["provider"]
            if "searxng_url" in incoming:
                url = str(incoming["searxng_url"]).strip().rstrip("/")
                if url and not re.match(r"^https?://[^\s/]+", url):
                    raise HTTPException(400, "SearXNG 地址不合法")
                clean["searxng_url"] = url
            if "max_results" in incoming:
                clean["max_results"] = max(min(int(incoming["max_results"]), 10), 1)
            if "fetch_pages" in incoming:
                clean["fetch_pages"] = max(min(int(incoming["fetch_pages"]), 5), 0)
            for name in ("brave_api_key", "tavily_api_key"):
                if name in incoming:
                    clean[name] = str(incoming[name] or "").strip()
            patch["search"] = clean
        result = db.update_settings(patch)
        if {"download_dir", "model_dirs"} & set(patch):
            registry.invalidate()
        return public_settings(result)

    @app.get("/api/usage")
    def usage() -> dict[str, Any]:
        return db.usage_summary()

    @app.post("/api/usage/clear")
    def clear_usage() -> dict[str, Any]:
        db.clear_usage()
        return {"ok": True}

    # ---------------------------------------------------------- 导入导出
    def _attachment(content: str, filename: str, media_type: str) -> Response:
        quoted = urllib.parse.quote(filename)
        return Response(
            content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted}"},
        )

    @app.get("/api/conversations/{conv_id}/export")
    def export_conversation(conv_id: str, format: str = "md") -> Response:
        conversation = db.get_conversation(conv_id)
        if conversation is None:
            raise HTTPException(404, "对话不存在")
        name = re.sub(r'[\\/:*?"<>|\n]+', "_", conversation["title"])[:60] or "conversation"
        if format == "json":
            return _attachment(service.export_json([conv_id]), f"{name}.json", "application/json")
        return _attachment(service.export_markdown(conv_id), f"{name}.md", "text/markdown; charset=utf-8")

    @app.get("/api/export")
    def export_all() -> Response:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return _attachment(service.export_json(), f"mlxui-backup-{stamp}.json", "application/json")

    @app.post("/api/import")
    async def import_data(request: Request) -> dict[str, Any]:
        try:
            payload = json.loads(await read_body(request, _MAX_IMPORT_BYTES))
        except ValueError as exc:
            raise HTTPException(400, "文件不是合法的 JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "文件格式不正确，只支持本应用导出的 JSON")
        return {"count": await asyncio.to_thread(service.import_json, payload)}

    # ------------------------------------------------- OpenAI 兼容接口
    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": m["id"], "object": "model", "created": int(m["modified"]), "owned_by": "local"}
                for m in registry.scan()
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            return _openai_error("请求体不是合法的 JSON", 400)
        model_id = body.get("model")
        if not registry.get(model_id):
            model_id = engine.model_id or db.get_settings().get("default_model")
        if not model_id or not registry.get(model_id):
            return _openai_error(f"model not found: {body.get('model')}", 404)

        params = dict(db.get_settings()["params"])
        params.update(
            normalize_params(
                {
                    key: body[key]
                    for key in ("temperature", "top_p", "top_k", "min_p", "seed", "stop", "repetition_penalty", "context_length")
                    if key in body
                }
            )
        )
        limit = body.get("max_completion_tokens", body.get("max_tokens"))
        if limit is not None:
            params.update(normalize_params({"max_tokens": limit}))
        system_parts: list[str] = []
        turns: list[Turn] = []
        for index, message in enumerate(body.get("messages") or []):
            role = message.get("role")
            text = _text_of(message.get("content"))
            if role in ("system", "developer"):
                system_parts.append(text)
            elif role in ("user", "assistant"):
                turns.append(Turn(f"m{index}", role, text))
        if not turns:
            return _openai_error("messages 中至少需要一条 user 消息", 400)

        def build(tokenizer: Any, counter: TokenCounter) -> ContextResult:
            return build_context(
                tokenizer,
                counter,
                system="\n\n".join(system_parts),
                summary=None,
                turns=turns,
                limit=int(params["context_length"]),
                max_tokens=int(params["max_tokens"]),
                thinking=bool(params["thinking"]),
            )

        job = Job(
            kind="api",
            model_id=model_id,
            params=params,
            build_context=build,
            sink=UsageSink(db),
            cache_key="__api__",
        )
        engine.enqueue(job)
        completion_id = f"chatcmpl-{job.id}"
        created = int(time.time())

        if body.get("stream"):
            async def stream():
                cursor = 0
                wake = job.subscribe()

                def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                try:
                    yield chunk({"role": "assistant"})
                    while True:
                        for event in job.slice(cursor):
                            cursor = event["seq"] + 1
                            if event["type"] == "delta":
                                yield chunk({"content": event["text"]})
                            elif event["type"] == "done":
                                reason = event["stats"].get("finish_reason")
                                yield chunk({}, "length" if reason == "length" else "stop")
                            elif event["type"] == "error":
                                error = {"error": {"message": event["message"]}}
                                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                        if job.finished and not job.slice(cursor):
                            yield "data: [DONE]\n\n"
                            return
                        wake.clear()
                        if job.slice(cursor):
                            continue
                        try:
                            await asyncio.wait_for(wake.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                        if await request.is_disconnected():
                            engine.stop(job)
                            return
                finally:
                    job.unsubscribe(wake)

            return StreamingResponse(stream(), media_type="text/event-stream")

        await asyncio.to_thread(job.done_event.wait)
        if job.error:
            return _openai_error(job.error, 500)
        stats = job.result_stats
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": job.result_text},
                    "finish_reason": "length" if stats.get("finish_reason") == "length" else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("completion_tokens", 0),
                "total_tokens": stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0),
            },
        }

    # ---------------------------------------------------------- 页面
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", NoCacheStatic(directory=STATIC_DIR), name="static")
    return app


def _openai_error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)
