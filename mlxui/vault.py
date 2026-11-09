"""数据库密钥管理。

两种模式：
- keychain：随机 256 位密钥存在 macOS 钥匙串里，启动时自动解锁，防的是磁盘文件被拿走；
- password：密钥由启动密码经 scrypt 派生，磁盘上不保存任何密钥，忘记密码数据无法恢复。
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Protocol

SERVICE = "mlxui-database-key"
ACCOUNT = "mlxui"
SECURITY = "/usr/bin/security"
MIN_PASSWORD_LENGTH = 8
_SCRYPT = {"n": 2**17, "r": 8, "p": 1}
_NOT_FOUND = 44  # security 命令约定的“条目不存在”退出码


class KeyStoreUnavailable(Exception):
    pass


class KeyStore(Protocol):
    def get(self) -> str | None: ...
    def put(self, key: str) -> None: ...
    def delete(self) -> None: ...


class KeychainStore:
    """把密钥放进 macOS 钥匙串。

    写入走 `security -i` 的标准输入，不用命令行参数，避免密钥出现在进程列表里。
    """

    def __init__(self, keychain: str | None = None, service: str = SERVICE):
        self.keychain = keychain
        self.service = service

    def _run(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise KeyStoreUnavailable("无法访问钥匙串") from exc

    def get(self) -> str | None:
        args = [SECURITY, "find-generic-password", "-a", ACCOUNT, "-s", self.service, "-w"]
        if self.keychain:
            args.append(self.keychain)
        result = self._run(args)
        if result.returncode == 0:
            return result.stdout.strip() or None
        if result.returncode == _NOT_FOUND:
            return None
        raise KeyStoreUnavailable("无法读取钥匙串")

    def put(self, key: str) -> None:
        target = f' "{self.keychain}"' if self.keychain else ""
        command = (
            f"add-generic-password -a {ACCOUNT} -s {self.service} "
            f"-l \"MLX UI 数据库密钥\" -T {SECURITY} -w {key}{target}\n"
        )
        self.delete()  # 覆盖已有条目（-U）会触发系统授权弹窗，先删后写不会
        self._run([SECURITY, "-i"], stdin=command)
        if self.get() != key:  # security -i 出错时退出码仍可能是 0，以读回为准
            raise KeyStoreUnavailable("无法写入钥匙串")

    def delete(self) -> None:
        args = [SECURITY, "delete-generic-password", "-a", ACCOUNT, "-s", self.service]
        if self.keychain:
            args.append(self.keychain)
        self._run(args)


def filevault_enabled() -> bool | None:
    try:
        output = subprocess.run(
            ["/usr/bin/fdesetup", "status"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    if "FileVault is On" in output:
        return True
    if "FileVault is Off" in output:
        return False
    return None


class Vault:
    def __init__(self, directory: Path, store: KeyStore | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.directory / "vault.json"
        self.store: KeyStore = store or KeychainStore()
        self.key: str | None = None
        self.last_activity = time.time()
        self._lock = threading.RLock()

    # ---------------------------------------------------------- 状态
    def _meta(self) -> dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_meta(self, meta: dict[str, Any]) -> None:
        temp = self.meta_path.with_suffix(".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.meta_path)

    @property
    def mode(self) -> str | None:
        return self._meta().get("mode")

    @property
    def unlocked(self) -> bool:
        return self.key is not None

    @property
    def state(self) -> str:
        if self.key:
            return "unlocked"
        mode = self.mode
        if mode == "password":
            return "locked"
        if mode == "keychain":
            return "key_missing"  # 钥匙串里的密钥没了，或者数据目录来自别的机器
        return "setup"

    def touch(self) -> None:
        self.last_activity = time.time()

    def forget(self) -> None:
        with self._lock:
            self.key = None

    # ---------------------------------------------------------- 钥匙串模式
    def auto_key(self) -> str | None:
        """钥匙串模式下直接取得密钥；首次运行时生成。需要密码时返回 None。"""
        with self._lock:
            mode = self.mode
            if mode == "password":
                return None
            existing = self.store.get()
            if mode == "keychain":
                self.key = existing
                return existing
            key = existing or secrets.token_hex(32)
            if not existing:
                self.store.put(key)
            self._save_meta({"version": 1, "mode": "keychain"})
            self.key = key
            return key

    def new_random_key(self) -> str:
        return secrets.token_hex(32)

    def commit_keychain(self, key: str) -> None:
        with self._lock:
            self.store.put(key)
            self._save_meta({"version": 1, "mode": "keychain"})
            self.key = key

    # ------------------------------------------------------------ 密码模式
    @staticmethod
    def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> str:
        return hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=512 * 1024 * 1024, dklen=32
        ).hex()

    def derive(self, password: str) -> str:
        meta = self._meta()
        return self._derive(password, bytes.fromhex(meta["salt"]), meta["n"], meta["r"], meta["p"])

    def prepare_password(self, password: str) -> tuple[str, dict[str, Any]]:
        """生成新盐并派生密钥，此时还没有生效。"""
        salt = secrets.token_bytes(16)
        meta = {"version": 1, "mode": "password", "salt": salt.hex(), **_SCRYPT}
        return self._derive(password, salt, **_SCRYPT), meta

    def commit_password(self, key: str, meta: dict[str, Any]) -> None:
        with self._lock:
            self._save_meta(meta)
            self.key = key
            try:
                self.store.delete()  # 切到密码模式后，钥匙串里不再留密钥
            except KeyStoreUnavailable:
                pass

    # -------------------------------------------------------------- 抹除
    def destroy(self) -> None:
        with self._lock:
            self.key = None
            try:
                self.store.delete()
            except KeyStoreUnavailable:
                pass
            self.meta_path.unlink(missing_ok=True)
