"""加密存储、无日志、删除即无痕。"""
from __future__ import annotations

import logging
import sqlite3 as plain_sqlite
import subprocess
from pathlib import Path

import pytest
from conftest import MemoryKeyStore, wait_job
from fastapi.testclient import TestClient

from mlxui.app import create_app
from mlxui.db import DB, WrongKey

MARKER = "ZZUNIQUEMARKER7741"
KEY = "ab" * 32
OTHER_KEY = "cd" * 32


def new_conv(client, model, **settings):
    base = {"max_tokens": 6, "temperature": 0.0, "context_length": 2048}
    return client.post("/api/conversations", json={"model": model, "settings": {**base, **settings}}).json()["id"]


def send(client, app, conv, text):
    res = client.post(f"/api/conversations/{conv}/send", json={"content": text}).json()
    return res, wait_job(app, res["job_id"])


def flush(app):
    app.state.engine.submit(lambda: None).result(timeout=30)


def assert_absent(root: Path, needle: str) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            assert needle.encode() not in path.read_bytes(), f"found plaintext in {path.name}"


# ------------------------------------------------------------ 加密存储
def test_database_file_has_no_plaintext(app_ctx, tmp_path):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    uploaded = client.post("/api/attachments", params={"name": "a.txt"}, content=MARKER.encode()).json()
    res = client.post(
        f"/api/conversations/{conv}/send", json={"content": f"hello {MARKER}", "attachments": [uploaded]}
    ).json()
    wait_job(app, res["job_id"])
    client.patch(f"/api/conversations/{conv}", json={"title": MARKER})
    assert any((tmp_path / name).exists() for name in ("test.db", "test.db-wal"))
    assert_absent(tmp_path, MARKER)


def test_plain_sqlite_cannot_read_the_file(app_ctx, tmp_path):
    conn = plain_sqlite.connect(tmp_path / "test.db")
    with pytest.raises(plain_sqlite.DatabaseError):
        conn.execute("SELECT * FROM conversations").fetchall()
    conn.close()


def test_database_files_are_private(app_ctx, tmp_path):
    for name in ("test.db", "test.db-wal", "test.db-shm", "vault.json"):
        assert (tmp_path / name).stat().st_mode & 0o077 == 0, name


def test_secure_delete_is_enabled(app_ctx):
    _, app, _ = app_ctx
    assert app.state.db.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    assert app.state.db.conn.execute("PRAGMA temp_store").fetchone()[0] == 2


def test_wrong_key_rejected(tmp_path):
    path = tmp_path / "x.db"
    db = DB(path)
    db.open(KEY)
    db.close()
    other = DB(path)
    with pytest.raises(WrongKey):
        other.open(OTHER_KEY)


def _plaintext_db(path: Path) -> None:
    conn = plain_sqlite.connect(path)
    conn.executescript(
        """CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '新对话',
             title_source TEXT NOT NULL DEFAULT 'default', created_at REAL NOT NULL, updated_at REAL NOT NULL,
             pinned INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0, archived_at REAL, model TEXT,
             settings TEXT NOT NULL DEFAULT '{}', leaf_id TEXT, summary TEXT, summary_upto TEXT);
           CREATE TABLE usage (day TEXT NOT NULL, model TEXT NOT NULL, requests INTEGER NOT NULL DEFAULT 0,
             prompt_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
             completion_tokens INTEGER NOT NULL DEFAULT 0, seconds REAL NOT NULL DEFAULT 0,
             PRIMARY KEY (day, model));"""
    )
    conn.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES('c1',?,1,1)", (MARKER,))
    conn.execute("INSERT INTO usage VALUES('2026-01-01','m',1,1,1,1,1)")
    conn.commit()
    conn.close()


def test_old_plaintext_database_is_encrypted_and_kept(tmp_path):
    path = tmp_path / "old.db"
    _plaintext_db(path)
    db = DB(path)
    db.open(KEY)
    assert db.get_conversation("c1")["title"] == MARKER
    assert path.read_bytes()[:16] != b"SQLite format 3\x00"
    db.close()
    assert_absent(tmp_path, MARKER)


def test_upgrade_clears_old_usage_log(tmp_path):
    path = tmp_path / "old.db"
    _plaintext_db(path)
    db = DB(path)
    db.open(KEY)
    assert db.conn.execute("SELECT COUNT(*) n FROM usage").fetchone()["n"] == 0


def test_interrupted_migration_recovers(tmp_path):
    path = tmp_path / "x.db"
    db = DB(path)
    db.open(KEY)
    db.create_conversation(title="kept")
    db.close()
    # 崩溃点：明文已改名待清除，新库还没换回原位
    path.rename(tmp_path / "x.db.new")
    (tmp_path / "x.db.wipe").write_bytes(MARKER.encode())
    recovered = DB(path)
    recovered.open(KEY)
    assert recovered.list_conversations()[0]["title"] == "kept"
    recovered.close()
    assert not (tmp_path / "x.db.wipe").exists()
    assert_absent(tmp_path, MARKER)


def test_leftover_partial_export_is_removed(tmp_path):
    path = tmp_path / "x.db"
    db = DB(path)
    db.open(KEY)
    db.create_conversation(title="kept")
    db.close()
    (tmp_path / "x.db.new").write_bytes(b"partial")
    recovered = DB(path)
    recovered.open(KEY)
    assert recovered.list_conversations()[0]["title"] == "kept"
    assert not (tmp_path / "x.db.new").exists()


def test_rekey_replaces_key_and_keeps_data(tmp_path):
    path = tmp_path / "x.db"
    db = DB(path)
    db.open(KEY)
    db.create_conversation(title="kept")
    db.rekey(OTHER_KEY)
    assert db.list_conversations()[0]["title"] == "kept"  # 重新加密后连接立即可用
    db.close()
    with pytest.raises(WrongKey):
        DB(path).open(KEY)
    again = DB(path)
    again.open(OTHER_KEY)
    assert again.list_conversations()[0]["title"] == "kept"


# ------------------------------------------------------------ 删除即无痕
def test_delete_truncates_wal_and_removes_row(app_ctx, tmp_path):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv, f"secret {MARKER}")
    client.delete(f"/api/conversations/{conv}")
    wal = tmp_path / "test.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0
    assert client.get("/api/search", params={"q": MARKER}).json()["results"] == []
    count = app.state.db.conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"]
    assert count == 0


def test_delete_purges_memory_caches(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    _, job = send(client, app, conv, f"secret {MARKER}")
    engine = app.state.engine
    assert conv in engine.caches
    assert job.id in engine.jobs
    client.post("/api/context/preview", json={"conversation_id": conv, "draft": MARKER})

    client.delete(f"/api/conversations/{conv}")
    flush(app)

    assert conv not in engine.caches
    assert job.id not in engine.jobs
    memos = [engine._counter._memo] + [pair[1]._memo for pair in engine._tokenizers.values()]
    assert all(not any(MARKER in text for text in memo) for memo in memos)


def test_delete_during_generation_leaves_nothing(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model, max_tokens=3000)
    res = client.post(f"/api/conversations/{conv}/send", json={"content": f"x {MARKER}"}).json()
    client.delete(f"/api/conversations/{conv}")
    wait_job(app, res["job_id"]) if res["job_id"] in app.state.engine.jobs else None
    flush(app)
    assert conv not in app.state.engine.caches
    assert app.state.db.conn.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] == 0


def test_deleting_message_drops_its_kv_cache(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    data, _ = send(client, app, conv, "first")
    client.post(f"/api/conversations/{conv}/delete-message", json={"message_id": data["messages"][0]["id"]})
    flush(app)
    assert conv not in app.state.engine.caches


def test_bulk_delete_goes_through_full_purge(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv, "x")
    assert client.post("/api/conversations/bulk", json={"ids": [conv], "action": "delete"}).json()["count"] == 1
    flush(app)
    assert conv not in app.state.engine.caches


# ------------------------------------------------------------------ 日志
def test_nothing_is_logged_about_content(app_ctx, capfd, caplog):
    client, app, model = app_ctx
    caplog.set_level(logging.DEBUG)
    conv = new_conv(client, model)
    send(client, app, conv, f"hello {MARKER}")
    client.post("/api/context/preview", json={"conversation_id": conv, "draft": MARKER})
    captured = capfd.readouterr()
    assert MARKER not in captured.out + captured.err + caplog.text


def test_generation_errors_log_only_the_type(app_ctx, monkeypatch, capfd, caplog):
    client, app, model = app_ctx
    caplog.set_level(logging.DEBUG)

    def boom(*args, **kwargs):
        raise RuntimeError(f"failed while handling {MARKER}")
        yield

    monkeypatch.setattr("mlxui.engine.stream_generate", boom)
    conv = new_conv(client, model)
    _, job = send(client, app, conv, "hi")
    assert job.state == "error"
    captured = capfd.readouterr()
    assert MARKER not in captured.out + captured.err + caplog.text


def test_usage_stats_off_by_default(tmp_path):
    app = create_app(allow_any_host=True, db_path=tmp_path / "a.db", keystore=MemoryKeyStore())
    with TestClient(app) as client:
        assert client.get("/api/settings").json()["privacy"]["usage_stats"] is False
        app.state.db.add_usage("m", 1, 1, 1, 1.0)
        assert client.get("/api/usage").json()["by_model"] == []


def test_usage_can_be_cleared(app_ctx):
    client, app, _ = app_ctx
    app.state.db.add_usage("m", 1, 1, 1, 1.0)
    assert client.get("/api/usage").json()["by_model"]
    client.post("/api/usage/clear")
    assert client.get("/api/usage").json()["by_model"] == []


def test_api_responses_are_never_cached(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    assert client.get(f"/api/conversations/{conv}").headers["cache-control"] == "no-store"
    assert client.get("/api/settings").headers["cache-control"] == "no-store"


def test_large_upload_stays_in_memory(app_ctx, monkeypatch):
    client, _, _ = app_ctx
    import tempfile

    def forbidden(*args, **kwargs):
        raise AssertionError("upload was spooled to disk")

    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", forbidden)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
    res = client.post("/api/attachments", params={"name": "big.txt"}, content=b"a" * (3 * 1024 * 1024))
    assert res.status_code == 200


def test_oversized_upload_rejected(app_ctx):
    client, _, _ = app_ctx
    res = client.post("/api/attachments", params={"name": "big.txt"}, content=b"a" * (31 * 1024 * 1024))
    assert res.status_code == 413


# ------------------------------------------------------------ 密码与锁定
@pytest.fixture()
def vaulted(tmp_path):
    store = MemoryKeyStore()
    app = create_app(allow_any_host=True, db_path=tmp_path / "v.db", keystore=store)
    with TestClient(app) as client:
        yield client, app, store


def status(client):
    return client.get("/api/vault/status").json()


def test_first_run_uses_keychain_automatically(vaulted):
    client, _, store = vaulted
    assert status(client)["state"] == "unlocked" and status(client)["mode"] == "keychain"
    assert store.key is not None


def test_password_lifecycle(vaulted, tmp_path):
    client, app, store = vaulted
    client.post("/api/conversations", json={})
    assert client.post("/api/vault/password", json={"password": "short"}).status_code == 400

    assert client.post("/api/vault/password", json={"password": "correct horse"}).json()["mode"] == "password"
    assert store.key is None, "钥匙串里不能再留密钥"
    assert "correct horse" not in (tmp_path / "vault.json").read_text()

    assert client.post("/api/vault/lock").json()["state"] == "locked"
    assert client.get("/api/conversations").status_code == 423
    assert client.get("/api/monitor").status_code == 423

    assert client.post("/api/vault/unlock", json={"password": "wrong one!"}).status_code == 401
    assert status(client)["state"] == "locked"
    assert client.post("/api/vault/unlock", json={"password": "correct horse"}).json()["state"] == "unlocked"
    assert len(client.get("/api/conversations").json()["conversations"]) == 1

    assert client.post("/api/vault/password", json={"current": "nope", "password": "another pass"}).status_code == 401
    client.post("/api/vault/password", json={"current": "correct horse", "password": "another pass"})
    client.post("/api/vault/lock")
    assert client.post("/api/vault/unlock", json={"password": "correct horse"}).status_code == 401
    assert client.post("/api/vault/unlock", json={"password": "another pass"}).status_code == 200

    assert client.post("/api/vault/password/remove", json={"current": "bad"}).status_code == 401
    assert client.post("/api/vault/password/remove", json={"current": "another pass"}).json()["mode"] == "keychain"
    assert store.key is not None
    assert len(client.get("/api/conversations").json()["conversations"]) == 1


def test_password_survives_restart(tmp_path):
    store = MemoryKeyStore()
    path = tmp_path / "v.db"
    first = create_app(allow_any_host=True, db_path=path, keystore=store)
    with TestClient(first) as client:
        client.post("/api/conversations", json={})
        client.post("/api/vault/password", json={"password": "correct horse"})
    second = create_app(allow_any_host=True, db_path=path, keystore=store)
    with TestClient(second) as client:
        assert status(client)["state"] == "locked"
        client.post("/api/vault/unlock", json={"password": "correct horse"})
        assert len(client.get("/api/conversations").json()["conversations"]) == 1


def test_keychain_mode_reopens_automatically(tmp_path):
    store = MemoryKeyStore()
    path = tmp_path / "v.db"
    with TestClient(create_app(allow_any_host=True, db_path=path, keystore=store)) as client:
        client.post("/api/conversations", json={})
    with TestClient(create_app(allow_any_host=True, db_path=path, keystore=store)) as client:
        assert len(client.get("/api/conversations").json()["conversations"]) == 1


def test_missing_key_is_reported_not_overwritten(tmp_path):
    store = MemoryKeyStore()
    path = tmp_path / "v.db"
    with TestClient(create_app(allow_any_host=True, db_path=path, keystore=store)) as client:
        client.post("/api/conversations", json={})
    before = path.read_bytes()[:64]
    store.key = None
    with TestClient(create_app(allow_any_host=True, db_path=path, keystore=store)) as client:
        assert status(client)["state"] == "key_missing"
        assert client.get("/api/conversations").status_code == 423
    assert path.read_bytes()[:64] == before


def test_retry_recovers_when_keychain_comes_back(tmp_path):
    store = MemoryKeyStore()
    path = tmp_path / "v.db"
    with TestClient(create_app(allow_any_host=True, db_path=path, keystore=store)) as client:
        client.post("/api/conversations", json={})
    key, store.key = store.key, None
    with TestClient(create_app(allow_any_host=True, db_path=path, keystore=store)) as client:
        assert client.post("/api/vault/retry").json()["state"] == "key_missing"
        store.key = key
        assert client.post("/api/vault/retry").json()["state"] == "unlocked"
        assert len(client.get("/api/conversations").json()["conversations"]) == 1


def test_key_rotation_keeps_data_and_changes_key(vaulted, tmp_path):
    client, _, store = vaulted
    client.post("/api/conversations", json={})
    old = store.key
    assert client.post("/api/vault/rotate").status_code == 200
    assert store.key != old
    assert len(client.get("/api/conversations").json()["conversations"]) == 1


def test_lock_clears_conversation_state_from_memory(app_ctx):
    client, app, model = app_ctx
    conv = new_conv(client, model)
    send(client, app, conv, f"x {MARKER}")
    client.post("/api/vault/password", json={"password": "correct horse"})
    client.post("/api/vault/lock")
    assert app.state.engine.caches == {} or len(app.state.engine.caches) == 0
    assert app.state.vault.key is None
    assert not app.state.db.is_open


def test_auto_lock_after_idle(vaulted):
    client, app, _ = vaulted
    client.post("/api/vault/password", json={"password": "correct horse"})
    client.patch("/api/settings", json={"privacy": {"auto_lock_minutes": 1}})
    app.state.vault.last_activity = 0
    app.state.engine.submit(lambda: app.state.engine._on_idle()).result(timeout=30)
    assert status(client)["state"] == "locked"


def test_polling_does_not_count_as_activity(vaulted):
    client, app, _ = vaulted
    app.state.vault.last_activity = 123.0
    client.get("/api/monitor")
    client.get("/api/downloads")
    assert app.state.vault.last_activity == 123.0
    client.get("/api/settings")
    assert app.state.vault.last_activity > 123.0


def test_wipe_requires_confirmation_and_destroys_everything(vaulted, tmp_path):
    client, app, store = vaulted
    client.post("/api/conversations", json={})
    old = store.key
    assert client.post("/api/vault/wipe", json={}).status_code == 400
    assert client.post("/api/vault/wipe", json={"confirm": "抹除"}).json()["state"] == "unlocked"
    assert store.key != old
    assert client.get("/api/conversations").json()["conversations"] == []


def test_wipe_works_when_password_is_forgotten(vaulted):
    client, _, _ = vaulted
    client.post("/api/conversations", json={})
    client.post("/api/vault/password", json={"password": "correct horse"})
    client.post("/api/vault/lock")
    assert client.post("/api/vault/wipe", json={"confirm": "抹除"}).json()["state"] == "unlocked"
    assert client.get("/api/conversations").json()["conversations"] == []


# ---------------------------------------------------------- 真实钥匙串
@pytest.fixture()
def throwaway_keychain(tmp_path):
    """在临时钥匙串上测试真实的 security 命令，不碰用户的登录钥匙串。"""
    path = tmp_path / "t.keychain-db"
    try:
        subprocess.run(["/usr/bin/security", "create-keychain", "-p", "t", str(path)], check=True, timeout=20)
        subprocess.run(["/usr/bin/security", "unlock-keychain", "-p", "t", str(path)], check=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("security 命令不可用")
    yield str(path)
    subprocess.run(["/usr/bin/security", "delete-keychain", str(path)], timeout=20)


def test_real_keychain_roundtrip(throwaway_keychain):
    from mlxui.vault import KeychainStore

    store = KeychainStore(keychain=throwaway_keychain, service="mlxui-test")
    assert store.get() is None
    store.put(KEY)
    assert store.get() == KEY
    store.put(OTHER_KEY)
    assert store.get() == OTHER_KEY
    store.delete()
    assert store.get() is None
    store.delete()


def test_key_is_not_passed_on_the_command_line(monkeypatch):
    from mlxui import vault as vault_module

    calls = []

    def run(args, input=None, **kwargs):
        calls.append((args, input))

        class Result:
            returncode = 0
            stdout = KEY
            stderr = ""

        return Result()

    monkeypatch.setattr(vault_module.subprocess, "run", run)
    vault_module.KeychainStore().put(KEY)
    assert all(KEY not in " ".join(args) for args, _ in calls)
    assert any(stdin and KEY in stdin for _, stdin in calls)
