"""用本地桩服务器模拟 Hugging Face，走真实的 snapshot_download，不访问外网。"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SHA = "a" * 40
GOOD = "test-org/tiny-model"
SLOW = "test-org/slow-model"
BERT = "test-org/bert-model"
EMPTY = "test-org/no-weights"


def make_handler(files: dict[str, bytes]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _not_found(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _repo_files(self, repo):
            if repo == EMPTY:
                return {k: v for k, v in files.items() if not k.endswith(".safetensors")}
            if repo == BERT:
                return {**files, "config.json": json.dumps({"model_type": "bert"}).encode()}
            return files

        def _serve_api(self, path):
            parts = path.split("/")  # ['', 'api', 'models', org, name, ...]
            repo = f"{parts[3]}/{parts[4]}"
            content = self._repo_files(repo)
            if len(parts) > 5 and parts[5] == "tree":
                self._json(
                    [
                        {"type": "file", "path": k, "size": len(v), "oid": hashlib.sha1(v).hexdigest()}
                        for k, v in content.items()
                    ]
                )
                return
            config = json.loads(content["config.json"])
            self._json(
                {
                    "id": repo,
                    "sha": SHA,
                    "gated": False,
                    "private": False,
                    "downloads": 1,
                    "config": {"model_type": config["model_type"]},
                    "siblings": [{"rfilename": k, "size": len(v)} for k, v in content.items()],
                }
            )

        def _serve_file(self, path, head):
            parts = path.strip("/").split("/")  # org/name/resolve/rev/file
            repo = f"{parts[0]}/{parts[1]}"
            name = "/".join(parts[4:])
            content = self._repo_files(repo).get(name)
            if content is None:
                self._not_found()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("ETag", f'"{hashlib.sha1(content).hexdigest()}"')
            self.send_header("X-Repo-Commit", SHA)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            if head:
                return
            if repo == SLOW and name.endswith(".safetensors"):
                # 慢速传输，给“下载中”和“取消”留出观察窗口
                for start in range(0, len(content), 2048):
                    self.wfile.write(content[start : start + 2048])
                    self.wfile.flush()
                    time.sleep(0.25)
            else:
                self.wfile.write(content)

        def do_HEAD(self):
            path = self.path.split("?", 1)[0]
            if "/resolve/" in path:
                self._serve_file(path, head=True)
            else:
                self._not_found()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path.startswith("/api/models/"):
                    self._serve_api(path)
                elif "/resolve/" in path:
                    self._serve_file(path, head=False)
                else:
                    self._not_found()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


@pytest.fixture()
def hub(tiny_model_dir, app_ctx, tmp_path):
    client, app, _ = app_ctx
    source = tiny_model_dir / "tiny-llama"
    files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(files))
    server.handle_error = lambda *args: None  # 取消下载时客户端断开是预期行为
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cache = tmp_path / "hub-cache"
    client.patch(
        "/api/settings",
        json={"hf_endpoint": f"http://127.0.0.1:{server.server_port}", "download_dir": str(cache)},
    )
    yield client, app, cache
    for task in client.get("/api/downloads").json()["tasks"]:
        client.post(f"/api/downloads/{task['id']}/cancel")
    server.shutdown()


def wait_status(client, task_id, statuses, timeout=45):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for task in client.get("/api/downloads").json()["tasks"]:
            if task["id"] == task_id and task["status"] in statuses:
                return task
        time.sleep(0.1)
    raise AssertionError(f"task never reached {statuses}: {client.get('/api/downloads').json()}")


def installed_ids(client):
    return {m["id"] for m in client.get("/api/models?refresh=true").json()["models"]}


def test_download_then_chat_then_delete(hub):
    client, app, cache = hub
    started = client.post("/api/downloads", json={"repo": GOOD})
    assert started.status_code == 200, started.text
    task = wait_status(client, started.json()["id"], {"done", "failed"})
    assert task["status"] == "done", task
    assert task["done"] == task["total"] > 0

    models = client.get("/api/models?refresh=true").json()["models"]
    entry = next(m for m in models if m["id"] == GOOD)
    assert entry["source"] == "huggingface" and entry["deletable"] is True
    assert entry["weights_bytes"] > 0

    from conftest import wait_job

    conv = client.post(
        "/api/conversations", json={"model": GOOD, "settings": {"max_tokens": 5, "temperature": 0}}
    ).json()["id"]
    res = client.post(f"/api/conversations/{conv}/send", json={"content": "hi"}).json()
    assert wait_job(app, res["job_id"]).state == "done"

    deleted = client.post("/api/models/delete", json={"id": GOOD})
    assert deleted.status_code == 200, deleted.text
    assert GOOD not in installed_ids(client)
    assert not (cache / "models--test-org--tiny-model").exists()
    assert client.get("/api/monitor").json()["engine"]["model_id"] is None


def test_running_download_is_hidden_then_cancel_cleans_up(hub):
    client, _, cache = hub
    task = client.post("/api/downloads", json={"repo": SLOW}).json()
    time.sleep(1.5)
    assert SLOW not in installed_ids(client)

    running = next(t for t in client.get("/api/downloads").json()["tasks"] if t["id"] == task["id"])
    assert running["status"] == "running" and running["total"] > running["done"]

    assert client.post(f"/api/downloads/{task['id']}/cancel").status_code == 200
    assert client.get("/api/downloads").json()["tasks"] == []
    assert not (cache / "models--test-org--slow-model").exists()
    assert SLOW not in app_hidden(client)


def app_hidden(client):
    return client.app.state.registry.hidden


def test_duplicate_start_returns_same_task(hub):
    client, _, _ = hub
    first = client.post("/api/downloads", json={"repo": SLOW}).json()
    second = client.post("/api/downloads", json={"repo": SLOW}).json()
    assert first["id"] == second["id"]


def test_disk_full_is_rejected_before_downloading(hub, monkeypatch):
    client, _, cache = hub
    monkeypatch.setattr(
        "mlxui.downloads.disk_info", lambda _: {"total": 10**12, "used": 10**12, "free": 1024}
    )
    res = client.post("/api/downloads", json={"repo": GOOD})
    assert res.status_code == 400
    assert "磁盘空间不足" in res.json()["detail"]
    assert client.get("/api/downloads").json()["tasks"] == []


def test_unsupported_architecture_rejected(hub):
    client, _, _ = hub
    res = client.post("/api/downloads", json={"repo": BERT})
    assert res.status_code == 400 and "bert" in res.json()["detail"]


def test_repo_without_weights_rejected(hub):
    client, _, _ = hub
    res = client.post("/api/downloads", json={"repo": EMPTY})
    assert res.status_code == 400 and "safetensors" in res.json()["detail"]


def test_unreachable_endpoint_gives_actionable_error(app_ctx):
    client, _, _ = app_ctx
    client.patch("/api/settings", json={"hf_endpoint": "http://127.0.0.1:9"})
    res = client.post("/api/downloads", json={"repo": GOOD})
    assert res.status_code == 400 and "hf-mirror" in res.json()["detail"]


def test_partial_snapshot_is_not_listed(hub):
    """快照里只有 config 没有权重时，不能当成已安装的模型。"""
    client, _, cache = hub
    repo = cache / "models--test-org--broken"
    snapshot = repo / "snapshots" / SHA
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert "test-org/broken" not in installed_ids(client)
