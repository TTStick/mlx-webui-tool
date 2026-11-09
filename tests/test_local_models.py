"""手动添加本地模型文件夹。"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import wait_job
from fastapi.testclient import TestClient

ADD = "/api/models/local"
REMOVE = "/api/models/local/remove"
PICK = "/api/system/pick-folder"


@pytest.fixture()
def library(tmp_path) -> Path:
    root = tmp_path / "library"
    root.mkdir()
    return root


@pytest.fixture()
def make_model(tiny_model_dir, library):
    def make(relative: str = "my-model") -> Path:
        target = library / relative
        shutil.copytree(tiny_model_dir / "tiny-llama", target)
        return target

    return make


def add(client, path) -> dict:
    res = client.post(ADD, json={"path": str(path)})
    assert res.status_code == 200, res.text
    return res.json()


def model_dirs(client) -> list[str]:
    return client.get("/api/settings").json()["model_dirs"]


def listed(client) -> dict[str, dict]:
    return {m["id"]: m for m in client.get("/api/models?refresh=true").json()["models"]}


def test_add_single_folder_and_chat(app_ctx, make_model):
    client, app, _ = app_ctx
    folder = make_model()
    result = add(client, folder)
    assert result["added"] == 1 and result["problems"] == []
    assert result["models"][0]["name"] == "my-model"
    assert "fit" in result["models"][0]
    assert str(folder) in model_dirs(client)

    entry = listed(client)[str(folder)]
    assert entry["source"] == "local" and entry["deletable"] is False

    conv = client.post(
        "/api/conversations", json={"model": str(folder), "settings": {"max_tokens": 5, "temperature": 0}}
    ).json()["id"]
    res = client.post(f"/api/conversations/{conv}/send", json={"content": "hi"}).json()
    assert wait_job(app, res["job_id"]).state == "done"


def test_add_parent_finds_every_model(app_ctx, make_model, library):
    client, _, _ = app_ctx
    make_model("a")
    make_model("org/b")
    make_model("x/y/c")
    result = add(client, library)
    assert result["added"] == 3
    assert sorted(m["name"] for m in result["models"]) == ["a", "b", "c"]
    assert model_dirs(client).count(str(library)) == 1


def test_search_stops_at_depth_limit(app_ctx, make_model, library):
    client, _, _ = app_ctx
    make_model("1/2/3/too-deep")
    result = add(client, library)
    assert result["models"] == []
    assert "3 层" in result["problems"][0]["message"]


def test_hidden_folders_are_skipped(app_ctx, make_model, library):
    client, _, _ = app_ctx
    make_model(".cache/secret")
    assert add(client, library)["models"] == []


def test_adding_twice_is_idempotent(app_ctx, make_model):
    client, _, _ = app_ctx
    folder = make_model()
    add(client, folder)
    before = model_dirs(client)
    again = add(client, folder)
    assert again["added"] == 0 and len(again["models"]) == 1
    assert model_dirs(client) == before


def test_parent_after_child_only_counts_new_models(app_ctx, make_model, library):
    client, _, _ = app_ctx
    first = make_model("first")
    make_model("second")
    add(client, first)
    result = add(client, library)
    assert result["added"] == 1 and len(result["models"]) == 2
    assert len([m for m in listed(client) if m.startswith(str(library))]) == 2


@pytest.mark.parametrize(
    "transform",
    [
        lambda p: f'"{p}"',
        lambda p: f"'{p}'",
        lambda p: f"  {p}/  ",
        lambda p: f"file://{p}",
    ],
)
def test_common_paste_forms_are_accepted(app_ctx, make_model, transform):
    client, _, _ = app_ctx
    folder = make_model()
    assert add(client, transform(folder))["added"] == 1


def test_escaped_spaces_from_terminal_drag(app_ctx, make_model):
    client, _, _ = app_ctx
    folder = make_model("My Models/q")
    assert add(client, str(folder).replace(" ", "\\ "))["added"] == 1


def test_percent_encoded_file_url(app_ctx, make_model):
    client, _, _ = app_ctx
    folder = make_model("My Models/q")
    assert add(client, "file://" + str(folder).replace(" ", "%20"))["added"] == 1


def test_tilde_expands_to_home(app_ctx, make_model, library, monkeypatch):
    client, _, _ = app_ctx
    make_model("m")
    monkeypatch.setenv("HOME", str(library))
    assert add(client, "~/m")["added"] == 1


def test_selecting_a_file_uses_its_folder(app_ctx, make_model):
    client, _, _ = app_ctx
    folder = make_model()
    assert add(client, folder / "config.json")["path"] == str(folder)


def test_empty_path_rejected(app_ctx):
    client, _, _ = app_ctx
    assert client.post(ADD, json={"path": "  "}).status_code == 400


@pytest.mark.parametrize(
    ("path", "keyword"),
    [("models/relative", "完整路径"), ("/definitely/not/here", "不存在")],
)
def test_bad_paths_explained(app_ctx, path, keyword):
    client, _, _ = app_ctx
    result = add(client, path)
    assert result["models"] == [] and keyword in result["problems"][0]["message"]


def _gguf(folder: Path) -> None:
    for item in folder.iterdir():
        item.unlink()
    (folder / "model-q4.gguf").write_bytes(b"x")


def _pytorch(folder: Path) -> None:
    (folder / "model.safetensors").unlink()
    (folder / "pytorch_model.bin").write_bytes(b"x")


def _no_config(folder: Path) -> None:
    (folder / "config.json").unlink()


def _no_tokenizer(folder: Path) -> None:
    (folder / "tokenizer.json").unlink()


def _no_weights(folder: Path) -> None:
    (folder / "model.safetensors").unlink()


def _missing_shard(folder: Path) -> None:
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors", "b": "model.safetensors"}})
    )


def _unsupported(folder: Path) -> None:
    (folder / "config.json").write_text(json.dumps({"model_type": "bert"}))


@pytest.mark.parametrize(
    ("damage", "keyword"),
    [
        (_gguf, "GGUF"),
        (_pytorch, ".bin"),
        (_no_config, "config.json"),
        (_no_tokenizer, "分词器"),
        (_no_weights, "safetensors"),
        (_missing_shard, "分片"),
        (_unsupported, "bert"),
    ],
)
def test_unusable_folder_is_explained_and_not_added(app_ctx, make_model, damage, keyword):
    client, _, _ = app_ctx
    folder = make_model()
    damage(folder)
    before = model_dirs(client)
    result = add(client, folder)
    assert result["models"] == []
    assert keyword in result["problems"][0]["message"]
    assert model_dirs(client) == before


def test_empty_folder_explained(app_ctx, library):
    client, _, _ = app_ctx
    assert "没有找到" in add(client, library)["problems"][0]["message"]


def test_unrelated_config_json_is_not_reported(app_ctx, library):
    """普通项目里的 config.json 不能被当成坏掉的模型。"""
    client, _, _ = app_ctx
    (library / "some-app").mkdir()
    (library / "some-app" / "config.json").write_text('{"theme": "dark"}')
    problems = add(client, library)["problems"]
    assert len(problems) == 1 and problems[0]["path"] == str(library)


def test_good_and_bad_models_reported_together(app_ctx, make_model, library):
    client, _, _ = app_ctx
    make_model("good")
    _gguf(make_model("bad"))
    result = add(client, library)
    assert [m["name"] for m in result["models"]] == ["good"]
    assert len(result["problems"]) == 1 and result["problems"][0]["path"].endswith("bad")


def test_unreadable_folder_explained(app_ctx, library, monkeypatch):
    client, _, _ = app_ctx
    real_listdir = __import__("os").listdir

    def deny(path):
        if str(path) == str(library):
            raise PermissionError(1, "denied")
        return real_listdir(path)

    monkeypatch.setattr("mlxui.models.os.listdir", deny)
    assert "权限" in add(client, library)["problems"][0]["message"]


def test_remove_only_forgets_the_folder(app_ctx, make_model):
    client, _, _ = app_ctx
    folder = make_model()
    add(client, folder)
    res = client.post(REMOVE, json={"path": str(folder)})
    assert res.status_code == 200
    assert str(folder) not in listed(client)
    assert (folder / "model.safetensors").exists()


def test_removing_default_model_clears_default(app_ctx, make_model):
    client, _, _ = app_ctx
    folder = make_model()
    add(client, folder)
    client.patch("/api/settings", json={"default_model": str(folder)})
    assert client.post(REMOVE, json={"path": str(folder)}).json()["default_model"] is None


def test_local_dirs_report_count_and_missing_drive(app_ctx, make_model, library):
    client, _, _ = app_ctx
    make_model("a")
    make_model("b")
    add(client, library)
    rows = client.get("/api/models?refresh=true").json()["local_dirs"]
    row = next(r for r in rows if r["path"] == str(library))
    assert row == {"path": str(library), "count": 2, "exists": True, "removable": True}

    shutil.rmtree(library)
    rows = client.get("/api/models?refresh=true").json()["local_dirs"]
    row = next(r for r in rows if r["path"] == str(library))
    assert row["exists"] is False and row["count"] == 0


def _hub_layout(cache: Path, source: Path) -> Path:
    sha = "b" * 40
    repo = cache / "models--acme--cool-model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(sha)
    snapshot = repo / "snapshots" / sha
    shutil.copytree(source, snapshot)
    return snapshot


def test_hub_snapshot_is_not_registered_twice(app_ctx, tiny_model_dir, tmp_path):
    client, _, _ = app_ctx
    cache = tmp_path / "hub"
    snapshot = _hub_layout(cache, tiny_model_dir / "tiny-llama")
    client.patch("/api/settings", json={"download_dir": str(cache)})
    result = add(client, snapshot)
    assert result["added"] == 0
    assert result["models"][0]["id"] == "acme/cool-model"
    assert str(snapshot) not in model_dirs(client)


def test_copied_hub_layout_uses_repo_name(app_ctx, tiny_model_dir, library):
    client, _, _ = app_ctx
    _hub_layout(library, tiny_model_dir / "tiny-llama")
    assert add(client, library)["models"][0]["name"] == "cool-model"


def test_symlinked_model_folder_is_found(app_ctx, make_model, library, tmp_path):
    client, _, _ = app_ctx
    real = make_model("store/real")
    links = tmp_path / "links"
    links.mkdir()
    (links / "linked").symlink_to(real, target_is_directory=True)
    assert add(client, links)["models"][0]["name"] == "linked"


# ------------------------------------------------------------ 系统选择窗口
@pytest.fixture()
def local_client(app_ctx):
    _, app, _ = app_ctx
    return TestClient(app, client=("127.0.0.1", 50000))


def fake_osascript(monkeypatch, returncode=0, stdout="", stderr=""):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr("mlxui.models.subprocess.run", run)
    return calls


def test_picker_returns_selected_folder(local_client, monkeypatch):
    calls = fake_osascript(monkeypatch, stdout="/Users/me/models/q/\n")
    assert local_client.post(PICK).json() == {"path": "/Users/me/models/q/"}
    assert calls[0][0] == "/usr/bin/osascript"


def test_picker_cancel_returns_null(local_client, monkeypatch):
    fake_osascript(monkeypatch, returncode=1, stderr="execution error: User canceled. (-128)")
    assert local_client.post(PICK).json() == {"path": None}


def test_picker_failure_suggests_pasting(local_client, monkeypatch):
    fake_osascript(monkeypatch, returncode=1, stderr="execution error: no UI (-1713)")
    res = local_client.post(PICK)
    assert res.status_code == 503 and "粘贴" in res.json()["detail"]


def test_picker_timeout_suggests_pasting(local_client, monkeypatch):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 600)

    monkeypatch.setattr("mlxui.models.subprocess.run", run)
    assert local_client.post(PICK).status_code == 503


def test_second_picker_is_refused_while_one_is_open(local_client):
    from mlxui import models

    assert models._picker_lock.acquire(blocking=False)
    try:
        res = local_client.post(PICK)
        assert res.status_code == 503 and "已经打开" in res.json()["detail"]
    finally:
        models._picker_lock.release()


def test_picker_refused_for_non_local_clients(app_ctx, monkeypatch):
    client, _, _ = app_ctx  # TestClient 默认的来源地址不是本机回环
    calls = fake_osascript(monkeypatch, stdout="/x/")
    assert client.post(PICK).status_code == 403
    assert calls == []
