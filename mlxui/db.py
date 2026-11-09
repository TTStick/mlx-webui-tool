"""加密存储（SQLCipher）：对话、消息树、记忆、提示词库、设置。

消息按 parent_id 组成一棵树：重新生成、编辑都会产生兄弟节点，
conversations.leaf_id 指向当前显示的分支末端。

删除语义：secure_delete 把被删内容所在的页清零，随后把 WAL 截断，
由被删对话产生的记忆也在同一事务里移除。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from sqlcipher3 import dbapi2 as sqlite3

from .config import DEFAULT_SETTINGS, deep_merge

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '新对话',
    title_source TEXT NOT NULL DEFAULT 'default',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at REAL,
    model TEXT,
    settings TEXT NOT NULL DEFAULT '{}',
    leaf_id TEXT,
    summary TEXT,
    summary_upto TEXT
);
CREATE INDEX IF NOT EXISTS idx_conv_list ON conversations(archived, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    attachments TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'done',
    error TEXT,
    error_code TEXT,
    model TEXT,
    stats TEXT NOT NULL DEFAULT '{}',
    tokens INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_msg_parent ON messages(parent_id);

CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    model TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'active',
    pinned INTEGER NOT NULL DEFAULT 0,
    locked INTEGER NOT NULL DEFAULT 0,
    derived INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT 'reflection',
    seen INTEGER NOT NULL DEFAULT 0,
    use_count INTEGER NOT NULL DEFAULT 0,
    last_used_at REAL,
    expires_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status, importance DESC);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    batch_from REAL NOT NULL,
    batch_to REAL NOT NULL,
    PRIMARY KEY (memory_id, conversation_id, batch_from)
);
CREATE INDEX IF NOT EXISTS idx_src_conv ON memory_sources(conversation_id);

CREATE TABLE IF NOT EXISTS reflections (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    upto_at REAL NOT NULL DEFAULT 0,
    reflected_at REAL NOT NULL DEFAULT 0,
    requested INTEGER NOT NULL DEFAULT 0
);
"""

SCHEMA_VERSION = 2
_PLAIN_HEADER = b"SQLite format 3\x00"
_KEY_FORMAT = re.compile(r"^[0-9a-f]{64}$")


class DatabaseLocked(Exception):
    pass


class WrongKey(Exception):
    pass


def shred(path: Path) -> None:
    """先覆写再删除。SSD 上不保证擦除物理块，这里只是减少明文残留的机会。"""
    try:
        size = path.stat().st_size
        with open(path, "r+b", buffering=0) as handle:
            zeros = bytes(1024 * 1024)
            remaining = size
            while remaining > 0:
                remaining -= handle.write(zeros[: min(remaining, len(zeros))])
            os.fsync(handle.fileno())
    except OSError:
        pass
    path.unlink(missing_ok=True)


STARTER_PROMPTS = [
    ("简洁助手", "你是一个简洁、准确的助手。先给结论，再给必要的解释，不说套话。"),
    ("代码审查", "你是资深工程师。审查用户提供的代码：指出真实的 bug、边界条件和安全问题，并给出可直接使用的修改。不要纠结风格问题。"),
    ("中英互译", "你是专业译者。中文译成英文，英文译成中文。只输出译文，保持原文的语气和格式。"),
    ("写作润色", "你是编辑。在不改变原意的前提下润色用户的文字，让它更清楚、更自然，并简短说明主要改动。"),
]
DAILY_CHAT_PROMPT = (
    "日常聊天",
    "你是我的老朋友，我们在随便聊天。说话自然、口语一点，回复简短，不用列表和小标题。"
    "我倾诉的时候先听、先共情，别急着给建议；需要建议时我会说。可以有自己的看法，也可以反问我。",
)


def new_id() -> str:
    return uuid.uuid4().hex


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._key: str | None = None
        self._generation = 0
        self._connections: list[sqlite3.Connection] = []
        self._registry = threading.Lock()
        self.needs_checkpoint = False

    # ------------------------------------------------------------ 连接
    @property
    def is_open(self) -> bool:
        return self._key is not None

    def _sidecar(self, suffix: str) -> Path:
        return self.path.with_name(self.path.name + suffix)

    def _connect(self, key: str, path: Path | None = None) -> sqlite3.Connection:
        if not _KEY_FORMAT.match(key):
            raise ValueError("invalid key format")
        c = sqlite3.connect(path or self.path, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute(f"PRAGMA key = \"x'{key}'\"")
        c.execute("PRAGMA cipher_memory_security = ON")
        return c

    @property
    def conn(self) -> sqlite3.Connection:
        key = self._key
        if key is None:
            raise DatabaseLocked()
        current = getattr(self._local, "conn", None)
        if current is not None and getattr(self._local, "generation", -1) == self._generation:
            return current
        c = self._connect(key)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA secure_delete=ON")
        c.execute("PRAGMA temp_store=MEMORY")  # 排序和临时表都不落盘
        with self._registry:
            self._connections.append(c)
        self._local.conn = c
        self._local.generation = self._generation
        return c

    def _is_plaintext(self) -> bool:
        try:
            with open(self.path, "rb") as handle:
                return handle.read(16) == _PLAIN_HEADER
        except OSError:
            return False

    def _recover_swap(self) -> None:
        """处理上次替换数据库时的中断：新库完整就启用，然后清掉残留。"""
        fresh, stale = self._sidecar(".new"), self._sidecar(".wipe")
        if not self.path.exists() and fresh.exists():
            os.replace(fresh, self.path)
        if self.path.exists():
            shred(fresh)
            shred(stale)

    def _export_to(self, source: sqlite3.Connection, target: Path, key: str) -> None:
        target.unlink(missing_ok=True)
        version = source.execute("PRAGMA user_version").fetchone()[0]
        source.execute("ATTACH DATABASE ? AS target KEY ?", (str(target), f"x'{key}'"))
        source.execute("SELECT sqlcipher_export('target')")
        source.execute(f"PRAGMA target.user_version = {int(version)}")
        source.commit()
        source.execute("DETACH DATABASE target")

    def _encrypt_plaintext(self, key: str) -> None:
        """把旧版本的明文数据库转成加密库，校验无误后才清除明文。"""
        fresh = self._sidecar(".new")
        source = sqlite3.connect(self.path)
        try:
            source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            count = source.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
            self._export_to(source, fresh, key)
        finally:
            source.close()
        check = self._connect(key, fresh)
        try:
            if check.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] != count:
                raise RuntimeError("encrypted copy failed verification")
        finally:
            check.close()
        shred(self._sidecar("-wal"))
        shred(self._sidecar("-shm"))
        stale = self._sidecar(".wipe")
        os.replace(self.path, stale)
        os.replace(fresh, self.path)
        shred(stale)

    def open(self, key: str) -> None:
        self._recover_swap()
        if self._is_plaintext():
            self._encrypt_plaintext(key)
        os.close(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600))  # SQLite 的 -wal 文件会继承这个权限
        probe = self._connect(key)
        try:
            probe.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            raise WrongKey() from exc
        finally:
            probe.close()
        self._key = key
        self._generation += 1
        with self.conn as c:
            c.executescript(SCHEMA)
        os.chmod(self.path, 0o600)
        self._migrate()
        self._seed_prompts()

    def _close_connections(self) -> None:
        self._generation += 1
        with self._registry:
            connections, self._connections = self._connections, []
        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error:
                pass

    def close(self) -> None:
        if self._key is not None:
            self.checkpoint()
        self._close_connections()
        self._key = None

    def rekey(self, new_key: str) -> None:
        """用新密钥重写整个数据库（新盐、新密文），旧文件随后被替换。"""
        fresh = self._sidecar(".new")
        self.checkpoint()
        source = self._connect(self._key)  # type: ignore[arg-type]
        try:
            self._export_to(source, fresh, new_key)
        finally:
            source.close()
        check = self._connect(new_key, fresh)
        try:
            check.execute("SELECT count(*) FROM conversations").fetchone()
        finally:
            check.close()
        self._close_connections()
        shred(self._sidecar("-wal"))
        shred(self._sidecar("-shm"))
        os.replace(fresh, self.path)
        os.chmod(self.path, 0o600)
        self._key = new_key
        self._generation += 1

    def wipe(self) -> None:
        self._close_connections()
        self._key = None
        for suffix in ("", "-wal", "-shm", ".new", ".wipe"):
            shred(self._sidecar(suffix))

    def checkpoint(self) -> None:
        """把 WAL 合并回主库并截断，让被删内容不再留在日志文件里。"""
        try:
            row = self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            self.needs_checkpoint = bool(row and row[0])
        except sqlite3.OperationalError:
            self.needs_checkpoint = True

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 2:
            with self.conn as c:
                c.execute("DELETE FROM usage")  # 旧版本默认记录用量，现在改为不留日志
        if version != SCHEMA_VERSION:
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _seed_prompts(self) -> None:
        row = self.conn.execute("SELECT value FROM settings WHERE key='_seeded'").fetchone()
        seeded = int(row["value"]) if row else 0
        wanted = []
        if seeded < 1:
            wanted += STARTER_PROMPTS
        if seeded < 2:
            wanted.append(DAILY_CHAT_PROMPT)
        if not wanted:
            return
        now = time.time()
        with self.conn as c:
            for name, content in wanted:
                c.execute(
                    "INSERT INTO prompts VALUES (?,?,?,?,?)",
                    (new_id(), name, content, now, now),
                )
            c.execute(
                "INSERT INTO settings(key,value) VALUES('_seeded','2') "
                "ON CONFLICT(key) DO UPDATE SET value='2'"
            )

    # ------------------------------------------------------------- 设置
    def get_settings(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT value FROM settings WHERE key='app'").fetchone()
        stored = json.loads(row["value"]) if row else {}
        return deep_merge(DEFAULT_SETTINGS, stored)

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        row = self.conn.execute("SELECT value FROM settings WHERE key='app'").fetchone()
        stored = deep_merge(json.loads(row["value"]) if row else {}, patch)
        with self.conn as c:
            c.execute(
                "INSERT INTO settings(key,value) VALUES('app',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(stored, ensure_ascii=False),),
            )
        return deep_merge(DEFAULT_SETTINGS, stored)

    # ------------------------------------------------------------- 对话
    @staticmethod
    def _conv(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        d["settings"] = json.loads(d["settings"] or "{}")
        d["pinned"] = bool(d["pinned"])
        d["archived"] = bool(d["archived"])
        return d

    def create_conversation(
        self,
        title: str = "新对话",
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        conv_id: str | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        cid = conv_id or new_id()
        now = created_at or time.time()
        with self.conn as c:
            c.execute(
                "INSERT INTO conversations(id,title,created_at,updated_at,model,settings) "
                "VALUES(?,?,?,?,?,?)",
                (cid, title, now, now, model, json.dumps(settings or {}, ensure_ascii=False)),
            )
        return self.get_conversation(cid)  # type: ignore[return-value]

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id=?", (conv_id,)
        ).fetchone()
        return self._conv(row)

    def list_conversations(self, archived: bool = False) -> list[dict[str, Any]]:
        order = "archived_at DESC" if archived else "pinned DESC, updated_at DESC"
        rows = self.conn.execute(
            f"""SELECT c.id,c.title,c.created_at,c.updated_at,c.pinned,c.archived,
                       c.archived_at,c.model,c.settings,c.leaf_id,c.title_source,
                       c.summary,c.summary_upto,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count
                FROM conversations c WHERE c.archived=? ORDER BY {order}""",
            (1 if archived else 0,),
        ).fetchall()
        result = []
        for row in rows:
            conv = self._conv(row)
            conv.pop("summary", None)  # type: ignore[union-attr]
            result.append(conv)
        return result  # type: ignore[return-value]

    def update_conversation(self, conv_id: str, **fields: Any) -> dict[str, Any] | None:
        if "settings" in fields:
            fields["settings"] = json.dumps(fields["settings"], ensure_ascii=False)
        for flag in ("pinned", "archived"):
            if flag in fields:
                fields[flag] = 1 if fields[flag] else 0
        if not fields:
            return self.get_conversation(conv_id)
        cols = ",".join(f"{k}=?" for k in fields)
        with self.conn as c:
            c.execute(
                f"UPDATE conversations SET {cols} WHERE id=?",
                (*fields.values(), conv_id),
            )
        return self.get_conversation(conv_id)

    def touch(self, conv_id: str) -> None:
        self.update_conversation(conv_id, updated_at=time.time())

    def set_archived(self, conv_id: str, archived: bool) -> dict[str, Any] | None:
        return self.update_conversation(
            conv_id,
            archived=archived,
            archived_at=time.time() if archived else None,
            pinned=False,
        )

    def _drop_provenance(
        self, c: sqlite3.Connection, conv_id: str, at_times: list[float] | None = None
    ) -> int:
        """撤销对话（或其中部分消息）对记忆的贡献，失去依据的记忆一并删除。

        普通记忆在还有其他对话佐证时保留；反思和合并得出的记忆是多条内容的综合，
        只要有一个来源被删，就整条移除，下次反思时如果仍成立会重新得出。
        """
        rows = c.execute(
            "SELECT memory_id, batch_from, batch_to FROM memory_sources WHERE conversation_id=?",
            (conv_id,),
        ).fetchall()
        if at_times is not None:
            rows = [r for r in rows if any(r["batch_from"] <= t <= r["batch_to"] for t in at_times)]
        for row in rows:
            c.execute(
                "DELETE FROM memory_sources WHERE memory_id=? AND conversation_id=? AND batch_from=?",
                (row["memory_id"], conv_id, row["batch_from"]),
            )
        removed = 0
        for memory_id in {r["memory_id"] for r in rows}:
            memory = c.execute(
                """SELECT m.derived,
                          (SELECT COUNT(*) FROM memory_sources s WHERE s.memory_id=m.id) AS remaining
                   FROM memories m WHERE m.id=?""",
                (memory_id,),
            ).fetchone()
            if memory is not None and (memory["derived"] or memory["remaining"] == 0):
                c.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                removed += 1
        return removed

    def delete_conversation(self, conv_id: str) -> int:
        """彻底删除对话，返回同时被移除的记忆条数。"""
        with self.conn as c:
            removed = self._drop_provenance(c, conv_id)
            c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self.checkpoint()
        return removed

    def search(self, query: str, limit: int = 60) -> list[dict[str, Any]]:
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = self.conn.execute(
            r"""SELECT c.id, c.title, c.updated_at, c.archived,
                       (SELECT m.content FROM messages m
                         WHERE m.conversation_id=c.id AND m.content LIKE ? ESCAPE '\'
                         ORDER BY m.created_at DESC LIMIT 1) AS hit
                FROM conversations c
                WHERE c.title LIKE ? ESCAPE '\'
                   OR EXISTS (SELECT 1 FROM messages m
                               WHERE m.conversation_id=c.id AND m.content LIKE ? ESCAPE '\')
                ORDER BY c.updated_at DESC LIMIT ?""",
            (like, like, like, limit),
        ).fetchall()
        out = []
        needle = query.lower()
        for row in rows:
            item = dict(row)
            item["archived"] = bool(item["archived"])
            hit = (item.pop("hit") or "").replace("<think>", "").replace("</think>", "").strip()
            pos = hit.lower().find(needle)
            if pos >= 0:
                start = max(pos - 40, 0)
                snippet = hit[start : pos + len(query) + 80].replace("\n", " ")
                item["snippet"] = ("…" if start else "") + snippet
            else:
                item["snippet"] = ""
            out.append(item)
        return out

    # ------------------------------------------------------------- 消息
    @staticmethod
    def _msg(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        d["attachments"] = json.loads(d["attachments"] or "[]")
        d["stats"] = json.loads(d["stats"] or "{}")
        return d

    def add_message(
        self,
        conv_id: str,
        parent_id: str | None,
        role: str,
        content: str = "",
        *,
        attachments: list[dict[str, Any]] | None = None,
        status: str = "done",
        model: str | None = None,
        msg_id: str | None = None,
        created_at: float | None = None,
        stats: dict[str, Any] | None = None,
        tokens: int | None = None,
    ) -> dict[str, Any]:
        mid = msg_id or new_id()
        with self.conn as c:
            c.execute(
                "INSERT INTO messages(id,conversation_id,parent_id,role,content,attachments,"
                "status,model,stats,tokens,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    conv_id,
                    parent_id,
                    role,
                    content,
                    json.dumps(attachments or [], ensure_ascii=False),
                    status,
                    model,
                    json.dumps(stats or {}, ensure_ascii=False),
                    tokens,
                    created_at or time.time(),
                ),
            )
        return self.get_message(mid)  # type: ignore[return-value]

    def get_message(self, msg_id: str) -> dict[str, Any] | None:
        return self._msg(
            self.conn.execute("SELECT * FROM messages WHERE id=?", (msg_id,)).fetchone()
        )

    def update_message(self, msg_id: str, **fields: Any) -> None:
        if "stats" in fields:
            fields["stats"] = json.dumps(fields["stats"], ensure_ascii=False)
        cols = ",".join(f"{k}=?" for k in fields)
        with self.conn as c:
            c.execute(f"UPDATE messages SET {cols} WHERE id=?", (*fields.values(), msg_id))

    def set_token_counts(self, counts: dict[str, int]) -> None:
        with self.conn as c:
            c.executemany(
                "UPDATE messages SET tokens=? WHERE id=?",
                [(count, mid) for mid, count in counts.items()],
            )

    def all_messages(self, conv_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at, rowid",
            (conv_id,),
        ).fetchall()
        return [self._msg(r) for r in rows]  # type: ignore[misc]

    def path_to(self, conv_id: str, message_id: str | None) -> list[dict[str, Any]]:
        """从根到 message_id（含）的消息链，附带兄弟分支信息。"""
        if message_id is None:
            return []
        messages = self.all_messages(conv_id)
        by_id = {m["id"]: m for m in messages}
        children: dict[str | None, list[str]] = {}
        for m in messages:
            children.setdefault(m["parent_id"], []).append(m["id"])
        path: list[dict[str, Any]] = []
        cursor = by_id.get(message_id)
        while cursor is not None:
            siblings = children.get(cursor["parent_id"], [cursor["id"]])
            cursor["siblings"] = siblings
            cursor["sibling_index"] = siblings.index(cursor["id"])
            path.append(cursor)
            cursor = by_id.get(cursor["parent_id"]) if cursor["parent_id"] else None
        path.reverse()
        return path

    def deepest_descendant(self, conv_id: str, message_id: str) -> str:
        """沿最新子节点走到底，用于切换分支后定位末端。"""
        messages = self.all_messages(conv_id)
        latest_child: dict[str, str] = {}
        for m in messages:  # 已按创建时间排序，后写入的覆盖先写入的
            if m["parent_id"]:
                latest_child[m["parent_id"]] = m["id"]
        cursor = message_id
        while cursor in latest_child:
            cursor = latest_child[cursor]
        return cursor

    def delete_subtree(self, conv_id: str, message_id: str) -> str | None:
        """删除消息及其全部后代，返回建议的新 leaf。"""
        messages = self.all_messages(conv_id)
        by_id = {m["id"]: m for m in messages}
        target = by_id.get(message_id)
        if target is None:
            return None
        children: dict[str, list[str]] = {}
        for m in messages:
            if m["parent_id"]:
                children.setdefault(m["parent_id"], []).append(m["id"])
        doomed, stack = [], [message_id]
        while stack:
            current = stack.pop()
            doomed.append(current)
            stack.extend(children.get(current, []))
        with self.conn as c:
            self._drop_provenance(c, conv_id, [by_id[d]["created_at"] for d in doomed])
            c.executemany("DELETE FROM messages WHERE id=?", [(d,) for d in doomed])
        self.checkpoint()

        survivors = [
            m for m in messages
            if m["parent_id"] == target["parent_id"] and m["id"] != message_id
        ]
        if survivors:
            return self.deepest_descendant(conv_id, survivors[-1]["id"])
        return target["parent_id"]

    def mark_interrupted(self) -> None:
        """服务重启时，把未完成的消息标记为中断。"""
        with self.conn as c:
            c.execute(
                "UPDATE messages SET status='interrupted' WHERE status IN ('pending','streaming')"
            )

    # ----------------------------------------------------------- 提示词
    def list_prompts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM prompts ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def save_prompt(self, prompt_id: str | None, name: str, content: str) -> dict[str, Any]:
        now = time.time()
        pid = prompt_id or new_id()
        with self.conn as c:
            c.execute(
                "INSERT INTO prompts(id,name,content,created_at,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, content=excluded.content, "
                "updated_at=excluded.updated_at",
                (pid, name, content, now, now),
            )
        row = self.conn.execute("SELECT * FROM prompts WHERE id=?", (pid,)).fetchone()
        return dict(row)

    def delete_prompt(self, prompt_id: str) -> None:
        with self.conn as c:
            c.execute("DELETE FROM prompts WHERE id=?", (prompt_id,))

    # ------------------------------------------------------------- 用量
    def add_usage(
        self, model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int, seconds: float
    ) -> None:
        if not self.get_settings()["privacy"]["usage_stats"]:
            return  # 默认关闭：除对话本身外不留任何记录
        day = datetime.now().strftime("%Y-%m-%d")
        with self.conn as c:
            c.execute(
                """INSERT INTO usage VALUES (?,?,1,?,?,?,?)
                   ON CONFLICT(day,model) DO UPDATE SET
                     requests=requests+1,
                     prompt_tokens=prompt_tokens+excluded.prompt_tokens,
                     cached_tokens=cached_tokens+excluded.cached_tokens,
                     completion_tokens=completion_tokens+excluded.completion_tokens,
                     seconds=seconds+excluded.seconds""",
                (day, model, prompt_tokens, cached_tokens, completion_tokens, seconds),
            )

    def usage_summary(self, days: int = 30) -> dict[str, Any]:
        by_day = self.conn.execute(
            """SELECT day, SUM(requests) requests, SUM(prompt_tokens) prompt_tokens,
                      SUM(cached_tokens) cached_tokens, SUM(completion_tokens) completion_tokens,
                      SUM(seconds) seconds
               FROM usage GROUP BY day ORDER BY day DESC LIMIT ?""",
            (days,),
        ).fetchall()
        by_model = self.conn.execute(
            """SELECT model, SUM(requests) requests, SUM(prompt_tokens) prompt_tokens,
                      SUM(cached_tokens) cached_tokens, SUM(completion_tokens) completion_tokens,
                      SUM(seconds) seconds
               FROM usage GROUP BY model ORDER BY SUM(completion_tokens) DESC"""
        ).fetchall()
        counts = self.conn.execute(
            """SELECT (SELECT COUNT(*) FROM conversations WHERE archived=0) active,
                      (SELECT COUNT(*) FROM conversations WHERE archived=1) archived,
                      (SELECT COUNT(*) FROM messages) messages"""
        ).fetchone()
        return {
            "by_day": [dict(r) for r in reversed(by_day)],
            "by_model": [dict(r) for r in by_model],
            "counts": dict(counts),
        }

    def clear_usage(self) -> None:
        with self.conn as c:
            c.execute("DELETE FROM usage")
        self.checkpoint()

    # ------------------------------------------------------- 批量 / 备份
    def dump_memories(self) -> list[dict[str, Any]]:
        memories = [dict(r) for r in self.conn.execute("SELECT * FROM memories").fetchall()]
        for memory in memories:
            memory["sources"] = [
                dict(r)
                for r in self.conn.execute(
                    "SELECT conversation_id, batch_from, batch_to FROM memory_sources WHERE memory_id=?",
                    (memory["id"],),
                ).fetchall()
            ]
        return memories

    def restore_memories(self, items: list[dict[str, Any]], mapping: dict[str, str]) -> None:
        """导入记忆并把来源映射到新的对话 id；来源全部丢失的提炼记忆不导入。"""
        columns = (
            "kind", "content", "importance", "status", "pinned", "locked", "derived",
            "origin", "seen", "use_count", "last_used_at", "expires_at", "created_at", "updated_at",
        )
        with self.conn as c:
            for item in items:
                sources = [s for s in item.get("sources") or [] if s.get("conversation_id") in mapping]
                if item.get("origin") != "manual" and not sources:
                    continue
                memory_id = new_id()
                c.execute(
                    f"INSERT INTO memories(id,{','.join(columns)}) VALUES(?,{','.join('?' * len(columns))})",
                    (memory_id, *[item.get(column) for column in columns]),
                )
                for source in sources:
                    c.execute(
                        "INSERT OR IGNORE INTO memory_sources VALUES(?,?,?,?)",
                        (memory_id, mapping[source["conversation_id"]], source["batch_from"], source["batch_to"]),
                    )

    def bulk(self, ids: Iterable[str], action: str) -> int:
        count = 0
        for cid in ids:
            if action == "archive":
                self.set_archived(cid, True)
            elif action == "unarchive":
                self.set_archived(cid, False)
            elif action == "delete":
                self.delete_conversation(cid)
            elif action == "pin":
                self.update_conversation(cid, pinned=True)
            elif action == "unpin":
                self.update_conversation(cid, pinned=False)
            else:
                raise ValueError(f"unknown action: {action}")
            count += 1
        return count

    def dump_conversation(self, conv_id: str) -> dict[str, Any] | None:
        conv = self.get_conversation(conv_id)
        if conv is None:
            return None
        conv["messages"] = self.all_messages(conv_id)
        return conv

    def restore_conversation(self, data: dict[str, Any]) -> dict[str, Any]:
        """导入一份导出的对话，全部重新分配 id，避免与现有数据冲突。"""
        conv = self.create_conversation(
            title=str(data.get("title") or "导入的对话"),
            model=data.get("model"),
            settings=data.get("settings") or {},
            created_at=data.get("created_at"),
        )
        mapping: dict[str, str] = {}
        for m in data.get("messages") or []:
            old_id = m.get("id") or new_id()
            mid = new_id()
            mapping[old_id] = mid
            status = m.get("status", "done")
            self.add_message(
                conv["id"],
                mapping.get(m.get("parent_id")),
                m.get("role", "user"),
                m.get("content", ""),
                attachments=m.get("attachments") or [],
                status="interrupted" if status in ("pending", "streaming") else status,
                model=m.get("model"),
                msg_id=mid,
                created_at=m.get("created_at"),
                stats=m.get("stats") or {},
                tokens=m.get("tokens"),
            )
        return self.update_conversation(
            conv["id"],
            leaf_id=mapping.get(data.get("leaf_id")),
            title_source="user",
            pinned=bool(data.get("pinned")),
            archived=bool(data.get("archived")),
            archived_at=data.get("archived_at"),
            summary=data.get("summary"),
            summary_upto=mapping.get(data.get("summary_upto")),
            updated_at=data.get("updated_at") or time.time(),
        )  # type: ignore[return-value]
