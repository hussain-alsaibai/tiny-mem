"""SQLite-backed persistence with WAL mode for safe concurrent reads."""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .core import MemoryItem


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   TEXT,
    user_id     TEXT,
    session_id  TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    importance  REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
"""


class SQLiteStore:
    """Thread-safe SQLite store with WAL mode.

    Used by all tiny-mem backends. The embedding column stores JSON;
    vector backends do in-memory similarity search after loading.
    For >10k vectors per user, switch to a real vector index.
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self._local = threading.local()
        # Bootstrap a connection on the constructing thread.
        self._conn = self._make_conn()
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage txns explicitly
        )
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        # Per-thread connection so WAL + autocommit plays nicely.
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._make_conn()
            if self.path != ":memory:":
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def insert(self, item: MemoryItem) -> None:
        row = item.to_row()
        with self._txn() as c:
            c.execute(
                """INSERT INTO memories
                   (id, text, embedding, user_id, session_id, metadata, created_at, importance)
                   VALUES (:id, :text, :embedding, :user_id, :session_id, :metadata, :created_at, :importance)
                """,
                row,
            )

    def delete(self, item_id: str) -> bool:
        with self._txn() as c:
            cur = c.execute("DELETE FROM memories WHERE id = ?", (item_id,))
            return cur.rowcount > 0

    def delete_where(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._txn() as c:
            cur = c.execute(f"DELETE FROM memories{where}", params)
            return cur.rowcount

    def get(self, item_id: str) -> Optional[MemoryItem]:
        c = self._get_conn()
        row = c.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()
        return MemoryItem.from_row(dict(row)) if row else None

    def query(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = f" LIMIT {int(limit)}" if limit is not None else ""
        c = self._get_conn()
        rows = c.execute(
            f"SELECT * FROM memories{where} ORDER BY created_at DESC{lim}",
            params,
        ).fetchall()
        return [MemoryItem.from_row(dict(r)) for r in rows]

    def count(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        c = self._get_conn()
        row = c.execute(f"SELECT COUNT(*) AS n FROM memories{where}", params).fetchone()
        return int(row["n"])

    def clear(self) -> None:
        with self._txn() as c:
            c.execute("DELETE FROM memories")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        c = getattr(self._local, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
