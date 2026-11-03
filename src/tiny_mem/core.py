"""Core types and protocol for tiny-mem."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryItem:
    """A single memory record."""

    id: str = field(default_factory=_new_id)
    text: str = ""
    embedding: Optional[List[float]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    importance: float = 1.0

    def to_row(self) -> Dict[str, Any]:
        """Serialize for SQLite storage."""
        import json
        return {
            "id": self.id,
            "text": self.text,
            "embedding": json.dumps(self.embedding) if self.embedding else None,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at,
            "importance": self.importance,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "MemoryItem":
        """Deserialize from SQLite row."""
        import json
        return cls(
            id=row["id"],
            text=row["text"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            user_id=row["user_id"],
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            importance=row["importance"],
        )


@dataclass
class RecallResult:
    """A memory item plus its score breakdown."""

    item: MemoryItem
    score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    recency_score: Optional[float] = None

    @property
    def text(self) -> str:
        return self.item.text

    @property
    def id(self) -> str:
        return self.item.id


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a string into a list of floats."""

    dim: int

    def __call__(self, text: str) -> List[float]:
        ...


@runtime_checkable
class Memory(Protocol):
    """Common memory interface across all backends."""

    def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> str:
        ...

    def recall(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[RecallResult]:
        ...

    def recall_by_session(self, session_id: str) -> List[MemoryItem]:
        ...

    def forget(
        self, user_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> int:
        ...

    def count(self, user_id: Optional[str] = None) -> int:
        ...

    def compact(self) -> None:
        ...


class MemoryScope:
    """Chained scoping helper: mem.user("u1").session("s42").add(...)"""

    def __init__(self, mem: "Memory", user_id: Optional[str] = None, session_id: Optional[str] = None):
        self._mem = mem
        self._user_id = user_id
        self._session_id = session_id

    def user(self, user_id: str) -> "MemoryScope":
        return MemoryScope(self._mem, user_id=user_id, session_id=self._session_id)

    def session(self, session_id: str) -> "MemoryScope":
        return MemoryScope(self._mem, user_id=self._user_id, session_id=session_id)

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> str:
        return self._mem.add(
            text,
            user_id=self._user_id,
            session_id=self._session_id,
            metadata=metadata,
            importance=importance,
        )

    def recall(self, query: str, limit: int = 5, min_score: float = 0.0) -> List[RecallResult]:
        return self._mem.recall(
            query,
            user_id=self._user_id,
            session_id=self._session_id,
            limit=limit,
            min_score=min_score,
        )

    def forget(self) -> int:
        return self._mem.forget(user_id=self._user_id, session_id=self._session_id)

    def count(self) -> int:
        return self._mem.count(user_id=self._user_id)
