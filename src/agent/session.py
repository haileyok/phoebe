"""
Session and memory layer.

Provides pluggable conversation persistence so the Agent class
never touches storage directly.

Backends:
  - InMemorySessionStore    → in-process dict (original behaviour, no persistence)
  - FileSessionStore        → JSON files on disk (lightweight, single-node)
  - RedisSessionStore       → Redis (multi-process, multi-host; good for OpenClaw)
  - ClickHouseSessionStore  → ClickHouse (append-only log; doubles as training data)

Usage:
    store = RedisSessionStore(url="redis://localhost:6379")
    session = await store.get_or_create("user-discord-12345")
    session.append({"role": "user", "content": "hello"})
    await store.save(session)
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Session object ─────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    agent_name: str
    mode: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.updated_at = time.time()

    def clear(self) -> None:
        self.messages = []
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "mode": self.mode,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(**data)


# ── Abstract store ─────────────────────────────────────────────────────────────

class SessionStore(ABC):
    """
    Abstract persistence layer for agent conversations.

    Use InMemorySessionStore for tests and single-process deployments,
    FileSessionStore for lightweight single-node persistence,
    RedisSessionStore for multi-process/multi-host deployments, and
    ClickHouseSessionStore when you want conversation history as RL training data.
    """

    @abstractmethod
    async def get_or_create(
        self,
        session_id: str,
        agent_name: str = "unknown",
        mode: str = "default",
    ) -> Session:
        ...

    @abstractmethod
    async def save(self, session: Session) -> None:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    async def list_sessions(self, agent_name: str | None = None) -> list[str]:
        ...


# ── In-memory (default, original behaviour) ────────────────────────────────────

class InMemorySessionStore(SessionStore):
    """
    Ephemeral in-process store. Identical to the original self._conversation
    list but now addressable by session_id so multiple users/channels work.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get_or_create(
        self, session_id: str, agent_name: str = "unknown", mode: str = "default"
    ) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id, agent_name=agent_name, mode=mode
            )
        return self._sessions[session_id]

    async def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def list_sessions(self, agent_name: str | None = None) -> list[str]:
        if agent_name:
            return [s for s, v in self._sessions.items() if v.agent_name == agent_name]
        return list(self._sessions.keys())


# ── File-based store ───────────────────────────────────────────────────────────

class FileSessionStore(SessionStore):
    """
    Persists each session as a JSON file under base_dir/<session_id>.json.
    Good for single-node deployments and local development.
    """

    def __init__(self, base_dir: str = ".sessions") -> None:
        self._base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        safe = session_id.replace("/", "_").replace(":", "_")
        return os.path.join(self._base_dir, f"{safe}.json")

    async def get_or_create(
        self, session_id: str, agent_name: str = "unknown", mode: str = "default"
    ) -> Session:
        path = self._path(session_id)
        if os.path.exists(path):
            with open(path) as f:
                return Session.from_dict(json.load(f))
        return Session(session_id=session_id, agent_name=agent_name, mode=mode)

    async def save(self, session: Session) -> None:
        with open(self._path(session.session_id), "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    async def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)

    async def list_sessions(self, agent_name: str | None = None) -> list[str]:
        sessions = []
        for fname in os.listdir(self._base_dir):
            if not fname.endswith(".json"):
                continue
            sid = fname[:-5]
            if agent_name:
                path = self._path(sid)
                with open(path) as f:
                    data = json.load(f)
                if data.get("agent_name") != agent_name:
                    continue
            sessions.append(sid)
        return sessions


# ── Redis store ────────────────────────────────────────────────────────────────

class RedisSessionStore(SessionStore):
    """
    Redis-backed store. Suitable for multi-process OpenClaw deployments
    where multiple agent workers share session state.

    Requires: pip install redis[asyncio]

    Keys: sara:session:<session_id>
    TTL:  ttl_seconds (default 7 days). Refreshed on every save().
    """

    def __init__(self, url: str = "redis://localhost:6379", ttl_seconds: int = 604_800) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError("pip install redis[asyncio]") from e
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"sara:session:{session_id}"

    async def get_or_create(
        self, session_id: str, agent_name: str = "unknown", mode: str = "default"
    ) -> Session:
        raw = await self._redis.get(self._key(session_id))
        if raw:
            return Session.from_dict(json.loads(raw))
        return Session(session_id=session_id, agent_name=agent_name, mode=mode)

    async def save(self, session: Session) -> None:
        await self._redis.setex(
            self._key(session.session_id),
            self._ttl,
            json.dumps(session.to_dict()),
        )

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def list_sessions(self, agent_name: str | None = None) -> list[str]:
        pattern = "sara:session:*"
        keys = [k async for k in self._redis.scan_iter(pattern)]
        if not agent_name:
            return [k.replace("sara:session:", "") for k in keys]
        result = []
        for key in keys:
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                if data.get("agent_name") == agent_name:
                    result.append(data["session_id"])
        return result


# ── ClickHouse append-only store ───────────────────────────────────────────────

class ClickHouseSessionStore(SessionStore):
    """
    ClickHouse-backed store. Every message is appended as a row —
    this doubles as the interaction log for the RL learning pipeline.

    Schema (run once):
        CREATE TABLE sara_sessions (
            session_id   String,
            agent_name   String,
            mode         String,
            role         String,
            content      String,
            metadata     String,   -- JSON
            created_at   DateTime64(3) DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (session_id, created_at);

    Requires: pip install clickhouse-driver
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        database: str = "default",
        user: str = "default",
        password: str = "",
    ) -> None:
        try:
            from clickhouse_driver import Client
        except ImportError as e:
            raise ImportError("pip install clickhouse-driver") from e
        self._client = Client(host=host, port=port, database=database,
                              user=user, password=password)
        # Use in-memory as read cache to avoid full-table scans per message
        self._cache = InMemorySessionStore()

    async def get_or_create(
        self, session_id: str, agent_name: str = "unknown", mode: str = "default"
    ) -> Session:
        cached = self._cache._sessions.get(session_id)
        if cached:
            return cached

        rows = self._client.execute(
            "SELECT role, content, metadata FROM sara_sessions "
            "WHERE session_id = %(sid)s ORDER BY created_at ASC",
            {"sid": session_id},
        )
        if not rows:
            session = Session(session_id=session_id, agent_name=agent_name, mode=mode)
        else:
            messages = [
                {**json.loads(meta or "{}"), "role": role, "content": json.loads(content)}
                for role, content, meta in rows
            ]
            session = Session(
                session_id=session_id,
                agent_name=agent_name,
                mode=mode,
                messages=messages,
            )
        self._cache._sessions[session_id] = session
        return session

    async def save(self, session: Session) -> None:
        if not session.messages:
            return
        last = session.messages[-1]
        self._client.execute(
            "INSERT INTO sara_sessions (session_id, agent_name, mode, role, content, metadata) VALUES",
            [{
                "session_id": session.session_id,
                "agent_name": session.agent_name,
                "mode": session.mode,
                "role": last.get("role", "unknown"),
                "content": json.dumps(last.get("content", "")),
                "metadata": json.dumps(session.metadata),
            }],
        )
        self._cache._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._client.execute(
            "ALTER TABLE sara_sessions DELETE WHERE session_id = %(sid)s",
            {"sid": session_id},
        )
        await self._cache.delete(session_id)

    async def list_sessions(self, agent_name: str | None = None) -> list[str]:
        if agent_name:
            rows = self._client.execute(
                "SELECT DISTINCT session_id FROM sara_sessions WHERE agent_name = %(a)s",
                {"a": agent_name},
            )
        else:
            rows = self._client.execute("SELECT DISTINCT session_id FROM sara_sessions")
        return [r[0] for r in rows]
