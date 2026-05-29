"""
RL Layer 1 — Interaction Logging
=================================

Captures every agent interaction as a structured record in ClickHouse.
This is the foundational data layer; without it, no RL approach is possible.

Every call to Agent.chat() should produce one InteractionRecord.
The reward field starts as None and is filled in asynchronously by:
  - Layer 2 (RewardComputer) for automatic signals
  - Human reviewers for soft signals (judge mode accuracy, admin decisions)

Schema (run once against your ClickHouse instance):

    CREATE TABLE sara_interactions (
        interaction_id   String,
        session_id       String,
        agent_name       String,
        mode             String,
        model_name       String,
        provider         String,
        user_message     String,
        response_text    String,
        tool_calls       String,   -- JSON array
        tool_results     String,   -- JSON array
        outcome          String,   -- JSON dict: scores, classifications, etc.
        reward           Nullable(Float32),
        human_label      Nullable(String),
        created_at       DateTime64(3) DEFAULT now(),
        updated_at       DateTime64(3) DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (agent_name, mode, created_at);

    -- Index for RL retrieval queries
    CREATE INDEX idx_reward ON sara_interactions (reward) TYPE minmax GRANULARITY 4;
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: dict[str, Any]
    duration_ms: float


@dataclass
class InteractionRecord:
    """One complete agent turn: user message → response (including all tool calls)."""

    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    agent_name: str = ""
    mode: str = ""
    model_name: str = ""
    provider: str = ""
    user_message: str = ""
    response_text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    human_label: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "mode": self.mode,
            "model_name": self.model_name,
            "provider": self.provider,
            "user_message": self.user_message,
            "response_text": self.response_text,
            "tool_calls": json.dumps([
                {
                    "name": tc.tool_name,
                    "input": tc.tool_input,
                    "result": tc.tool_result,
                    "duration_ms": tc.duration_ms,
                }
                for tc in self.tool_calls
            ]),
            "tool_results": json.dumps([tc.tool_result for tc in self.tool_calls]),
            "outcome": json.dumps(self.outcome),
            "reward": self.reward,
            "human_label": self.human_label,
            "created_at": self.created_at,
        }


class InteractionStore:
    """
    Writes InteractionRecords to ClickHouse.
    Reads them back for RL training and retrieval.

    Requires: pip install clickhouse-driver
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        database: str = "default",
        user: str = "default",
        password: str = "",
        table: str = "sara_interactions",
    ) -> None:
        from clickhouse_driver import Client
        self._client = Client(host=host, port=port, database=database,
                              user=user, password=password)
        self._table = table

    async def save(self, record: InteractionRecord) -> None:
        self._client.execute(
            f"INSERT INTO {self._table} VALUES",
            [record.to_dict()],
        )

    async def set_reward(self, interaction_id: str, reward: float) -> None:
        self._client.execute(
            f"ALTER TABLE {self._table} UPDATE reward = %(r)s, "
            f"updated_at = now() WHERE interaction_id = %(id)s",
            {"r": reward, "id": interaction_id},
        )

    async def set_human_label(self, interaction_id: str, label: str) -> None:
        self._client.execute(
            f"ALTER TABLE {self._table} UPDATE human_label = %(l)s, "
            f"updated_at = now() WHERE interaction_id = %(id)s",
            {"l": label, "id": interaction_id},
        )

    async def get_high_reward_examples(
        self,
        agent_name: str,
        mode: str,
        min_reward: float = 0.7,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch top interactions for Layer 3 few-shot injection."""
        rows = self._client.execute(
            f"""
            SELECT interaction_id, user_message, response_text,
                   tool_calls, outcome, reward
            FROM {self._table}
            WHERE agent_name = %(a)s
              AND mode = %(m)s
              AND reward >= %(r)s
            ORDER BY reward DESC
            LIMIT %(l)s
            """,
            {"a": agent_name, "m": mode, "r": min_reward, "l": limit},
        )
        cols = ["interaction_id", "user_message", "response_text",
                "tool_calls", "outcome", "reward"]
        return [dict(zip(cols, row)) for row in rows]

    async def get_unlabelled(
        self, agent_name: str, mode: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch interactions that still need a reward label."""
        rows = self._client.execute(
            f"""
            SELECT interaction_id, user_message, response_text, outcome
            FROM {self._table}
            WHERE agent_name = %(a)s AND mode = %(m)s AND reward IS NULL
            ORDER BY created_at DESC
            LIMIT %(l)s
            """,
            {"a": agent_name, "m": mode, "l": limit},
        )
        cols = ["interaction_id", "user_message", "response_text", "outcome"]
        return [dict(zip(cols, row)) for row in rows]
