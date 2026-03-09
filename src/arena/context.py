"""
Arena context — provides access to the persistent store from tool handlers.

Passed into ToolContext so that Deno sandbox tools can query arena state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.arena.models import Bounty
    from src.arena.store import ArenaStore


class ArenaContext:
    """
    Shared arena state accessible from tool handlers.

    Backed by the ArenaStore (ClickHouse) in production.
    """

    def __init__(self, store: ArenaStore | None = None) -> None:
        self._store = store
        self.active_bounty: Bounty | None = None

    @property
    def store(self) -> ArenaStore:
        if self._store is None:
            raise RuntimeError("ArenaStore not configured")
        return self._store
