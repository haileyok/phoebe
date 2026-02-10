"""
Arena context — holds references to bounties, submissions, and evaluations.

Passed into ToolContext so that tools in the Deno sandbox can access
arena state.
"""

from src.arena.models import Bounty, EvaluationResult, Submission


class ArenaContext:
    """
    Shared arena state accessible from tool handlers.

    In a production deployment this would be backed by ClickHouse. For the
    prototype it uses the in-memory dicts from ArenaServer (passed by ref).
    """

    def __init__(self) -> None:
        self.bounties: dict[str, Bounty] = {}
        self.submissions: dict[str, Submission] = {}
        self.evaluations: dict[str, EvaluationResult] = {}
        self.active_bounty: Bounty | None = None
