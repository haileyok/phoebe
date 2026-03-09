"""
Data models for the Sandbox Arena.

These are the core domain objects: bounties, submissions, evaluations,
and the scoring result. All persisted to ClickHouse.
"""

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Bounties — funded by model developers
# ---------------------------------------------------------------------------


class BountyStatus(str, Enum):
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Bounty(BaseModel):
    bounty_id: str = Field(default_factory=_new_id)
    funder_wallet: str = ""
    target_model_endpoint: str = ""
    target_model_name: str = ""
    categories: list[str] = Field(default_factory=list)
    pool_usdc: float = 0.0
    remaining_usdc: float = 0.0
    max_payout_per_finding: float = 10.0
    created_at: float = Field(default_factory=_now)
    expires_at: float | None = None
    status: BountyStatus = BountyStatus.ACTIVE


# ---------------------------------------------------------------------------
# Submissions — sent by red teamers
# ---------------------------------------------------------------------------


class SubmissionStatus(str, Enum):
    QUEUED = "queued"
    EVALUATING = "evaluating"
    SCORED = "scored"
    PAID = "paid"
    REJECTED = "rejected"


class AttackPrompt(BaseModel):
    """A single adversarial prompt within a submission."""

    prompt: str
    category: str = ""
    technique: str = ""


class Submission(BaseModel):
    submission_id: str = Field(default_factory=_new_id)
    bounty_id: str = ""
    teamer_wallet: str = ""
    prompts: list[AttackPrompt] = Field(default_factory=list)
    submitted_at: float = Field(default_factory=_now)
    status: SubmissionStatus = SubmissionStatus.QUEUED


# ---------------------------------------------------------------------------
# Evaluation — produced by Phoebe scoring pipeline
# ---------------------------------------------------------------------------


class PromptEvaluation(BaseModel):
    """Evaluation result for a single adversarial prompt."""

    prompt: str
    category: str
    target_output: str = ""
    attack_success: bool = False
    severity: int = 0
    novelty_score: float = 0.0
    explanation: str = ""


class EvaluationResult(BaseModel):
    """Full evaluation of a submission."""

    submission_id: str
    bounty_id: str
    prompt_evaluations: list[PromptEvaluation] = Field(default_factory=list)
    total_score: float = 0.0
    payout_usdc: float = 0.0
    category_coverage: dict[str, int] = Field(default_factory=dict)
    duplicate_penalty: float = 0.0
    evaluated_at: float = Field(default_factory=_now)

    def summary(self) -> dict[str, Any]:
        successful = sum(1 for e in self.prompt_evaluations if e.attack_success)
        return {
            "submission_id": self.submission_id,
            "prompts_evaluated": len(self.prompt_evaluations),
            "successful_attacks": successful,
            "total_score": round(self.total_score, 4),
            "payout_usdc": round(self.payout_usdc, 4),
            "categories_hit": list(self.category_coverage.keys()),
        }
