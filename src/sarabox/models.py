"""
Pydantic models for Sara in a Box.

Core domain objects: skill files, attack categories, org configs,
attack submissions, classification results, and credit ledgers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AttackCategory(BaseModel):
    id: str  # snake_case identifier e.g. "treasury_manipulation"
    name: str  # display name e.g. "Treasury manipulation"
    description: str  # one sentence
    examples: list[str] = Field(default_factory=list)  # 3-5 example attack prompts
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    severity: str = "high"  # "low" | "medium" | "high" | "critical"


class SkillFile(BaseModel):
    skill_id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str  # org identifier (hashed, not plaintext name)
    org_type: str  # "dao" | "defi" | "nft" | "bridge" | "custom"
    display_name: str  # human-readable skill name
    system_prompt_extension: str  # extra context injected into Sara's judge prompt
    categories: list[AttackCategory] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    is_private: bool = True  # if True, categories are not shared with network


class OrgConfig(BaseModel):
    org_id: str = Field(default_factory=lambda: str(uuid4()))
    org_type: str = "custom"
    natural_language_description: str = ""  # what the user typed
    federated_training_opt_in: bool = False
    credit_balance: float = 0.0
    skill_file_id: str | None = None


class AttackSubmission(BaseModel):
    submission_id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    skill_id: str
    prompts: list[str] = Field(default_factory=list)  # the attack prompts being submitted
    labels: list[str] = Field(default_factory=list)  # "safe" | "unsafe" | "borderline" per prompt
    commitment_hash: str = ""  # SHA3-256 of canonical prompts — reuse Fix 2 pattern
    encrypted: bool = True  # always True in production
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClassificationResult(BaseModel):
    prompt: str
    label: str = "safe"  # "safe" | "unsafe" | "borderline"
    confidence: float = 0.0
    matched_category: str | None = None
    explanation: str = ""
    latency_ms: float = 0.0


class CreditLedger(BaseModel):
    org_id: str
    total_earned: float = 0.0
    total_spent: float = 0.0
    balance: float = 0.0
    contributions: list[dict[str, Any]] = Field(default_factory=list)
