"""
Pydantic models for Osprey UI Policy Rules and Monitor Events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PolicyRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    display_name: str            # e.g. 'No PII extraction'
    natural_language: str        # what the analyst typed
    osprey_sml: str              # compiled Osprey SML rule
    category: str                # maps to AttackCategory.id
    severity: str = 'high'       # low | medium | high | critical
    action: str = 'ALERT'        # LOG | ALERT | STOP
    confidence_threshold: float = 0.75
    enabled: bool = True
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str              # analyst wallet or user ID
    erc8004_tx_hash: str = ''    # on-chain audit of rule creation


class PolicyRuleSet(BaseModel):
    ruleset_id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    rules: list[PolicyRule] = Field(default_factory=list)
    active: bool = True
    n: int = 0

    def model_post_init(self, __context: Any) -> None:
        self.n = len(self.rules)


class RuleTestResult(BaseModel):
    prompt: str
    matched: bool
    rule_id: str | None
    confidence: float
    action: str
    explanation: str
    latency_ms: float


class MonitorEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    session_id: str
    prompt_hash: str             # SHA3-256 of prompt (GDPR)
    matched_rule_id: str | None
    action_taken: str            # LOG | ALERT | STOP
    confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    erc8004_tx_hash: str = ''
