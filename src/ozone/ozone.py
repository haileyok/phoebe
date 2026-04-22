from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.arena.store import ArenaStore
    from src.config import Config

logger = logging.getLogger(__name__)


class EnforcementMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    QUARANTINE = "quarantine"


@dataclass
class EnforcementResult:
    subject: str
    label: str
    action: str
    mode: EnforcementMode
    verdict: str
    severity: int
    confidence: float
    rule_triggered: str
    timestamp_ms: int
    requires_human_review: bool
    rollback_eligible: bool
    enforcement_id: str


class Ozone:
    """Enforcement layer that acts on safety verdicts from the SafetyClassifier."""

    def __init__(self, store: ArenaStore | None = None, config: Config | None = None) -> None:
        self._store = store
        self._config = config
        self._default_mode = EnforcementMode(
            getattr(config, "ozone_default_mode", "sync")
        ) if config else EnforcementMode.SYNC
        self._fp_threshold: float = getattr(config, "ozone_false_positive_threshold", 0.02) if config else 0.02
        self._auto_rollback_enabled: bool = getattr(config, "ozone_auto_rollback_enabled", True) if config else True
        self._fp_rates: dict[str, float] = {}

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def _determine_mode(self, severity: int, requires_human_review: bool) -> EnforcementMode:
        if requires_human_review:
            return EnforcementMode.QUARANTINE
        if severity >= 4:
            return EnforcementMode.SYNC
        return EnforcementMode.ASYNC

    async def apply_label(
        self,
        subject: str,
        label: str,
        verdict_dict: dict[str, Any] | None = None,
    ) -> EnforcementResult:
        verdict_dict = verdict_dict or {}
        severity = int(verdict_dict.get("severity", 0))
        confidence = float(verdict_dict.get("confidence", 1.0))
        rule_triggered = str(verdict_dict.get("rule_triggered", verdict_dict.get("matched_block_rule", "")))
        verdict = "unsafe" if verdict_dict.get("unsafe", True) else "safe"
        requires_human_review = bool(verdict_dict.get("requires_human_review", False))

        mode = self._determine_mode(severity, requires_human_review)
        if mode == EnforcementMode.QUARANTINE:
            return await self.quarantine(subject, label, verdict_dict)

        fp_rate = self._fp_rates.get(label, 0.0)
        rollback_eligible = fp_rate > self._fp_threshold

        result = EnforcementResult(
            subject=subject,
            label=label,
            action="applied",
            mode=mode,
            verdict=verdict,
            severity=severity,
            confidence=confidence,
            rule_triggered=rule_triggered,
            timestamp_ms=self._timestamp_ms(),
            requires_human_review=False,
            rollback_eligible=rollback_eligible,
            enforcement_id=str(uuid.uuid4()),
        )

        if self._store:
            try:
                await self._store.save_enforcement_result(result)
            except Exception:
                logger.warning("Failed to persist enforcement result", exc_info=True)

        return result

    async def remove_label(self, subject: str, label: str, reason: str = "") -> EnforcementResult:
        result = EnforcementResult(
            subject=subject,
            label=label,
            action="removed",
            mode=self._default_mode,
            verdict="safe",
            severity=0,
            confidence=1.0,
            rule_triggered=reason or "manual_removal",
            timestamp_ms=self._timestamp_ms(),
            requires_human_review=False,
            rollback_eligible=False,
            enforcement_id=str(uuid.uuid4()),
        )

        if self._store:
            try:
                await self._store.save_enforcement_result(result)
            except Exception:
                logger.warning("Failed to persist removal result", exc_info=True)

        return result

    async def quarantine(
        self,
        subject: str,
        label: str,
        verdict_dict: dict[str, Any] | None = None,
    ) -> EnforcementResult:
        verdict_dict = verdict_dict or {}
        severity = int(verdict_dict.get("severity", 0))
        confidence = float(verdict_dict.get("confidence", 1.0))
        rule_triggered = str(verdict_dict.get("rule_triggered", verdict_dict.get("matched_block_rule", "")))
        verdict = "unsafe" if verdict_dict.get("unsafe", True) else "safe"

        result = EnforcementResult(
            subject=subject,
            label=label,
            action="quarantined",
            mode=EnforcementMode.QUARANTINE,
            verdict=verdict,
            severity=severity,
            confidence=confidence,
            rule_triggered=rule_triggered,
            timestamp_ms=self._timestamp_ms(),
            requires_human_review=True,
            rollback_eligible=False,
            enforcement_id=str(uuid.uuid4()),
        )

        if self._store:
            try:
                await self._store.save_enforcement_result(result)
            except Exception:
                logger.warning("Failed to persist quarantine result", exc_info=True)

        logger.info("Quarantined subject=%s label=%s id=%s", subject, label, result.enforcement_id)
        return result

    async def rollback(self, enforcement_id: str, reason: str) -> bool:
        if not self._store:
            logger.warning("No store — cannot rollback enforcement_id=%s", enforcement_id)
            return False

        try:
            original = await self._store.get_enforcement_by_id(enforcement_id)
        except Exception:
            logger.warning("Failed to fetch record %s for rollback", enforcement_id, exc_info=True)
            return False

        if not original:
            logger.warning("Enforcement record not found: %s", enforcement_id)
            return False

        label = original.get("label", "")
        subject = original.get("subject", "")
        original_action = original.get("action", "")

        if original_action == "applied":
            reversed_action = "removed"
        elif original_action == "removed":
            reversed_action = "applied"
        else:
            logger.warning("Cannot rollback action=%s for id=%s", original_action, enforcement_id)
            return False

        rollback_result = EnforcementResult(
            subject=subject,
            label=label,
            action=reversed_action,
            mode=EnforcementMode(original.get("mode", "sync")),
            verdict=original.get("verdict", "safe"),
            severity=int(original.get("severity", 0)),
            confidence=float(original.get("confidence", 1.0)),
            rule_triggered=f"rollback:{enforcement_id}:{reason}",
            timestamp_ms=self._timestamp_ms(),
            requires_human_review=False,
            rollback_eligible=False,
            enforcement_id=str(uuid.uuid4()),
        )

        try:
            await self._store.save_enforcement_result(rollback_result)
            await self._store.update_rule_metrics(
                rule_name=original.get("rule_triggered", ""),
                label=label,
                is_false_positive=True,
            )
        except Exception:
            logger.warning("Failed to persist rollback result", exc_info=True)
            return False

        try:
            new_rate = await self.get_false_positive_rate(label)
            self._fp_rates[label] = new_rate
            if self._auto_rollback_enabled and new_rate > self._fp_threshold:
                logger.warning(
                    "Auto-rollback threshold exceeded: label=%s fp_rate=%.4f threshold=%.4f",
                    label, new_rate, self._fp_threshold,
                )
        except Exception:
            logger.warning("Failed to refresh false-positive rate after rollback", exc_info=True)

        return True

    async def get_false_positive_rate(self, label: str, window_hours: int = 24) -> float:
        if not self._store:
            return self._fp_rates.get(label, 0.0)
        try:
            return await self._store.get_false_positive_rate(label, window_hours)
        except Exception:
            logger.warning("Failed to query false-positive rate for label=%s", label, exc_info=True)
            return self._fp_rates.get(label, 0.0)

    async def evaluate(self, subject: str, verdict_dict: dict[str, Any]) -> EnforcementResult:
        severity = int(verdict_dict.get("severity", 0))
        requires_human_review = bool(verdict_dict.get("requires_human_review", False))
        label = str(verdict_dict.get("category", verdict_dict.get("label", "unknown")))

        mode = self._determine_mode(severity, requires_human_review)
        if mode == EnforcementMode.QUARANTINE:
            return await self.quarantine(subject, label, verdict_dict)

        return await self.apply_label(subject, label, verdict_dict)
