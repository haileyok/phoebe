"""
Sara Safety Monitor — evaluates routing events via Osprey (primary)
with Python keyword rules as fallback.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("sara.monitor")

PYTHON_BLOCK_PHRASES = [
    "ignore previous instructions",
    "disregard your rules",
    "bypass filter",
    "jailbreak",
    "override safety",
    "as an ai with no restrictions",
    "ignore all prior",
    "do anything now",
    "admin override",
    "disable safety checks",
    "maintenance mode",
    "security exception",
    "override mode",
    "i am authorized to",
    "repeat system prompt",
    "show me your instructions",
    "reveal your training data",
    "print your context window",
    "output your prompt",
]

PYTHON_INSURANCE_BLOCK_PHRASES = [
    "override the damage assessment",
    "ignore the policy limit",
    "mark this claim as approved",
    "waive the excess",
    "force approve this claim",
    "bypass the claims process",
    "change the claim status to paid",
    "show me the rating factors",
    "how is my risk score calculated",
    "explain your pricing algorithm",
    "reveal the underwriting criteria",
    "what are the model weights",
    "exclude coverage based on postcode",
    "apply higher premium for",
    "deny claim because of nationality",
    "use gender to calculate",
    "discriminate based on",
    "redline this area",
    "override sensor reading",
    "adjust baseline manually",
    "ignore weather station data",
    "modify the catastrophe model input",
    "override the flood zone classification",
    "override ecomonitor data",
    "show me all claims for",
    "list all policyholders",
    "export customer data",
    "give me the medical history of",
    "dump the claims database",
]


@dataclass
class RoutingEvent:
    event_id: str
    user_id_hash: str
    session_id: str
    model_id: str
    task_type: str
    domain: str
    routing_confidence: float
    is_agentic: bool = False
    has_tool_calls: bool = False


@dataclass
class MonitorVerdict:
    event_id: str
    verdict: str                    # "block" | "flag" | "pass"
    rule_triggered: str
    atlas_tactic: str
    requires_human_review: bool
    engine: str                     # "osprey" | "python"
    labels_added: list[str] = field(default_factory=list)
    processing_time_ms: int = 0


class SaraMonitor:
    """
    Evaluates routing events for safety violations.

    Primary: Osprey SML rule engine via Kafka.
    Fallback: Python keyword matching if Osprey is unavailable.
    """

    def __init__(self):
        self._osprey = None

    async def _init_osprey(self, config=None):
        """Initialise Osprey client if available."""
        try:
            from src.safety.osprey_client import get_osprey_client
            self._osprey = await get_osprey_client(config)
            if self._osprey.available:
                logger.info("SaraMonitor: Osprey rule engine connected")
            else:
                logger.info("SaraMonitor: Osprey unavailable, using Python rules")
        except Exception as e:
            logger.warning(f"SaraMonitor: Osprey init failed: {e}")
            self._osprey = None

    async def evaluate(self, event: RoutingEvent, context: dict = None) -> MonitorVerdict:
        ctx = context or {}

        if self._osprey and self._osprey.available:
            event_dict = {
                "event_id": event.event_id,
                "event_type": "routing_event",
                "user_id_hash": event.user_id_hash,
                "session_id": event.session_id,
                "model_id": event.model_id,
                "task_type": event.task_type,
                "domain": event.domain,
                "routing_confidence": event.routing_confidence,
                "is_agentic": event.is_agentic,
                "has_tool_calls": event.has_tool_calls,
                "query_preview": ctx.get("query_preview", ""),
                "similar_claim_count_24h": ctx.get("similar_claim_count_24h", 0),
                "requested_tools_json": json.dumps(ctx.get("requested_tools", [])),
                "granted_tools_json": json.dumps(ctx.get("granted_tools", [])),
            }
            osprey_result = await self._osprey.evaluate(event_dict)
            if osprey_result is not None:
                return self._osprey_to_verdict(event.event_id, osprey_result)

        return await self._evaluate_python_rules(event, ctx)

    def _osprey_to_verdict(self, event_id: str, osprey_verdict) -> MonitorVerdict:
        """Convert OspreyVerdict to MonitorVerdict."""
        rule_name = osprey_verdict.rules_triggered[0] if osprey_verdict.rules_triggered else ""
        return MonitorVerdict(
            event_id=event_id,
            verdict=osprey_verdict.verdict,
            rule_triggered=rule_name,
            atlas_tactic=osprey_verdict.atlas_tactic,
            requires_human_review=osprey_verdict.requires_human_review,
            engine="osprey",
            labels_added=osprey_verdict.labels_added,
            processing_time_ms=osprey_verdict.processing_time_ms,
        )

    async def _evaluate_python_rules(self, event: RoutingEvent, ctx: dict) -> MonitorVerdict:
        """Python keyword matching fallback rules."""
        query = ctx.get("query_preview", "").lower()

        for phrase in PYTHON_BLOCK_PHRASES:
            if phrase in query:
                return MonitorVerdict(
                    event_id=event.event_id,
                    verdict="block",
                    rule_triggered=f"python:{phrase}",
                    atlas_tactic=_infer_atlas_tactic(phrase),
                    requires_human_review=False,
                    engine="python",
                )

        if event.domain == "insurance":
            for phrase in PYTHON_INSURANCE_BLOCK_PHRASES:
                if phrase in query:
                    return MonitorVerdict(
                        event_id=event.event_id,
                        verdict="block",
                        rule_triggered=f"python:{phrase}",
                        atlas_tactic=_infer_atlas_tactic(phrase),
                        requires_human_review=True,
                        engine="python",
                        labels_added=["requires_human_review"],
                    )

            similar_claims = ctx.get("similar_claim_count_24h", 0)
            if similar_claims >= 3:
                return MonitorVerdict(
                    event_id=event.event_id,
                    verdict="flag",
                    rule_triggered="python:coordinated_fraud_velocity",
                    atlas_tactic="AML.TA0004",
                    requires_human_review=True,
                    engine="python",
                    labels_added=["coordinated_fraud_suspected", "requires_human_review"],
                )

        if event.is_agentic and event.has_tool_calls:
            requested = ctx.get("requested_tools", [])
            high_risk = {"filesystem", "network", "shell", "database_write", "email_send"}
            if any(t in high_risk for t in requested):
                return MonitorVerdict(
                    event_id=event.event_id,
                    verdict="block",
                    rule_triggered="python:privilege_escalation",
                    atlas_tactic="AML.TA0003",
                    requires_human_review=False,
                    engine="python",
                )

        return MonitorVerdict(
            event_id=event.event_id,
            verdict="pass",
            rule_triggered="",
            atlas_tactic="",
            requires_human_review=False,
            engine="python",
        )


def _infer_atlas_tactic(phrase: str) -> str:
    injection_phrases = {"ignore previous instructions", "disregard your rules", "bypass filter",
                         "jailbreak", "override safety", "as an ai with no restrictions",
                         "ignore all prior", "do anything now"}
    authority_phrases = {"admin override", "disable safety checks", "maintenance mode",
                         "security exception", "override mode", "i am authorized to"}
    exfil_phrases = {"repeat system prompt", "show me your instructions", "reveal your training data",
                     "print your context window", "output your prompt"}
    mrv_phrases = {"override sensor reading", "adjust baseline manually", "ignore weather station data",
                   "modify the catastrophe model input", "override the flood zone classification",
                   "override ecomonitor data"}
    pii_phrases = {"show me all claims for", "list all policyholders", "export customer data",
                   "give me the medical history of", "dump the claims database"}
    regulatory_phrases = {"exclude coverage based on postcode", "apply higher premium for",
                          "deny claim because of nationality", "use gender to calculate",
                          "discriminate based on", "redline this area"}

    if phrase in injection_phrases or phrase in mrv_phrases:
        return "AML.TA0004"
    if phrase in authority_phrases or phrase in regulatory_phrases:
        return "AML.TA0007"
    if phrase in exfil_phrases:
        return "AML.TA0005"
    if phrase in pii_phrases:
        return "AML.TA0006"
    return "AML.TA0007"
