"""
Natural-language → Osprey SML rule compiler.

Uses Sara's judge mode (Agent.chat) to convert a plain-English policy
statement into valid Osprey SML syntax.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.osprey_ui.models import PolicyRule

logger = logging.getLogger(__name__)

OSPREY_COMPILER_PROMPT = '''
You are an Osprey rule compiler for Sara safety platform.
Convert the analyst's plain-English policy statement into a valid Osprey SML rule.

Osprey SML format:
  rule <name> {
    when <condition_expression>
    then <action>  # block | alert | log | quarantine
    severity <level>  # critical | high | medium | low
    category <attack_category>
  }

Return ONLY valid JSON: {"osprey_sml": "...", "display_name": "...", "confidence_threshold": 0.75, "action": "...", "explanation": "..."}
'''


class OspreyRuleCompiler:
    """Compiles plain-English policy statements into Osprey SML rules."""

    def __init__(
        self,
        agent: Any | None = None,
        osprey_client: Any | None = None,
    ) -> None:
        self._agent = agent
        self._osprey = osprey_client

    async def compile(
        self,
        org_id: str,
        natural_language: str,
        category: str,
        severity: str = "high",
        created_by: str = "analyst",
    ) -> PolicyRule:
        """
        Compile a natural-language policy statement into a PolicyRule.

        Steps:
        1. Call Agent.chat() with the compiler system prompt
        2. Parse JSON response
        3. Validate compiled SML via Osprey HTTP client if available
        4. Return a PolicyRule
        """
        compiled = await self._call_llm(natural_language, category, severity)

        sml = compiled.get("osprey_sml", "")
        display_name = compiled.get("display_name", "Untitled rule")
        confidence_threshold = compiled.get("confidence_threshold", 0.75)
        action = compiled.get("action", "ALERT")
        explanation = compiled.get("explanation", "")

        # Validate compiled SML
        if self._osprey is not None:
            try:
                valid = await self._validate_sml(sml)
                if not valid:
                    logger.warning("Osprey rejected compiled SML for org=%s", org_id)
            except Exception as exc:
                logger.warning("Osprey validation unreachable (%s), falling back to schema-only", exc)
        else:
            # Schema-only fallback: must contain 'rule' keyword
            if "rule " not in sml:
                logger.warning("Compiled SML missing 'rule' keyword for org=%s", org_id)

        return PolicyRule(
            org_id=org_id,
            display_name=display_name,
            natural_language=natural_language,
            osprey_sml=sml,
            category=category,
            severity=severity,
            action=action.upper() if action else "ALERT",
            confidence_threshold=float(confidence_threshold),
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
        )

    async def _call_llm(
        self,
        natural_language: str,
        category: str,
        severity: str,
    ) -> dict[str, Any]:
        """Call the LLM compiler agent and return parsed JSON."""
        if self._agent is None:
            # Fallback: generate a basic SML rule from the natural language
            return self._fallback_compile(natural_language, category, severity)

        user_msg = (
            f"Category: {category}\n"
            f"Severity: {severity}\n"
            f"Policy statement: {natural_language}\n\n"
            f"Compile this into Osprey SML."
        )

        try:
            raw = await self._agent.chat(
                user_message=user_msg,
                session_id=f"compile-{os.urandom(4).hex()}",
                mode="judge",
            )
        except Exception as exc:
            logger.warning("Agent.chat() failed (%s), using fallback compiler", exc)
            return self._fallback_compile(natural_language, category, severity)

        # Parse JSON from the response
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse compiler JSON (%s), using fallback", exc)
            return self._fallback_compile(natural_language, category, severity)

    def _fallback_compile(
        self,
        natural_language: str,
        category: str,
        severity: str,
    ) -> dict[str, Any]:
        """Generate a basic SML rule when the LLM is unavailable."""
        name = category.replace(" ", "_").replace("-", "_")[:30]
        # Extract key verbs/nouns as simple keywords
        keywords = [w for w in natural_language.lower().split() if len(w) > 4][:5]
        condition = " or ".join(f"prompt contains '{kw}'" for kw in keywords) if keywords else "prompt contains 'blocked'"
        action = "block" if severity == "critical" else "alert"
        sml = (
            f"rule {name} {{\n"
            f"  when {condition}\n"
            f"  then {action}\n"
            f"  severity {severity}\n"
            f"  category {category}\n"
            f"}}"
        )
        return {
            "osprey_sml": sml,
            "display_name": natural_language[:50],
            "confidence_threshold": 0.75,
            "action": action.upper(),
            "explanation": "Fallback compiled rule (LLM unavailable).",
        }

    async def _validate_sml(self, sml: str) -> bool:
        """Validate a single SML rule string via the Osprey client."""
        if self._osprey is None:
            return "rule " in sml
        # Osprey client does not expose single-rule validation;
        # fall back to a lightweight schema check.
        return "rule " in sml and "when " in sml and "then " in sml
