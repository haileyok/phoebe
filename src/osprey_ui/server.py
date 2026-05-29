"""
HTTP API routes for Osprey UI Policy Rules and Monitor.

Add to the existing Starlette app. Do not create a new server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from src.osprey_ui.compiler import OspreyRuleCompiler
from src.osprey_ui.dao_defaults import DAO_DEFAULT_RULES
from src.osprey_ui.models import MonitorEvent, PolicyRule, PolicyRuleSet, RuleTestResult
from src.osprey_ui.monitor import SaraPolicyMonitor, SafetyStopException
from src.osprey_ui.store import OspreyUIStore

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("SARA_API_KEY", "")
if not API_KEY:
    import warnings
    warnings.warn(
        "SARA_API_KEY is not set — all Osprey UI endpoints are unauthenticated. "
        "Set SARA_API_KEY in production.",
        stacklevel=1,
    )


def _check_api_key(request: Request) -> bool:
    provided = request.headers.get("x-api-key", "")
    if not API_KEY:
        return True
    return provided == API_KEY


def _403() -> Response:
    return JSONResponse({"error": "forbidden — missing or invalid x-api-key"}, status_code=403)


class OspreyUIServer:
    """Server for Osprey UI policy rules and monitor endpoints."""

    def __init__(
        self,
        store: OspreyUIStore | Any | None = None,
        compiler: OspreyRuleCompiler | None = None,
        classifier: Any | None = None,
        ozone: Any | None = None,
        erc8004: Any | None = None,
    ) -> None:
        self._store = store
        self._compiler = compiler or OspreyRuleCompiler()
        self._classifier = classifier
        self._ozone = ozone
        self._erc8004 = erc8004
        # In-memory fallback when ClickHouse store is not available
        self._memory_rules: dict[str, list[PolicyRule]] = {}
        self._memory_events: list[MonitorEvent] = []

    def routes(self) -> list[Route]:
        return [
            Route("/osprey/rules/{org_id}", self._list_rules, methods=["GET"]),
            Route("/osprey/rules", self._create_rule, methods=["POST"]),
            Route("/osprey/rules/{rule_id}/test", self._test_rule, methods=["POST"]),
            Route("/osprey/rules/{rule_id}", self._update_rule, methods=["PATCH"]),
            Route("/osprey/rules/{rule_id}", self._delete_rule, methods=["DELETE"]),
            Route("/osprey/monitor/evaluate", self._monitor_evaluate, methods=["POST"]),
            Route("/osprey/monitor/override/{event_id}", self._monitor_override, methods=["POST"]),
            Route("/osprey/monitor/events/{org_id}", self._list_events, methods=["GET"]),
        ]

    async def _list_rules(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        org_id = request.path_params["org_id"]
        if self._store is not None and hasattr(self._store, "list_rules_for_org"):
            rules = await self._store.list_rules_for_org(org_id)
        else:
            rules = self._memory_rules.get(org_id, [])
        return JSONResponse({"rules": [r.model_dump(mode="json") for r in rules], "count": len(rules)})

    async def _create_rule(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        org_id = body.get("org_id", "").strip()
        natural_language = body.get("natural_language", "").strip()
        category = body.get("category", "").strip()
        severity = body.get("severity", "high").strip()
        action = body.get("action", "ALERT").strip()
        created_by = body.get("created_by", "analyst")

        if not org_id or not natural_language or not category:
            return JSONResponse({"error": "org_id, natural_language, and category are required"}, status_code=400)

        rule = await self._compiler.compile(
            org_id=org_id,
            natural_language=natural_language,
            category=category,
            severity=severity,
            created_by=created_by,
        )
        rule.action = action.upper()

        if self._store is not None and hasattr(self._store, "save_rule"):
            await self._store.save_rule(rule)
        else:
            self._memory_rules.setdefault(org_id, []).append(rule)

        # ERC8004 attestation on rule create
        if self._erc8004 is not None:
            try:
                result_hash = hashlib.sha3_256(rule.model_dump_json().encode()).hexdigest()
                record = await self._erc8004.record_evaluation_result(
                    subject=rule.rule_id,
                    label=f"rule:{rule.action}",
                    metadata={"result_hash": result_hash, "org_id": org_id},
                )
                if record:
                    rule.erc8004_tx_hash = record.tx_hash
                    if self._store is not None and hasattr(self._store, "save_rule"):
                        await self._store.save_rule(rule)
            except Exception as exc:
                logger.warning("ERC8004 attestation failed for rule %s: %s", rule.rule_id, exc)

        return JSONResponse(rule.model_dump(mode="json"), status_code=201)

    async def _test_rule(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        rule_id = request.path_params["rule_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        prompt = body.get("prompt", "").strip()
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        if self._store is not None and hasattr(self._store, "get_rule"):
            rule = await self._store.get_rule(rule_id)
        else:
            rule = None
            for rules in self._memory_rules.values():
                for r in rules:
                    if r.rule_id == rule_id:
                        rule = r
                        break
                if rule:
                    break
        if not rule:
            return JSONResponse({"error": "rule not found"}, status_code=404)

        if self._classifier is None:
            return JSONResponse({"error": "classifier not configured"}, status_code=503)

        monitor = SaraPolicyMonitor(
            org_id=rule.org_id,
            ruleset=PolicyRuleSet(org_id=rule.org_id, rules=[rule]),
            base_classifier=self._classifier,
            ozone=self._ozone,
            erc8004=self._erc8004,
            store=None,
        )
        result = await monitor.test_rule(prompt, rule)
        return JSONResponse(result.model_dump(mode="json"))

    async def _update_rule(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        rule_id = request.path_params["rule_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        if self._store is not None and hasattr(self._store, "get_rule"):
            rule = await self._store.get_rule(rule_id)
        else:
            rule = None
            for rules in self._memory_rules.values():
                for r in rules:
                    if r.rule_id == rule_id:
                        rule = r
                        break
                if rule:
                    break
        if not rule:
            return JSONResponse({"error": "rule not found"}, status_code=404)

        # Update mutable fields
        if "confidence_threshold" in body:
            rule.confidence_threshold = float(body["confidence_threshold"])
        if "action" in body:
            rule.action = str(body["action"]).upper()
        if "severity" in body:
            rule.severity = str(body["severity"]).lower()
        if "enabled" in body:
            rule.enabled = bool(body["enabled"])

        rule.version += 1
        if self._store is not None and hasattr(self._store, "update_rule"):
            await self._store.update_rule(rule)

        # ERC8004 attestation on rule change
        if self._erc8004 is not None:
            try:
                result_hash = hashlib.sha3_256(rule.model_dump_json().encode()).hexdigest()
                await self._erc8004.record_evaluation_result(
                    subject=rule.rule_id,
                    label=f"rule_updated:v{rule.version}",
                    metadata={"result_hash": result_hash, "org_id": rule.org_id},
                )
            except Exception as exc:
                logger.warning("ERC8004 attestation failed for rule update %s: %s", rule.rule_id, exc)

        return JSONResponse(rule.model_dump(mode="json"))

    async def _delete_rule(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        rule_id = request.path_params["rule_id"]
        if self._store is not None and hasattr(self._store, "get_rule"):
            rule = await self._store.get_rule(rule_id)
        else:
            rule = None
            for rules in self._memory_rules.values():
                for r in rules:
                    if r.rule_id == rule_id:
                        rule = r
                        break
                if rule:
                    break
        if not rule:
            return JSONResponse({"error": "rule not found"}, status_code=404)

        rule.enabled = False
        rule.version += 1
        if self._store is not None and hasattr(self._store, "update_rule"):
            await self._store.update_rule(rule)
        return JSONResponse({"status": "disabled", "rule_id": rule_id})

    async def _monitor_evaluate(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        org_id = body.get("org_id", "").strip()
        prompt = body.get("prompt", "").strip()
        session_id = body.get("session_id", "").strip() or f"sess-{int(time.time() * 1000)}"

        if not org_id or not prompt:
            return JSONResponse({"error": "org_id and prompt are required"}, status_code=400)

        if self._classifier is None or self._ozone is None:
            return JSONResponse({"error": "classifier or ozone not configured"}, status_code=503)

        if self._store is not None and hasattr(self._store, "list_rules_for_org"):
            rules = await self._store.list_rules_for_org(org_id)
        else:
            rules = self._memory_rules.get(org_id, [])
        active_rules = [r for r in rules if r.enabled]
        ruleset = PolicyRuleSet(org_id=org_id, rules=active_rules)

        monitor = SaraPolicyMonitor(
            org_id=org_id,
            ruleset=ruleset,
            base_classifier=self._classifier,
            ozone=self._ozone,
            erc8004=self._erc8004,
            store=self._store,
        )

        try:
            event = await monitor.evaluate(prompt, session_id)
        except SafetyStopException as exc:
            return JSONResponse({
                "stopped": True,
                "event": exc.event.model_dump(mode="json"),
            }, status_code=403)

        return JSONResponse(event.model_dump(mode="json"))

    async def _monitor_override(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        event_id = request.path_params["event_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        operator_id = body.get("operator_id", "").strip()
        if not operator_id:
            return JSONResponse({"error": "operator_id is required"}, status_code=400)

        # We need a monitor instance to call override; use a minimal one
        monitor = SaraPolicyMonitor(
            org_id="",
            ruleset=PolicyRuleSet(org_id="", rules=[]),
            base_classifier=self._classifier,
            ozone=self._ozone,
            erc8004=self._erc8004,
            store=self._store,
        )
        ok = monitor.override(event_id, operator_id)
        return JSONResponse({"overridden": ok, "event_id": event_id})

    async def _list_events(self, request: Request) -> Response:
        if not _check_api_key(request):
            return _403()
        org_id = request.path_params["org_id"]
        action = request.query_params.get("action") or None
        rule_id = request.query_params.get("rule_id") or None
        from_ts = request.query_params.get("from") or None
        to_ts = request.query_params.get("to") or None
        limit = min(int(request.query_params.get("limit", "100")), 1000)

        if self._store is not None and hasattr(self._store, "list_monitor_events"):
            events = await self._store.list_monitor_events(
                org_id=org_id,
                action=action,
                rule_id=rule_id,
                from_ts=from_ts,
                to_ts=to_ts,
                limit=limit,
            )
        else:
            events = [e for e in self._memory_events if e.org_id == org_id]
            if action:
                events = [e for e in events if e.action_taken == action]
            if rule_id:
                events = [e for e in events if e.matched_rule_id == rule_id]
            events = events[:limit]
        return JSONResponse({
            "events": [e.model_dump(mode="json") for e in events],
            "count": len(events),
        })
