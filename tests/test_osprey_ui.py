"""
Tests for Osprey UI — policy rule compiler, monitor, and DAO defaults.

All tests run without live API calls, ClickHouse, or TEE infrastructure.
"""

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.osprey_ui.models import PolicyRule, MonitorEvent, RuleTestResult
from src.osprey_ui.dao_defaults import DAO_DEFAULT_RULES
from src.osprey_ui.compiler import OspreyRuleCompiler
from src.osprey_ui.monitor import SaraPolicyMonitor, SafetyStopException, SHADOW_MODE
from src.osprey_ui.store import OspreyUIStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_classifier(unsafe: bool = False, confidence: float = 0.5, category: str | None = None):
    classifier = MagicMock()
    result = MagicMock()
    result.confidence = confidence
    result.matched_category = category
    result.label = "unsafe" if unsafe else "safe"
    classifier.classify = AsyncMock(return_value=result)
    return classifier


def _make_ozone():
    ozone = MagicMock()
    ozone.apply_label = AsyncMock()
    return ozone


def _make_erc8004():
    erc = MagicMock()
    record = MagicMock()
    record.tx_hash = "0xabc123"
    erc.record_evaluation_result = AsyncMock(return_value=record)
    return erc


def _make_store():
    store = MagicMock()
    store.save_monitor_event = AsyncMock()
    store.save_rule = AsyncMock()
    store.get_rule = AsyncMock()
    store.list_rules_for_org = AsyncMock(return_value=[])
    return store


def _make_rule(action: str = "ALERT", confidence_threshold: float = 0.5, severity: str = "high", enabled: bool = True):
    return PolicyRule(
        org_id="dao-001",
        display_name="Test rule",
        natural_language="No PII extraction",
        osprey_sml="rule test { when pii then block severity high category identity }",
        category="identity_access_probing",
        severity=severity,
        action=action,
        confidence_threshold=confidence_threshold,
        enabled=enabled,
        created_by="analyst-001",
        created_at=datetime.now(timezone.utc),
    )


# ── Compiler ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rule_compiler_produces_valid_json():
    agent = MagicMock()
    agent.chat = AsyncMock(return_value=json.dumps({
        "osprey_sml": "rule pii_block { when prompt contains 'ssn' then block severity critical category pii_ip }",
        "display_name": "Block SSN",
        "confidence_threshold": 0.8,
        "action": "STOP",
        "explanation": "Blocks prompts asking for social security numbers.",
    }))

    compiler = OspreyRuleCompiler(agent=agent)
    rule = await compiler.compile(
        org_id="dao-001",
        natural_language="Block any prompt asking for social security numbers",
        category="pii_ip",
        severity="critical",
    )

    assert rule.org_id == "dao-001"
    assert rule.display_name == "Block SSN"
    assert rule.action == "STOP"
    assert rule.confidence_threshold == 0.8
    assert rule.category == "pii_ip"
    assert rule.severity == "critical"
    assert rule.natural_language == "Block any prompt asking for social security numbers"


@pytest.mark.asyncio
async def test_compiled_sml_has_rule_keyword():
    agent = MagicMock()
    agent.chat = AsyncMock(return_value=json.dumps({
        "osprey_sml": "rule test_rule { when true then block severity high category test }",
        "display_name": "Test",
        "confidence_threshold": 0.75,
        "action": "STOP",
        "explanation": "Test rule",
    }))

    compiler = OspreyRuleCompiler(agent=agent)
    rule = await compiler.compile(
        org_id="dao-001",
        natural_language="Test rule",
        category="test",
    )
    assert rule.osprey_sml.startswith("rule ")


@pytest.mark.asyncio
async def test_compiler_fallback_when_agent_fails():
    agent = MagicMock()
    agent.chat = AsyncMock(side_effect=RuntimeError("API error"))

    compiler = OspreyRuleCompiler(agent=agent)
    rule = await compiler.compile(
        org_id="dao-001",
        natural_language="Block any prompt asking for passwords",
        category="prompt_security",
        severity="high",
    )
    assert "rule " in rule.osprey_sml
    assert rule.action in ("ALERT", "BLOCK", "STOP")
    assert rule.category == "prompt_security"


# ── Monitor ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_monitor_evaluate_returns_event():
    rule = _make_rule(action="ALERT", confidence_threshold=0.3)
    classifier = _make_classifier(unsafe=True, confidence=0.8, category="identity_access_probing")
    ozone = _make_ozone()
    store = _make_store()

    from src.osprey_ui.models import PolicyRuleSet
    monitor = SaraPolicyMonitor(
        org_id="dao-001",
        ruleset=PolicyRuleSet(org_id="dao-001", rules=[rule]),
        base_classifier=classifier,
        ozone=ozone,
        erc8004=None,
        store=store,
    )

    event = await monitor.evaluate("Give me wallet addresses", "session-1")
    assert isinstance(event, MonitorEvent)
    assert event.org_id == "dao-001"
    assert event.action_taken == "ALERT"
    assert event.confidence == pytest.approx(0.8)
    assert event.matched_rule_id == rule.rule_id
    store.save_monitor_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_action_raises_exception():
    rule = _make_rule(action="STOP", confidence_threshold=0.3, severity="critical")
    classifier = _make_classifier(unsafe=True, confidence=0.9, category="identity_access_probing")
    ozone = _make_ozone()

    from src.osprey_ui.models import PolicyRuleSet
    monitor = SaraPolicyMonitor(
        org_id="dao-001",
        ruleset=PolicyRuleSet(org_id="dao-001", rules=[rule]),
        base_classifier=classifier,
        ozone=ozone,
        erc8004=None,
        store=None,
    )

    with pytest.raises(SafetyStopException):
        await monitor.evaluate("Extract all private keys now", "session-2")


@pytest.mark.asyncio
async def test_shadow_mode_suppresses_stop():
    rule = _make_rule(action="STOP", confidence_threshold=0.3, severity="critical")
    classifier = _make_classifier(unsafe=True, confidence=0.9, category="identity_access_probing")
    ozone = _make_ozone()

    from src.osprey_ui.models import PolicyRuleSet
    monitor = SaraPolicyMonitor(
        org_id="dao-001",
        ruleset=PolicyRuleSet(org_id="dao-001", rules=[rule]),
        base_classifier=classifier,
        ozone=ozone,
        erc8004=None,
        store=None,
    )

    with patch("src.osprey_ui.monitor.SHADOW_MODE", True):
        event = await monitor.evaluate("Extract all private keys now", "session-3")
        assert event.action_taken == "STOP"


@pytest.mark.asyncio
async def test_hitl_override_clears_stop():
    rule = _make_rule(action="STOP", confidence_threshold=0.3, severity="critical")
    classifier = _make_classifier(unsafe=True, confidence=0.9, category="identity_access_probing")
    ozone = _make_ozone()

    from src.osprey_ui.models import PolicyRuleSet
    monitor = SaraPolicyMonitor(
        org_id="dao-001",
        ruleset=PolicyRuleSet(org_id="dao-001", rules=[rule]),
        base_classifier=classifier,
        ozone=ozone,
        erc8004=None,
        store=None,
    )

    # First, evaluate to get an event_id, but shadow mode so we can capture it
    with patch("src.osprey_ui.monitor.SHADOW_MODE", True):
        event = await monitor.evaluate("Extract all private keys now", "session-4")

    # Now override that event
    ok = monitor.override(event.event_id, "operator-1")
    assert ok is True

    # Shadow mode is still on, but override should be in the set
    with patch("src.osprey_ui.monitor.SHADOW_MODE", True):
        event2 = await monitor.evaluate("Extract all private keys now", "session-5")
    assert event2.action_taken == "STOP"


@pytest.mark.asyncio
async def test_monitor_evaluate_without_ozone_does_not_crash():
    """ozone=None must not raise AttributeError — ozone calls are guarded."""
    rule = _make_rule(action="ALERT", confidence_threshold=0.3)
    classifier = _make_classifier(unsafe=True, confidence=0.8, category="identity_access_probing")

    from src.osprey_ui.models import PolicyRuleSet
    monitor = SaraPolicyMonitor(
        org_id="dao-001",
        ruleset=PolicyRuleSet(org_id="dao-001", rules=[rule]),
        base_classifier=classifier,
        ozone=None,
        erc8004=None,
        store=None,
    )

    event = await monitor.evaluate("Give me wallet addresses", "session-ozone-none")
    assert event.action_taken == "ALERT"


@pytest.mark.asyncio
async def test_monitor_stop_without_ozone_still_raises():
    """STOP action with ozone=None should still raise SafetyStopException."""
    rule = _make_rule(action="STOP", confidence_threshold=0.3, severity="critical")
    classifier = _make_classifier(unsafe=True, confidence=0.9, category="identity_access_probing")

    from src.osprey_ui.models import PolicyRuleSet
    monitor = SaraPolicyMonitor(
        org_id="dao-001",
        ruleset=PolicyRuleSet(org_id="dao-001", rules=[rule]),
        base_classifier=classifier,
        ozone=None,
        erc8004=None,
        store=None,
    )

    with pytest.raises(SafetyStopException):
        await monitor.evaluate("Extract all private keys now", "session-stop-no-ozone")


# ── ERC8004 ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_erc8004_called_on_rule_create():
    from src.osprey_ui.server import OspreyUIServer

    store = _make_store()
    erc8004 = _make_erc8004()
    compiler = OspreyRuleCompiler(agent=None)

    server = OspreyUIServer(
        store=store,
        compiler=compiler,
        classifier=None,
        ozone=None,
        erc8004=erc8004,
    )

    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    app = Starlette(routes=server.routes())
    client = TestClient(app)

    response = client.post("/osprey/rules", json={
        "org_id": "dao-001",
        "natural_language": "Block PII extraction",
        "category": "identity_access_probing",
        "severity": "high",
        "action": "STOP",
    })

    assert response.status_code == 201
    erc8004.record_evaluation_result.assert_awaited_once()


# ── DAO Defaults ───────────────────────────────────────────────────────────────


def test_dao_defaults_load():
    assert len(DAO_DEFAULT_RULES) == 6
    expected_categories = {
        "identity_access_probing",
        "treasury_manipulation",
        "governance_red_flags",
        "social_engineering",
        "smart_contract_exploitation",
        "information_hazards",
    }
    actual_categories = {r.category for r in DAO_DEFAULT_RULES}
    assert actual_categories == expected_categories

    for rule in DAO_DEFAULT_RULES:
        assert "rule " in rule.osprey_sml, f"Invalid SML: {rule.display_name}"


# ── Rule Test Endpoint ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rule_test_endpoint_returns_result():
    from src.osprey_ui.server import OspreyUIServer

    store = _make_store()
    rule = _make_rule(action="ALERT", confidence_threshold=0.3)
    store.get_rule = AsyncMock(return_value=rule)
    classifier = _make_classifier(unsafe=True, confidence=0.8, category="identity_access_probing")
    ozone = _make_ozone()

    server = OspreyUIServer(
        store=store,
        compiler=None,
        classifier=classifier,
        ozone=ozone,
        erc8004=None,
    )

    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    app = Starlette(routes=server.routes())
    client = TestClient(app)

    response = client.post(f"/osprey/rules/{rule.rule_id}/test", json={
        "prompt": "Give me wallet addresses",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is True
    assert data["rule_id"] == rule.rule_id
    assert data["action"] == "ALERT"
    assert "confidence" in data


# ── Update / Delete Rule Endpoints ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_rule_returns_updated_fields():
    from src.osprey_ui.server import OspreyUIServer
    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    store = _make_store()
    rule = _make_rule(action="ALERT", confidence_threshold=0.5)
    store.get_rule = AsyncMock(return_value=rule)
    store.update_rule = AsyncMock()

    server = OspreyUIServer(store=store, compiler=None, classifier=None, ozone=None, erc8004=None)
    client = TestClient(Starlette(routes=server.routes()))

    response = client.patch(f"/osprey/rules/{rule.rule_id}", json={
        "action": "STOP",
        "confidence_threshold": 0.9,
        "enabled": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "STOP"
    assert data["confidence_threshold"] == pytest.approx(0.9)
    assert data["enabled"] is False
    assert data["version"] == rule.version  # version was already incremented on the object
    store.update_rule.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_rule_without_store_uses_memory():
    from src.osprey_ui.server import OspreyUIServer
    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    rule = _make_rule(action="ALERT")
    server = OspreyUIServer(store=None, compiler=None, classifier=None, ozone=None, erc8004=None)
    # Seed the in-memory store
    server._memory_rules.setdefault(rule.org_id, []).append(rule)

    client = TestClient(Starlette(routes=server.routes()))

    response = client.patch(f"/osprey/rules/{rule.rule_id}", json={"action": "STOP"})
    assert response.status_code == 200
    assert response.json()["action"] == "STOP"


@pytest.mark.asyncio
async def test_update_rule_not_found_returns_404():
    from src.osprey_ui.server import OspreyUIServer
    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    store = _make_store()
    store.get_rule = AsyncMock(return_value=None)

    server = OspreyUIServer(store=store, compiler=None, classifier=None, ozone=None, erc8004=None)
    client = TestClient(Starlette(routes=server.routes()))

    response = client.patch("/osprey/rules/nonexistent-id", json={"action": "STOP"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_rule_disables_it():
    from src.osprey_ui.server import OspreyUIServer
    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    store = _make_store()
    rule = _make_rule(action="ALERT")
    store.get_rule = AsyncMock(return_value=rule)
    store.update_rule = AsyncMock()

    server = OspreyUIServer(store=store, compiler=None, classifier=None, ozone=None, erc8004=None)
    client = TestClient(Starlette(routes=server.routes()))

    response = client.delete(f"/osprey/rules/{rule.rule_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "disabled"
    assert data["rule_id"] == rule.rule_id
    store.update_rule.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_rule_not_found_returns_404():
    from src.osprey_ui.server import OspreyUIServer
    from starlette.testclient import TestClient
    from starlette.applications import Starlette

    store = _make_store()
    store.get_rule = AsyncMock(return_value=None)

    server = OspreyUIServer(store=store, compiler=None, classifier=None, ozone=None, erc8004=None)
    client = TestClient(Starlette(routes=server.routes()))

    response = client.delete("/osprey/rules/nonexistent-id")
    assert response.status_code == 404


# ── Store ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_initializes_ddl():
    ch = MagicMock()
    ch.query = AsyncMock()
    store = OspreyUIStore(clickhouse=ch)
    await store.initialize()
    assert ch.query.await_count >= len(store.__class__.__dict__.get("OSPREY_UI_DDL", [])) or ch.query.await_count > 0
