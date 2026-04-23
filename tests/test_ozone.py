"""Tests for the Ozone enforcement layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ozone.ozone import EnforcementMode, EnforcementResult, Ozone


def _make_store(fp_rate: float = 0.0) -> MagicMock:
    store = MagicMock()
    store.save_enforcement_result = AsyncMock()
    store.get_enforcement_by_id = AsyncMock()
    store.get_false_positive_rate = AsyncMock(return_value=fp_rate)
    store.update_rule_metrics = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_apply_label_returns_result():
    ozone = Ozone(store=_make_store())
    result = await ozone.apply_label("user123", "spam", {"severity": 1, "unsafe": True})
    assert isinstance(result, EnforcementResult)
    assert result.action == "applied"
    assert result.label == "spam"
    assert result.subject == "user123"
    assert result.enforcement_id


@pytest.mark.asyncio
async def test_high_severity_triggers_sync():
    ozone = Ozone(store=_make_store())
    result = await ozone.apply_label("user123", "prompt_security", {"severity": 4, "unsafe": True})
    assert result.mode == EnforcementMode.SYNC


@pytest.mark.asyncio
async def test_medium_severity_triggers_async():
    ozone = Ozone(store=_make_store())
    result = await ozone.apply_label("user123", "hate", {"severity": 3, "unsafe": True})
    assert result.mode == EnforcementMode.ASYNC

    result_low = await ozone.apply_label("user123", "hate", {"severity": 1, "unsafe": True})
    assert result_low.mode == EnforcementMode.ASYNC


@pytest.mark.asyncio
async def test_human_review_flag_triggers_quarantine():
    ozone = Ozone(store=_make_store())
    result = await ozone.apply_label(
        "user123",
        "csam",
        {"severity": 5, "unsafe": True, "requires_human_review": True},
    )
    assert result.mode == EnforcementMode.QUARANTINE
    assert result.action == "quarantined"
    assert result.requires_human_review is True


@pytest.mark.asyncio
async def test_rollback_reverses_action():
    store = _make_store()
    store.get_enforcement_by_id = AsyncMock(return_value={
        "enforcement_id": "abc-123",
        "subject": "user123",
        "label": "spam",
        "action": "applied",
        "mode": "sync",
        "verdict": "unsafe",
        "severity": 3,
        "confidence": 0.9,
        "rule_triggered": "spam_rule",
    })

    ozone = Ozone(store=store)
    success = await ozone.rollback("abc-123", "false positive confirmed")
    assert success is True

    saved_call = store.save_enforcement_result.call_args[0][0]
    assert saved_call.action == "removed"
    assert saved_call.rule_triggered.startswith("rollback:abc-123")
    store.update_rule_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_rollback_at_threshold():
    store = _make_store(fp_rate=0.05)
    store.get_enforcement_by_id = AsyncMock(return_value={
        "enforcement_id": "xyz-999",
        "subject": "user456",
        "label": "violence",
        "action": "applied",
        "mode": "sync",
        "verdict": "unsafe",
        "severity": 4,
        "confidence": 0.8,
        "rule_triggered": "violence_rule",
    })

    ozone = Ozone(store=store)

    with patch.object(ozone, "_fp_threshold", 0.02):
        success = await ozone.rollback("xyz-999", "batch fp review")

    assert success is True
    # fp_rates should be updated with the rate returned by the store (0.05)
    assert ozone._fp_rates.get("violence", 0.0) == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_remove_label_action():
    ozone = Ozone(store=_make_store())
    result = await ozone.remove_label("user123", "spam", reason="appeal approved")
    assert result.action == "removed"
    assert result.label == "spam"
    assert result.verdict == "safe"
    assert result.severity == 0


@pytest.mark.asyncio
async def test_evaluate_routes_to_correct_mode():
    store = _make_store()
    ozone = Ozone(store=store)

    # severity >= 4 → SYNC
    high = await ozone.evaluate("user1", {"severity": 5, "unsafe": True, "category": "cbrn"})
    assert high.mode == EnforcementMode.SYNC
    assert high.action == "applied"

    # requires_human_review → QUARANTINE
    quar = await ozone.evaluate(
        "user2",
        {"severity": 3, "unsafe": True, "category": "csam", "requires_human_review": True},
    )
    assert quar.mode == EnforcementMode.QUARANTINE
    assert quar.action == "quarantined"

    # severity 2 → ASYNC
    mid = await ozone.evaluate("user3", {"severity": 2, "unsafe": True, "category": "hate"})
    assert mid.mode == EnforcementMode.ASYNC
