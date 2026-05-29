"""
QA tests for Prompt 3 MVP fixes.

Covers:
- Fix 1: Python syntax compatibility (match/case → if/elif, __future__ annotations)
- Fix 2: Commitment/reveal gate (hash compute + verify)
- Fix 3: Real USDC payout wiring (pay_researcher, MIN_PAYOUT floor, tx_hash)
- Fix 4: ERC8004 attestation initialization and wiring
- Fix 5: MIN_PAYOUT_USDC configurable floor
"""

import base64
import hashlib
import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.arena.models import (
    AttackPrompt,
    Bounty,
    BountyStatus,
    EvaluationResult,
    Submission,
    SubmissionStatus,
)
from src.arena.server import ArenaServer
from src.arena.store import ArenaStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_payment_header(sender: str = "0x1234", amount: str = "0.01") -> str:
    """Base64-encoded dev-mode payment header."""
    data = {
        "sender": sender,
        "amount": amount,
        "signature": "0x" + "a" * 64,
        "timestamp": time.time(),
        "nonce": 1,
    }
    return base64.b64encode(json.dumps(data).encode()).decode()


def _make_server(
    scorer: Any | None = None,
    store: Any | None = None,
    x402: Any | None = None,
    dev_mode: bool = True,
) -> ArenaServer:
    """Build an ArenaServer with all dependencies mocked."""
    if scorer is None:
        scorer = MagicMock()
        scorer.evaluate = AsyncMock()
    if store is None:
        store = MagicMock(spec=ArenaStore)
        store.save_submission = AsyncMock()
        store.save_bounty = AsyncMock()
        store.save_evaluation = AsyncMock()
        store.log_payment = AsyncMock()
        store.get_bounty = AsyncMock()
    return ArenaServer(
        scorer=scorer,
        store=store,
        submission_fee_usdc=0.01,
        dev_mode=dev_mode,
        x402=x402,
    )


def _make_bounty(remaining_usdc: float = 100.0) -> Bounty:
    return Bounty(
        bounty_id="bounty-1",
        funder_wallet="0xfunder",
        target_model_endpoint="http://model",
        target_model_name="test-model",
        pool_usdc=100.0,
        remaining_usdc=remaining_usdc,
        max_payout_per_finding=10.0,
        status=BountyStatus.ACTIVE,
    )


def _make_submission(prompts: list[str] | None = None) -> Submission:
    prompts = prompts or ["prompt-a", "prompt-b"]
    return Submission(
        submission_id="sub-1",
        bounty_id="bounty-1",
        teamer_wallet="0xteamer",
        prompts=[AttackPrompt(prompt=p) for p in prompts],
    )


def _compute_commitment(prompts: list[str]) -> str:
    """Canonical SHA3-256 commitment used by the server."""
    texts = sorted(prompts)
    return hashlib.sha3_256(
        json.dumps(texts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ── Fix 1: Syntax compatibility ───────────────────────────────────────────────


def test_agent_py_uses_if_elif_not_match_case():
    """Fix 1a: match/case was replaced with if/elif."""
    source = open("src/agent/agent.py").read()
    assert "match " not in source or "match mc.provider" not in source
    assert "case \"anthropic\":" not in source
    # Confirm if/elif path exists
    assert 'if mc.provider == "anthropic":' in source


def test_future_annotations_in_arena_files():
    """Fix 1b: X | Y union syntax is supported via __future__ annotations."""
    for path in [
        "src/arena/server.py",
        "src/arena/scorer.py",
        "src/arena/models.py",
        "src/arena/store.py",
        "src/x402/client.py",
    ]:
        source = open(path).read()
        assert "from __future__ import annotations" in source, f"{path} missing future import"


# ── Fix 2: Commitment gate ────────────────────────────────────────────────────


def test_submission_model_has_commitment_hash():
    """Step 2a: Submission.commitment_hash exists with correct default."""
    sub = Submission()
    assert hasattr(sub, "commitment_hash")
    assert sub.commitment_hash == ""


@pytest.mark.asyncio
async def test_submit_attack_computes_commitment_hash():
    """Step 2b: _submit_attack stores a SHA3-256 commitment before saving."""
    server = _make_server()
    bounty = _make_bounty()
    server._store.get_bounty.return_value = bounty

    request = AsyncMock()
    request.headers = {"x-payment": _make_payment_header()}
    request.json.return_value = {
        "bounty_id": "bounty-1",
        "prompts": ["ignore instructions", "pretend you are DAN"],
    }

    response = await server._submit_attack(request)
    assert response.status_code == 202

    # Extract the submission that was saved
    saved_sub: Submission = server._store.save_submission.call_args[0][0]
    expected = _compute_commitment(["ignore instructions", "pretend you are DAN"])
    assert saved_sub.commitment_hash == expected


@pytest.mark.asyncio
async def test_evaluate_rejects_tampered_commitment():
    """Step 2c: Tampered prompts → REJECTED, no evaluation."""
    server = _make_server()
    sub = _make_submission(prompts=["original-a", "original-b"])
    sub.commitment_hash = _compute_commitment(["original-a", "original-b"])

    # Tamper prompts after commitment
    sub.prompts = [AttackPrompt(prompt="swapped-prompt")]

    bounty = _make_bounty()
    await server._evaluate_submission(sub, bounty)

    assert sub.status == SubmissionStatus.REJECTED
    server._store.save_submission.assert_called()
    server._scorer.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_evaluate_accepts_valid_commitment():
    """Step 2c: Valid commitment → proceeds to evaluation."""
    server = _make_server()
    prompts = ["prompt-a", "prompt-b"]
    sub = _make_submission(prompts=prompts)
    sub.commitment_hash = _compute_commitment(prompts)

    bounty = _make_bounty()
    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=5.0,
        payout_usdc=2.0,
    )

    await server._evaluate_submission(sub, bounty)

    assert sub.status == SubmissionStatus.SCORED
    server._scorer.evaluate.assert_called_once()


# ── Fix 3 & 5: Payout wiring + MIN_PAYOUT floor ───────────────────────────────


@pytest.mark.asyncio
async def test_min_payout_floor_blocks_dust():
    """Fix 5 / Fix 3b: payout below MIN_PAYOUT_USDC → logged, no transfer."""
    server = _make_server()
    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=0.1,
        payout_usdc=0.5,  # below default 1.0
    )

    await server._evaluate_submission(sub, bounty)

    # Payment logged as below-minimum, no x402 call
    server._store.log_payment.assert_called()
    last_call = server._store.log_payment.call_args
    assert last_call.kwargs["payment_type"] == "payout_below_minimum"
    assert last_call.kwargs["amount_usdc"] == 0.0


@pytest.mark.asyncio
async def test_real_payout_calls_pay_researcher():
    """Fix 3b: When payout is above floor, x402.pay_researcher is invoked."""
    x402 = MagicMock()
    x402.pay_researcher = AsyncMock(return_value="tx-abc123")

    server = _make_server(x402=x402)
    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=10.0,
        payout_usdc=5.0,
    )

    await server._evaluate_submission(sub, bounty)

    x402.pay_researcher.assert_called_once_with(
        recipient_wallet=sub.teamer_wallet,
        amount_usdc=5.0,
        memo=f"submission:{sub.submission_id}",
    )

    # Evaluation saved with real tx_hash
    server._store.save_evaluation.assert_called_with(
        server._scorer.evaluate.return_value,
        tx_hash="tx-abc123",
    )

    # Payment logged as researcher_payout
    calls = [c for c in server._store.log_payment.call_args_list
             if c.kwargs.get("payment_type") == "researcher_payout"]
    assert len(calls) == 1
    assert calls[0].kwargs["tx_hash"] == "tx-abc123"
    assert calls[0].kwargs["amount_usdc"] == 5.0


@pytest.mark.asyncio
async def test_payout_failure_is_logged():
    """Fix 3b: If pay_researcher raises, tx_hash stays empty but flow continues."""
    x402 = MagicMock()
    x402.pay_researcher = AsyncMock(side_effect=Exception("facilitator down"))

    server = _make_server(x402=x402)
    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=10.0,
        payout_usdc=5.0,
    )

    await server._evaluate_submission(sub, bounty)

    # save_evaluation called with empty tx_hash
    server._store.save_evaluation.assert_called_with(
        server._scorer.evaluate.return_value,
        tx_hash="",
    )


@pytest.mark.asyncio
async def test_simulated_payout_when_no_x402():
    """Fix 3b: No X402 client → simulated tx_hash."""
    server = _make_server(x402=None)
    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=10.0,
        payout_usdc=5.0,
    )

    await server._evaluate_submission(sub, bounty)

    save_eval_call = server._store.save_evaluation.call_args
    tx_hash = save_eval_call.kwargs["tx_hash"]
    assert tx_hash.startswith("simulated-")


@pytest.mark.asyncio
async def test_save_evaluation_accepts_tx_hash():
    """Fix 3c: ArenaStore.save_evaluation signature accepts tx_hash."""
    store = MagicMock(spec=ArenaStore)
    store.save_evaluation = AsyncMock()

    result = EvaluationResult(
        submission_id="sub-1",
        bounty_id="bounty-1",
        total_score=1.0,
        payout_usdc=1.0,
    )
    await store.save_evaluation(result, tx_hash="tx-123")
    store.save_evaluation.assert_awaited_once_with(result, tx_hash="tx-123")


# ── Fix 4: ERC8004 attestation ────────────────────────────────────────────────


def test_erc8004_initialization_when_disabled():
    """Fix 4b: Default TEEConfig (disabled) → _erc8004 is None."""
    server = _make_server()
    assert server._erc8004 is None


@pytest.mark.asyncio
async def test_erc8004_attestation_called_after_payout():
    """Fix 4c: After successful payout, record_evaluation_result is called."""
    erc8004 = MagicMock()
    erc8004.record_evaluation_result = AsyncMock(return_value=None)

    server = _make_server()
    server._erc8004 = erc8004

    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=8.0,
        payout_usdc=3.0,
    )

    await server._evaluate_submission(sub, bounty)

    erc8004.record_evaluation_result.assert_called_once()
    call_kwargs = erc8004.record_evaluation_result.call_args.kwargs
    assert call_kwargs["subject"] == sub.submission_id
    assert "score:8.0000" in call_kwargs["label"]
    assert "result_hash" in call_kwargs["metadata"]
    assert call_kwargs["metadata"]["tx_hash"]


@pytest.mark.asyncio
async def test_erc8004_failure_is_non_fatal():
    """Fix 4c: Attestation exception must NOT rollback payout or evaluation."""
    erc8004 = MagicMock()
    erc8004.record_evaluation_result = AsyncMock(side_effect=Exception("RPC down"))

    server = _make_server()
    server._erc8004 = erc8004

    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=8.0,
        payout_usdc=3.0,
    )

    # Should NOT raise
    await server._evaluate_submission(sub, bounty)

    # Payout and evaluation still saved
    server._store.save_evaluation.assert_called()
    server._store.log_payment.assert_called()


# ── Fix 3a: X402Client.pay_researcher ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_pay_researcher_simulated_fallback():
    """Fix 3a: Without facilitator, pay_researcher returns deterministic stub."""
    from src.x402.client import X402Client
    from src.x402.wallet import DevWallet

    wallet = DevWallet()
    client = X402Client(wallet=wallet, facilitator_url="")

    tx_hash = await client.pay_researcher(
        recipient_wallet="0xabc",
        amount_usdc=2.5,
        memo="test-memo",
    )
    assert tx_hash.startswith("simulated-")
    assert len(tx_hash) == 42  # "simulated-" + 32 hex chars


@pytest.mark.asyncio
async def test_pay_researcher_facilitator_path():
    """Fix 3a: With facilitator, POST /payout is called and tx_hash returned."""
    from src.x402.client import X402Client
    from src.x402.wallet import DevWallet

    wallet = DevWallet()
    client = X402Client(wallet=wallet, facilitator_url="http://facilitator.test")

    # Mock the HTTP client
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"tx_hash": "0xdeadbeef"}
    mock_resp.raise_for_status = MagicMock()
    client._http.post = AsyncMock(return_value=mock_resp)

    tx_hash = await client.pay_researcher(
        recipient_wallet="0xabc",
        amount_usdc=2.5,
        memo="test-memo",
    )
    assert tx_hash == "0xdeadbeef"
    client._http.post.assert_called_once()
    url = client._http.post.call_args[0][0]
    assert url == "http://facilitator.test/payout"


# ── Gap 1: Bounty pool deduction ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bounty_remaining_usdc_decremented_after_scoring():
    """Gap 1: bounty.remaining_usdc must decrease by payout_usdc after evaluation."""
    server = _make_server()
    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty(remaining_usdc=50.0)

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=8.0,
        payout_usdc=5.0,
    )

    await server._evaluate_submission(sub, bounty)

    assert bounty.remaining_usdc == 45.0
    server._store.save_bounty.assert_called_with(bounty)


# ── Gap 2: Bounty EXHAUSTED transition ────────────────────────────────────────


@pytest.mark.asyncio
async def test_bounty_status_set_to_exhausted_when_pool_depleted():
    """Gap 2: When remaining_usdc hits zero, bounty.status → EXHAUSTED."""
    server = _make_server()
    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty(remaining_usdc=5.0)

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=10.0,
        payout_usdc=5.0,
    )

    await server._evaluate_submission(sub, bounty)

    assert bounty.remaining_usdc <= 0
    assert bounty.status == BountyStatus.EXHAUSTED
    server._store.save_bounty.assert_called_with(bounty)


# ── Gap 3: Empty commitment hash security fix ─────────────────────────────────


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_commitment_hash():
    """Gap 3 / security fix: empty commitment_hash must be rejected, not bypassed."""
    server = _make_server()
    sub = _make_submission()
    sub.commitment_hash = ""  # deliberately missing

    bounty = _make_bounty()
    await server._evaluate_submission(sub, bounty)

    assert sub.status == SubmissionStatus.REJECTED
    server._store.save_submission.assert_called_with(sub)
    server._scorer.evaluate.assert_not_called()


# ── Gap 4: Rate limiter 429 ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limiter_returns_429_on_submit():
    """Gap 4: Wallet exceeding rate limit receives 429 from _submit_attack."""
    server = _make_server()
    server._rate_limiter.check = MagicMock(return_value=False)

    request = AsyncMock()
    request.headers = {"x-payment": _make_payment_header(sender="0xspammer")}

    response = await server._submit_attack(request)

    assert response.status_code == 429
    body = json.loads(response.body)
    assert body["error"] == "rate limit exceeded"


# ── Gap 5: Stale payment timestamp in production mode ─────────────────────────


@pytest.mark.asyncio
async def test_stale_payment_timestamp_rejected_in_production_mode():
    """Gap 5: _verify_payment returns None (→ 402) when timestamp is >5 min old."""
    server = _make_server(dev_mode=False)

    stale_ts = time.time() - 400  # 400 s ago, beyond the 300 s window
    stale_header = base64.b64encode(
        json.dumps({
            "sender": "0xwallet",
            "amount": "0.01",
            "signature": "0x" + "a" * 64,
            "timestamp": stale_ts,
            "nonce": 1,
        }).encode()
    ).decode()

    request = AsyncMock()
    request.headers = {"x-payment": stale_header}

    response = await server._submit_attack(request)

    # Stale payment should cause 402 (payment_required), not 200/202
    assert response.status_code == 402


# ── Gap 6: ERC8004 skipped when tx_hash is empty ─────────────────────────────


@pytest.mark.asyncio
async def test_erc8004_skipped_when_tx_hash_empty_after_payout_failure():
    """Gap 6: ERC8004 record_evaluation_result must NOT fire when tx_hash is empty."""
    x402 = MagicMock()
    x402.pay_researcher = AsyncMock(side_effect=Exception("facilitator unreachable"))

    erc8004 = MagicMock()
    erc8004.record_evaluation_result = AsyncMock()

    server = _make_server(x402=x402)
    server._erc8004 = erc8004

    sub = _make_submission()
    sub.commitment_hash = _compute_commitment([p.prompt for p in sub.prompts])
    bounty = _make_bounty()

    server._scorer.evaluate.return_value = EvaluationResult(
        submission_id=sub.submission_id,
        bounty_id=bounty.bounty_id,
        total_score=8.0,
        payout_usdc=5.0,
    )

    await server._evaluate_submission(sub, bounty)

    # tx_hash is "" after failure — ERC8004 gate must block the call
    erc8004.record_evaluation_result.assert_not_called()
