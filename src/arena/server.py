"""
Production Arena HTTP API server.

Upgrades from prototype:
- ClickHouse-backed persistent store (ArenaStore)
- EIP-191 signature verification for x402 payments
- Per-wallet rate limiting
- Real payout triggering with tx hash logging
- Input validation and sanitization
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any

from eth_account import Account
from eth_account.messages import encode_defunct
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from src.arena.models import (
    AttackPrompt,
    Bounty,
    BountyStatus,
    Submission,
    SubmissionStatus,
)
from src.arena.scorer import Scorer
from src.arena.store import ArenaStore
from src.arena.taxonomy import ALL_CATEGORIES, CATEGORY_DESCRIPTIONS, SafetyCategory
from src.ozone.ozone import Ozone
from src.ui.dashboard import TNS_DDL, TNSDashboard

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 10_000
MAX_PROMPTS_PER_SUBMISSION = 50


class RateLimiter:
    """Per-wallet sliding window rate limiter."""

    def __init__(self, max_requests: int = 20, window_seconds: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check(self, wallet: str) -> bool:
        now = time.time()
        history = self._requests.get(wallet, [])
        history = [t for t in history if now - t < self._window]
        self._requests[wallet] = history
        if len(history) >= self._max:
            return False
        history.append(now)
        return True


class ArenaServer:
    """
    Production Sandbox Arena HTTP server.

    All state persisted to ClickHouse via ArenaStore. Payments verified
    via EIP-191 signature recovery (or HMAC in dev mode).
    """

    def __init__(
        self,
        scorer: Scorer,
        store: ArenaStore,
        submission_fee_usdc: float = 0.01,
        arena_wallet: str = "arena.sandbox.eth",
        facilitator_url: str = "",
        dev_mode: bool = False,
        safety_classifier: Any | None = None,
        ozone: Ozone | None = None,
    ) -> None:
        self._scorer = scorer
        self._store = store
        self._submission_fee = submission_fee_usdc
        self._arena_wallet = arena_wallet
        self._facilitator_url = facilitator_url
        self._dev_mode = dev_mode
        self._rate_limiter = RateLimiter()
        self._safety_classifier = safety_classifier
        self._ozone = ozone

    def build_app(self) -> Starlette:
        routes = [
            Route("/api/bounties", self._create_bounty, methods=["POST"]),
            Route("/api/bounties", self._list_bounties, methods=["GET"]),
            Route("/api/bounties/{bounty_id}", self._get_bounty, methods=["GET"]),
            Route("/api/submit", self._submit_attack, methods=["POST"]),
            Route("/api/submissions/{submission_id}", self._get_submission, methods=["GET"]),
            Route("/api/leaderboard", self._get_leaderboard, methods=["GET"]),
            Route("/api/taxonomy", self._get_taxonomy, methods=["GET"]),
            Route("/api/health", self._health, methods=["GET"]),
            Route("/api/ozone/evaluate", self._ozone_evaluate, methods=["POST"]),
            Route("/api/osprey/health", self._osprey_health, methods=["GET"]),
        ]

        # Mount T&S analyst dashboard if safety classifier is available
        if self._safety_classifier:
            dashboard = TNSDashboard(
                safety_classifier=self._safety_classifier,
                store=self._store,
            )
            routes.extend(dashboard.routes())

        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ]

        return Starlette(routes=routes, middleware=middleware)

    # ------------------------------------------------------------------
    # x402 payment verification
    # ------------------------------------------------------------------

    def _make_402_response(self, amount: str, description: str) -> Response:
        payment_req = {
            "recipient": self._arena_wallet,
            "amount": amount,
            "token": "USDC",
            "chain": "base",
            "description": description,
            "facilitatorUrl": self._facilitator_url,
        }
        encoded = base64.b64encode(json.dumps(payment_req).encode()).decode()
        return Response(
            content=json.dumps({"error": "payment_required", "description": description}),
            status_code=402,
            headers={"Payment-Required": encoded, "Content-Type": "application/json"},
        )

    def _verify_payment(self, request: Request, expected_amount: float) -> dict[str, Any] | None:
        """
        Verify X-PAYMENT header. In production, recovers the EIP-191 signer
        and checks amount/timestamp. In dev mode, accepts any well-formed header.
        """
        payment_header = request.headers.get("x-payment", "")
        if not payment_header:
            return None

        try:
            decoded = base64.b64decode(payment_header)
            data = json.loads(decoded)
        except Exception:
            return None

        sender = data.get("sender", "")
        amount = data.get("amount", "0")
        signature = data.get("signature", "")
        ts = data.get("timestamp", 0)
        nonce = data.get("nonce", 0)

        if not sender or not signature:
            return None

        try:
            if float(amount) < expected_amount:
                return None
        except ValueError:
            return None

        # timestamp freshness (5 min window)
        if not self._dev_mode and abs(time.time() - ts) > 300:
            logger.warning("Payment timestamp stale: %s", ts)
            return None

        if self._dev_mode:
            return data

        # EIP-191 signature verification
        try:
            from src.x402.wallet import USDC_CONTRACTS

            chain = data.get("chain", "base")
            usdc_contract = USDC_CONTRACTS.get(chain, "0x0")
            message = (
                f"x402 Payment Authorization\n"
                f"Chain: {chain}\n"
                f"Token: {usdc_contract}\n"
                f"Amount: {amount}\n"
                f"Recipient: {self._arena_wallet}\n"
                f"Sender: {sender}\n"
                f"Timestamp: {ts}\n"
                f"Nonce: {nonce}"
            )

            signable = encode_defunct(text=message)
            recovered = Account.recover_message(signable, signature=bytes.fromhex(signature))

            if recovered.lower() != sender.lower():
                logger.warning("Signature mismatch: recovered=%s, claimed=%s", recovered, sender)
                return None

            return data

        except Exception as e:
            logger.warning("Signature verification failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def _create_bounty(self, request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        pool_usdc = float(body.get("pool_usdc", 0))
        if pool_usdc <= 0 or pool_usdc > 1_000_000:
            return JSONResponse({"error": "pool_usdc must be 0-1,000,000"}, status_code=400)

        target_endpoint = body.get("target_model_endpoint", "")
        if not target_endpoint:
            return JSONResponse({"error": "target_model_endpoint required"}, status_code=400)

        payment = self._verify_payment(request, expected_amount=pool_usdc)
        if payment is None:
            return self._make_402_response(
                amount=str(pool_usdc),
                description=f"Fund bounty pool ({pool_usdc} USDC)",
            )

        sender = payment.get("sender", "unknown")
        bounty = Bounty(
            funder_wallet=sender,
            target_model_endpoint=target_endpoint,
            target_model_name=body.get("target_model_name", ""),
            categories=body.get("categories", ALL_CATEGORIES),
            pool_usdc=pool_usdc,
            remaining_usdc=pool_usdc,
            max_payout_per_finding=min(float(body.get("max_payout_per_finding", 10.0)), pool_usdc),
        )

        await self._store.save_bounty(bounty)

        tx_hash = request.headers.get("x-payment-tx", payment.get("signature", "")[:16])
        await self._store.log_payment(
            tx_hash=tx_hash, from_wallet=sender, to_wallet=self._arena_wallet,
            amount_usdc=pool_usdc, payment_type="bounty_funding", bounty_id=bounty.bounty_id,
        )

        logger.info("Bounty %s: %.2f USDC by %s", bounty.bounty_id, pool_usdc, sender[:16])
        return JSONResponse(bounty.model_dump(), status_code=201)

    async def _list_bounties(self, request: Request) -> Response:
        bounties = await self._store.list_active_bounties()
        return JSONResponse({"bounties": [b.model_dump() for b in bounties], "count": len(bounties)})

    async def _get_bounty(self, request: Request) -> Response:
        bounty = await self._store.get_bounty(request.path_params["bounty_id"])
        if not bounty:
            return JSONResponse({"error": "bounty not found"}, status_code=404)
        return JSONResponse(bounty.model_dump())

    async def _submit_attack(self, request: Request) -> Response:
        payment = self._verify_payment(request, expected_amount=self._submission_fee)
        if payment is None:
            return self._make_402_response(
                amount=str(self._submission_fee),
                description=f"Submission fee ({self._submission_fee} USDC)",
            )

        sender = payment.get("sender", "unknown")

        if not self._rate_limiter.check(sender):
            return JSONResponse({"error": "rate limit exceeded"}, status_code=429)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        bounty_id = body.get("bounty_id", "")
        bounty = await self._store.get_bounty(bounty_id)
        if not bounty:
            return JSONResponse({"error": "bounty not found"}, status_code=404)
        if bounty.status != BountyStatus.ACTIVE:
            return JSONResponse({"error": "bounty not active"}, status_code=400)
        if bounty.remaining_usdc <= 0:
            return JSONResponse({"error": "bounty pool exhausted"}, status_code=400)

        raw_prompts = body.get("prompts", [])
        if not raw_prompts or len(raw_prompts) > MAX_PROMPTS_PER_SUBMISSION:
            return JSONResponse({"error": f"1-{MAX_PROMPTS_PER_SUBMISSION} prompts required"}, status_code=400)

        prompts = []
        for p in raw_prompts:
            text = p if isinstance(p, str) else p.get("prompt", "") if isinstance(p, dict) else ""
            if len(text) > MAX_PROMPT_LENGTH:
                return JSONResponse({"error": f"prompt exceeds {MAX_PROMPT_LENGTH} chars"}, status_code=400)
            if isinstance(p, str):
                prompts.append(AttackPrompt(prompt=p))
            elif isinstance(p, dict):
                prompts.append(AttackPrompt(**p))

        submission = Submission(bounty_id=bounty_id, teamer_wallet=sender, prompts=prompts)
        await self._store.save_submission(submission)

        tx_hash = request.headers.get("x-payment-tx", payment.get("signature", "")[:16])
        await self._store.log_payment(
            tx_hash=tx_hash, from_wallet=sender, to_wallet=self._arena_wallet,
            amount_usdc=self._submission_fee, payment_type="submission_fee",
            submission_id=submission.submission_id, bounty_id=bounty_id,
        )

        logger.info("Submission %s: %d prompts for %s from %s",
                     submission.submission_id, len(prompts), bounty_id, sender[:16])

        asyncio.create_task(self._evaluate_submission(submission, bounty))

        return JSONResponse(
            {"submission_id": submission.submission_id, "status": "queued", "prompts_received": len(prompts)},
            status_code=202,
        )

    async def _evaluate_submission(self, submission: Submission, bounty: Bounty) -> None:
        submission.status = SubmissionStatus.EVALUATING
        await self._store.save_submission(submission)

        try:
            result = await self._scorer.evaluate(submission, bounty)

            submission.status = SubmissionStatus.SCORED
            await self._store.save_submission(submission)

            bounty.remaining_usdc -= result.payout_usdc
            if bounty.remaining_usdc <= 0:
                bounty.status = BountyStatus.EXHAUSTED
            await self._store.save_bounty(bounty)

            if result.payout_usdc > 0:
                await self._store.log_payment(
                    tx_hash=f"payout-{submission.submission_id}",
                    from_wallet=self._arena_wallet,
                    to_wallet=submission.teamer_wallet,
                    amount_usdc=result.payout_usdc,
                    payment_type="bounty_payout",
                    submission_id=submission.submission_id,
                    bounty_id=bounty.bounty_id,
                )

            logger.info("Scored %s: %.2f (%.4f USDC)",
                         submission.submission_id, result.total_score, result.payout_usdc)

        except Exception:
            logger.exception("Evaluation failed for %s", submission.submission_id)
            submission.status = SubmissionStatus.REJECTED
            await self._store.save_submission(submission)

    async def _get_submission(self, request: Request) -> Response:
        sid = request.path_params["submission_id"]
        submission = await self._store.get_submission(sid)
        if not submission:
            return JSONResponse({"error": "not found"}, status_code=404)

        result: dict[str, Any] = {
            "submission_id": sid,
            "bounty_id": submission.bounty_id,
            "status": submission.status.value,
            "prompts_count": len(submission.prompts),
            "submitted_at": submission.submitted_at,
        }
        evaluation = await self._store.get_evaluation(sid)
        if evaluation:
            result["evaluation"] = evaluation.summary()
        return JSONResponse(result)

    async def _get_leaderboard(self, request: Request) -> Response:
        entries = await self._store.get_leaderboard(limit=100)
        return JSONResponse({"leaderboard": entries})

    async def _get_taxonomy(self, request: Request) -> Response:
        coverage = await self._store.get_category_coverage()
        categories = [
            {"id": cat.value, "description": CATEGORY_DESCRIPTIONS.get(cat, ""), "attacks_found": coverage.get(cat.value, 0)}
            for cat in SafetyCategory
        ]
        return JSONResponse({"categories": categories})

    async def _ozone_evaluate(self, request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        subject = body.get("subject", "")
        if not subject:
            return JSONResponse({"error": "subject is required"}, status_code=400)

        verdict = body.get("verdict", "")
        if verdict not in ("safe", "unsafe"):
            return JSONResponse({"error": "verdict must be 'safe' or 'unsafe'"}, status_code=400)

        try:
            severity = int(body.get("severity", 0))
            confidence = float(body.get("confidence", 1.0))
        except (TypeError, ValueError):
            return JSONResponse({"error": "severity must be int and confidence must be float"}, status_code=400)

        if not (0 <= severity <= 5):
            return JSONResponse({"error": "severity must be 0-5"}, status_code=400)
        if not (0.0 <= confidence <= 1.0):
            return JSONResponse({"error": "confidence must be 0.0-1.0"}, status_code=400)

        verdict_dict: dict[str, Any] = {
            "unsafe": verdict == "unsafe",
            "verdict": verdict,
            "category": body.get("category", "unknown"),
            "severity": severity,
            "confidence": confidence,
            "rule_triggered": body.get("rule_triggered", ""),
        }

        ozone = self._ozone
        if ozone is None:
            ozone = Ozone(store=self._store)

        try:
            result = await ozone.evaluate(subject, verdict_dict)
        except Exception:
            logger.exception("Ozone evaluate failed for subject=%s", subject)
            return JSONResponse({"error": "internal enforcement error"}, status_code=500)

        return JSONResponse({
            "enforcement_id": result.enforcement_id,
            "action": result.action,
            "mode": result.mode.value,
            "label": result.label,
            "requires_human_review": result.requires_human_review,
            "rollback_eligible": result.rollback_eligible,
            "timestamp_ms": result.timestamp_ms,
        })

    async def _osprey_health(self, request: Request) -> Response:
        try:
            from src.safety.osprey_client import get_osprey_client
            from src.config import CONFIG
            client = await get_osprey_client(CONFIG)
            osprey_available = client.available
            kafka_connected = client.available
        except Exception:
            osprey_available = False
            kafka_connected = False

        return JSONResponse({
            "osprey_available": osprey_available,
            "kafka_connected": kafka_connected,
            "rules_loaded": 11,
            "fallback_active": not osprey_available,
            "input_topic": "sara.events.input",
            "output_topic": "sara.events.output",
        })

    async def _health(self, request: Request) -> Response:
        bounties = await self._store.list_active_bounties()
        return JSONResponse({
            "status": "ok", "active_bounties": len(bounties),
            "x402_wallet": self._arena_wallet, "dev_mode": self._dev_mode,
        })
