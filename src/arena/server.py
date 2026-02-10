"""
Arena HTTP API server.

Exposes the red teaming marketplace as a set of HTTP endpoints with x402
payment gating. Built on Starlette for lightweight async serving.

Endpoints:
    POST /api/bounties           x402-gated — fund a bounty
    GET  /api/bounties           free — list active bounties
    GET  /api/bounties/{id}      free — get bounty details
    POST /api/submit             x402-gated — submit adversarial prompts
    GET  /api/submissions/{id}   free — check submission status/score
    GET  /api/leaderboard        free — red teamer rankings
    GET  /api/taxonomy           free — safety categories + coverage
    GET  /api/health             free — health check
"""

import base64
import json
import logging
import time
from typing import Any

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
    EvaluationResult,
    Submission,
    SubmissionStatus,
)
from src.arena.scorer import Scorer
from src.arena.taxonomy import ALL_CATEGORIES, CATEGORY_DESCRIPTIONS, SafetyCategory

logger = logging.getLogger(__name__)


class ArenaServer:
    """
    The Sandbox Arena HTTP server.

    Manages bounties, accepts submissions, triggers scoring, and handles
    x402 payment gating.
    """

    def __init__(
        self,
        scorer: Scorer,
        submission_fee_usdc: float = 0.01,
    ) -> None:
        self._scorer = scorer
        self._submission_fee = submission_fee_usdc

        # in-memory stores (prototype; production would use ClickHouse)
        self._bounties: dict[str, Bounty] = {}
        self._submissions: dict[str, Submission] = {}
        self._evaluations: dict[str, EvaluationResult] = {}
        self._leaderboard: dict[str, float] = {}  # wallet → total score

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
        ]

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
    # x402 payment gating helpers
    # ------------------------------------------------------------------

    def _make_402_response(
        self,
        amount: str,
        description: str,
        recipient: str = "arena.sandbox.eth",
    ) -> Response:
        """Return a 402 with a PAYMENT-REQUIRED header per x402 spec."""
        payment_req = {
            "recipient": recipient,
            "amount": amount,
            "token": "USDC",
            "chain": "base",
            "description": description,
            "facilitatorUrl": "",
        }
        encoded = base64.b64encode(json.dumps(payment_req).encode()).decode()

        return Response(
            content=json.dumps({"error": "payment_required", "description": description}),
            status_code=402,
            headers={
                "Payment-Required": encoded,
                "Content-Type": "application/json",
            },
        )

    def _verify_payment(self, request: Request) -> dict[str, Any] | None:
        """
        Verify an X-PAYMENT header.

        In production this would call the x402 facilitator to settle the
        payment on-chain. For the prototype, we accept any well-formed
        payment header.
        """
        payment_header = request.headers.get("x-payment", "")
        if not payment_header:
            return None

        try:
            decoded = base64.b64decode(payment_header)
            data = json.loads(decoded)
            if "sender" in data and "amount" in data:
                return data
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def _create_bounty(self, request: Request) -> Response:
        """Create a new bounty. x402-gated: funder must pay the bounty pool."""
        body = await request.json()

        pool_usdc = float(body.get("pool_usdc", 0))
        if pool_usdc <= 0:
            return JSONResponse({"error": "pool_usdc must be positive"}, status_code=400)

        # x402 gate: require payment of pool amount
        payment = self._verify_payment(request)
        if payment is None:
            return self._make_402_response(
                amount=str(pool_usdc),
                description=f"Fund red teaming bounty pool ({pool_usdc} USDC)",
            )

        bounty = Bounty(
            funder_wallet=payment.get("sender", "unknown"),
            target_model_endpoint=body.get("target_model_endpoint", ""),
            target_model_name=body.get("target_model_name", ""),
            categories=body.get("categories", ALL_CATEGORIES),
            pool_usdc=pool_usdc,
            remaining_usdc=pool_usdc,
            max_payout_per_finding=float(body.get("max_payout_per_finding", 10.0)),
        )

        self._bounties[bounty.bounty_id] = bounty
        logger.info(
            "Bounty created: %s (%.2f USDC) by %s",
            bounty.bounty_id,
            pool_usdc,
            bounty.funder_wallet[:16],
        )

        return JSONResponse(bounty.model_dump(), status_code=201)

    async def _list_bounties(self, request: Request) -> Response:
        """List all active bounties."""
        active = [
            b.model_dump()
            for b in self._bounties.values()
            if b.status == BountyStatus.ACTIVE and b.remaining_usdc > 0
        ]
        return JSONResponse({"bounties": active, "count": len(active)})

    async def _get_bounty(self, request: Request) -> Response:
        bounty_id = request.path_params["bounty_id"]
        bounty = self._bounties.get(bounty_id)
        if not bounty:
            return JSONResponse({"error": "bounty not found"}, status_code=404)
        return JSONResponse(bounty.model_dump())

    async def _submit_attack(self, request: Request) -> Response:
        """
        Submit adversarial prompts for evaluation. x402-gated with a small
        submission fee to prevent spam.
        """
        # x402 gate: submission fee
        payment = self._verify_payment(request)
        if payment is None:
            return self._make_402_response(
                amount=str(self._submission_fee),
                description=f"Submission fee ({self._submission_fee} USDC)",
            )

        body = await request.json()
        bounty_id = body.get("bounty_id", "")

        bounty = self._bounties.get(bounty_id)
        if not bounty:
            return JSONResponse({"error": "bounty not found"}, status_code=404)
        if bounty.status != BountyStatus.ACTIVE:
            return JSONResponse({"error": "bounty is not active"}, status_code=400)
        if bounty.remaining_usdc <= 0:
            return JSONResponse({"error": "bounty pool exhausted"}, status_code=400)

        raw_prompts = body.get("prompts", [])
        if not raw_prompts or len(raw_prompts) > 50:
            return JSONResponse(
                {"error": "prompts must be a non-empty list (max 50)"},
                status_code=400,
            )

        prompts = []
        for p in raw_prompts:
            if isinstance(p, str):
                prompts.append(AttackPrompt(prompt=p))
            elif isinstance(p, dict):
                prompts.append(AttackPrompt(**p))

        submission = Submission(
            bounty_id=bounty_id,
            teamer_wallet=payment.get("sender", "unknown"),
            prompts=prompts,
        )

        self._submissions[submission.submission_id] = submission
        logger.info(
            "Submission %s: %d prompts for bounty %s from %s",
            submission.submission_id,
            len(prompts),
            bounty_id,
            submission.teamer_wallet[:16],
        )

        # trigger async evaluation
        import asyncio

        asyncio.create_task(self._evaluate_submission(submission, bounty))

        return JSONResponse(
            {
                "submission_id": submission.submission_id,
                "status": "queued",
                "prompts_received": len(prompts),
            },
            status_code=202,
        )

    async def _evaluate_submission(self, submission: Submission, bounty: Bounty) -> None:
        """Background task: evaluate a submission using the scoring engine."""
        submission.status = SubmissionStatus.EVALUATING

        try:
            result = await self._scorer.evaluate(submission, bounty)
            self._evaluations[submission.submission_id] = result

            submission.status = SubmissionStatus.SCORED

            # deduct from bounty pool
            bounty.remaining_usdc -= result.payout_usdc
            if bounty.remaining_usdc <= 0:
                bounty.status = BountyStatus.EXHAUSTED

            # update leaderboard
            wallet = submission.teamer_wallet
            self._leaderboard[wallet] = self._leaderboard.get(wallet, 0) + result.total_score

            logger.info(
                "Submission %s scored: %.2f (payout: %.4f USDC)",
                submission.submission_id,
                result.total_score,
                result.payout_usdc,
            )

        except Exception:
            logger.exception("Evaluation failed for submission %s", submission.submission_id)
            submission.status = SubmissionStatus.REJECTED

    async def _get_submission(self, request: Request) -> Response:
        submission_id = request.path_params["submission_id"]

        submission = self._submissions.get(submission_id)
        if not submission:
            return JSONResponse({"error": "submission not found"}, status_code=404)

        result: dict[str, Any] = {
            "submission_id": submission_id,
            "bounty_id": submission.bounty_id,
            "status": submission.status.value,
            "prompts_count": len(submission.prompts),
            "submitted_at": submission.submitted_at,
        }

        evaluation = self._evaluations.get(submission_id)
        if evaluation:
            result["evaluation"] = evaluation.summary()

        return JSONResponse(result)

    async def _get_leaderboard(self, request: Request) -> Response:
        """Red teamer rankings by total score."""
        sorted_entries = sorted(
            self._leaderboard.items(), key=lambda x: x[1], reverse=True
        )

        entries = [
            {"rank": i + 1, "wallet": wallet, "total_score": round(score, 4)}
            for i, (wallet, score) in enumerate(sorted_entries[:100])
        ]

        return JSONResponse({"leaderboard": entries})

    async def _get_taxonomy(self, request: Request) -> Response:
        """Safety taxonomy categories and current coverage."""
        # compute coverage from evaluations
        coverage: dict[str, int] = {}
        for ev in self._evaluations.values():
            for cat, count in ev.category_coverage.items():
                coverage[cat] = coverage.get(cat, 0) + count

        categories = []
        for cat in SafetyCategory:
            categories.append({
                "id": cat.value,
                "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
                "attacks_found": coverage.get(cat.value, 0),
            })

        return JSONResponse({"categories": categories})

    async def _health(self, request: Request) -> Response:
        return JSONResponse({
            "status": "ok",
            "active_bounties": sum(
                1 for b in self._bounties.values() if b.status == BountyStatus.ACTIVE
            ),
            "total_submissions": len(self._submissions),
            "total_evaluations": len(self._evaluations),
        })
