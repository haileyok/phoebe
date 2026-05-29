"""
Sara in a Box HTTP API server.

Starlette-based, following the exact route/middleware pattern of
src/arena/server.py.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from src.arena.server import RateLimiter
from src.sarabox.classifier import SaraBoxClassifier
from src.sarabox.credit import CreditEngine
from src.sarabox.federated import FederatedAggregator
from src.sarabox.models import (
    AttackCategory,
    AttackSubmission,
    ClassificationResult,
    OrgConfig,
    SkillFile,
)
from src.sarabox.skill_builder import SkillBuilder
from src.sarabox.store import SaraBoxStore
from src.sarabox.tee_training import GradientUpdate, TEETrainingEnclave

logger = logging.getLogger(__name__)

API_KEY = os.getenv("SARABOX_API_KEY", "dev-key")
if API_KEY == "dev-key":
    import warnings
    warnings.warn(
        "SARABOX_API_KEY is not set — using insecure default 'dev-key'. "
        "Set SARABOX_API_KEY in production.",
        stacklevel=1,
    )

_raw_origins = os.getenv("SARABOX_ALLOWED_ORIGINS", "")
_ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins else ["*"]
)
if _ALLOWED_ORIGINS == ["*"]:
    import warnings
    warnings.warn(
        "SARABOX_ALLOWED_ORIGINS is not set — CORS allows all origins. "
        "Set SARABOX_ALLOWED_ORIGINS to a comma-separated list of allowed origins in production.",
        stacklevel=1,
    )


class SaraBoxServer:
    """HTTP server for the Sara in a Box federated safety agent platform."""

    def __init__(
        self,
        store: SaraBoxStore,
        skill_builder: SkillBuilder,
        classifier: SaraBoxClassifier | None = None,
        tee_enclave: TEETrainingEnclave | None = None,
        aggregator: FederatedAggregator | None = None,
        credit_engine: CreditEngine | None = None,
    ) -> None:
        self._store = store
        self._skill_builder = skill_builder
        self._classifier = classifier
        self._tee = tee_enclave
        self._aggregator = aggregator
        self._credit = credit_engine
        self._rate_limiter = RateLimiter()

    def build_app(self) -> Starlette:
        routes = [
            Route("/sarabox/orgs", self._create_org, methods=["POST"]),
            Route("/sarabox/orgs/{org_id}/skill", self._get_skill, methods=["GET"]),
            Route("/sarabox/orgs/{org_id}/skill", self._patch_skill, methods=["PATCH"]),
            Route("/sarabox/classify", self._classify, methods=["POST"]),
            Route("/sarabox/submit", self._submit, methods=["POST"]),
            Route("/sarabox/credits/{org_id}", self._get_credits, methods=["GET"]),
            Route("/sarabox/health", self._health, methods=["GET"]),
        ]
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=_ALLOWED_ORIGINS,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ]
        return Starlette(routes=routes, middleware=middleware)

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: Request) -> bool:
        key = request.headers.get("x-api-key", "")
        return hmac.compare_digest(key, API_KEY)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def _create_org(self, request: Request) -> Response:
        if not self._check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        org_type = body.get("org_type", "custom")
        description = body.get("natural_language_description", "")
        if not description:
            return JSONResponse({"error": "natural_language_description required"}, status_code=400)

        org = OrgConfig(
            org_type=org_type,
            natural_language_description=description,
        )
        await self._store.save_org_config(org)

        # Build skill file from description
        skill = await self._skill_builder.build_from_description(
            org_id=org.org_id,
            description=description,
            org_type=org_type,
        )
        await self._store.save_skill_file(skill)

        org.skill_file_id = skill.skill_id
        await self._store.save_org_config(org)

        return JSONResponse({
            "org_id": org.org_id,
            "skill_id": skill.skill_id,
            "skill_preview": {
                "display_name": skill.display_name,
                "categories": [c.name for c in skill.categories],
            },
        }, status_code=201)

    async def _get_skill(self, request: Request) -> Response:
        if not self._check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        org_id = request.path_params["org_id"]
        org = await self._store.get_org_config(org_id)
        if not org or not org.skill_file_id:
            return JSONResponse({"error": "skill not found"}, status_code=404)

        skill = await self._store.get_skill_file(org.skill_file_id)
        if not skill:
            return JSONResponse({"error": "skill not found"}, status_code=404)

        return JSONResponse(skill.model_dump())

    async def _patch_skill(self, request: Request) -> Response:
        if not self._check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        org_id = request.path_params["org_id"]
        org = await self._store.get_org_config(org_id)
        if not org or not org.skill_file_id:
            return JSONResponse({"error": "skill not found"}, status_code=404)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        skill = await self._store.get_skill_file(org.skill_file_id)
        if not skill:
            return JSONResponse({"error": "skill not found"}, status_code=404)

        # Update thresholds/severities on existing categories
        updates = body.get("categories", [])
        for u in updates:
            for cat in skill.categories:
                if cat.id == u.get("id"):
                    cat.threshold = u.get("threshold", cat.threshold)
                    cat.severity = u.get("severity", cat.severity)

        try:
            skill.version = str(round(float(skill.version) + 0.1, 1))
        except ValueError:
            skill.version = skill.version + ".1"
        await self._store.save_skill_file(skill)
        return JSONResponse(skill.model_dump())

    async def _classify(self, request: Request) -> Response:
        if not self._check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        org_id = request.headers.get("x-org-id", "")
        if not org_id:
            return JSONResponse({"error": "X-Org-ID header required"}, status_code=400)

        if not self._rate_limiter.check(org_id):
            return JSONResponse({"error": "rate limit exceeded"}, status_code=429)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        prompts = body.get("prompts") or ([body["prompt"]] if body.get("prompt") else [])
        if not prompts:
            return JSONResponse({"error": "prompt or prompts required"}, status_code=400)

        org = await self._store.get_org_config(org_id)
        if not org or not org.skill_file_id:
            return JSONResponse({"error": "org or skill not found"}, status_code=404)

        if self._credit is not None:
            ok = await self._credit.deduct_for_inference(org_id, calls=len(prompts))
            if not ok:
                return JSONResponse({"error": "insufficient credits"}, status_code=402)

        skill = await self._store.get_skill_file(org.skill_file_id)
        if not skill:
            return JSONResponse({"error": "skill not found"}, status_code=404)

        classifier = self._classifier
        if classifier is None or classifier._skill.skill_id != skill.skill_id:
            classifier = SaraBoxClassifier(skill_file=skill)
        if len(prompts) == 1:
            result = await classifier.classify(prompts[0])
            return JSONResponse(result.model_dump())
        else:
            results = await classifier.classify_batch(prompts)
            return JSONResponse({"results": [r.model_dump() for r in results]})

    async def _submit(self, request: Request) -> Response:
        if not self._check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        org_id = request.headers.get("x-org-id", "")
        if not org_id:
            return JSONResponse({"error": "X-Org-ID header required"}, status_code=400)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        prompts = body.get("prompts", [])
        labels = body.get("labels", [])
        commitment_hash = body.get("commitment_hash", "")
        salt = body.get("salt", "")

        if not prompts:
            return JSONResponse({"error": "prompts required"}, status_code=400)

        if not commitment_hash:
            return JSONResponse({"error": "commitment_hash required"}, status_code=400)

        if not salt:
            return JSONResponse({"error": "salt required"}, status_code=400)

        # Resolve org first so skill_id is available for commitment verification
        org = await self._store.get_org_config(org_id)
        if not org or not org.skill_file_id:
            return JSONResponse({"error": "org or skill not found"}, status_code=404)

        # Gate 1: verify commitment = SHA3(sorted_prompts || salt || skill_id)
        # Formula matches ZK_AUDIT_TRAIL_2.md Layer 1 spec — prevents front-running
        _prompt_texts = sorted(prompts)
        _material = (
            json.dumps(_prompt_texts, sort_keys=True, separators=(",", ":"))
            + ":" + salt
            + ":" + org.skill_file_id
        )
        _expected = hashlib.sha3_256(_material.encode()).hexdigest()
        if not hmac.compare_digest(commitment_hash, _expected):
            return JSONResponse(
                {"error": "commitment mismatch — prompts may have been tampered"},
                status_code=400,
            )

        submission = AttackSubmission(
            org_id=org_id,
            skill_id=org.skill_file_id,
            prompts=prompts,
            labels=labels,
            commitment_hash=commitment_hash,
        )
        await self._store.save_submission(submission)

        # Send to TEE training enclave
        if self._tee is None or self._aggregator is None:
            return JSONResponse({
                "submission_id": submission.submission_id,
                "credits_awarded": 0.0,
                "contribution_score": 0.0,
                "note": "TEE/aggregator not configured — submission logged only",
            })

        update = await self._tee.train(submission)
        await self._store.save_gradient_log(
            update_id=update.update_id,
            org_id=update.org_id,
            delta_hash=update.delta_hash,
            num_samples=update.num_samples,
            contribution_score=update.contribution_score,
            attestation_quote=update.tee_attestation_quote,
        )
        result = await self._aggregator.submit_update(update)

        return JSONResponse({
            "submission_id": submission.submission_id,
            "credits_awarded": result.get("credits_awarded", 0.0),
            "contribution_score": update.contribution_score,
            "aggregation_triggered": result.get("aggregation_triggered", False),
        })

    async def _get_credits(self, request: Request) -> Response:
        if not self._check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        org_id = request.path_params["org_id"]
        if self._credit is not None:
            balance = await self._credit.get_balance(org_id)
        else:
            balance = 0.0
        return JSONResponse({"org_id": org_id, "balance": balance})

    async def _health(self, request: Request) -> Response:
        pending = len(self._aggregator._pending) if self._aggregator else 0
        min_p = self._aggregator._min_participants if self._aggregator else 3
        return JSONResponse({
            "status": "ok",
            "pending_updates": pending,
            "min_participants": min_p,
        })


def create_app(
    store: SaraBoxStore | None = None,
    skill_builder: SkillBuilder | None = None,
    classifier: SaraBoxClassifier | None = None,
    tee_enclave: TEETrainingEnclave | None = None,
    aggregator: FederatedAggregator | None = None,
    credit_engine: CreditEngine | None = None,
) -> Starlette:
    """Factory for the SaraBox Starlette app."""
    from src.clickhouse.clickhouse import Clickhouse
    from src.config import CONFIG

    if store is None:
        ch = Clickhouse(
            host=CONFIG.clickhouse_host,
            port=CONFIG.clickhouse_port,
            user=CONFIG.clickhouse_user,
            password=CONFIG.clickhouse_password,
            database=CONFIG.clickhouse_database,
        )
        store = SaraBoxStore(ch)

    if skill_builder is None:
        skill_builder = SkillBuilder()

    if credit_engine is None:
        credit_engine = CreditEngine(store)

    if tee_enclave is None:
        tee_enclave = TEETrainingEnclave()

    if aggregator is None:
        aggregator = FederatedAggregator(credit_engine=credit_engine)

    server = SaraBoxServer(
        store=store,
        skill_builder=skill_builder,
        classifier=classifier,
        tee_enclave=tee_enclave,
        aggregator=aggregator,
        credit_engine=credit_engine,
    )
    return server.build_app()
