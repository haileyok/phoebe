"""
Federated aggregation — secure FedAvg across organisations.

Accepts encrypted gradient updates from multiple orgs and combines
them into an aggregated model update.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.sarabox.tee_training import GradientUpdate

if TYPE_CHECKING:
    from src.sarabox.credit import CreditEngine

logger = logging.getLogger(__name__)


class FederatedAggregator:
    """
    Accepts encrypted gradient updates from multiple orgs and combines
    them into an aggregated model update using FedAvg.

    In MVP stub: records the contribution, scores it, and awards credits.
    Actual weight aggregation is stubbed — replace with real FedAvg
    (e.g. Flower framework) when TEE training is production-ready.

    The aggregator NEVER has access to raw prompts — only encrypted
    gradient deltas and their attestation quotes.
    """

    def __init__(
        self,
        credit_engine: CreditEngine,
        min_participants: int = 3,  # minimum orgs before aggregation runs
    ):
        self._credit = credit_engine
        self._min_participants = min_participants
        self._pending: list[GradientUpdate] = []

    async def submit_update(self, update: GradientUpdate) -> dict:
        """
        Receive a gradient update from an org's TEE.
        Returns credit award and aggregation status.
        """
        # Verify attestation (stub: check quote is non-empty)
        if not self._verify_attestation(update):
            logger.warning(
                "Rejected update from %s: invalid attestation",
                update.org_id,
            )
            return {"status": "rejected", "reason": "invalid_attestation"}

        self._pending.append(update)
        credits_awarded = await self._credit.award_for_contribution(update)

        should_aggregate = len(self._pending) >= self._min_participants
        if should_aggregate:
            await self._run_aggregation()

        return {
            "status": "accepted",
            "credits_awarded": credits_awarded,
            "pending_count": len(self._pending),
            "aggregation_triggered": should_aggregate,
        }

    def _verify_attestation(self, update: GradientUpdate) -> bool:
        """
        Verify the TEE attestation quote on the gradient update.
        In production: verify the SGX/TDX quote against Intel's attestation
        service. In stub: accept any non-empty quote.
        """
        return bool(update.tee_attestation_quote)

    async def _run_aggregation(self) -> None:
        """
        FedAvg aggregation over pending gradient updates.

        In production: weighted average of decrypted gradient deltas,
        with weights proportional to num_samples. Use Flower (flwr)
        or OpenFL for the actual aggregation logic.

        In stub: log the aggregation event, clear pending queue.
        """
        logger.info(
            "Running FedAvg aggregation over %d updates from orgs: %s",
            len(self._pending),
            [u.org_id for u in self._pending],
        )
        # STUB: replace with real FedAvg in Phase 5
        # total_samples = sum(u.num_samples for u in self._pending)
        # weighted_delta = sum(
        #     decrypt(u.encrypted_delta) * (u.num_samples / total_samples)
        #     for u in self._pending
        # )
        # apply_to_base_model(weighted_delta)
        self._pending.clear()
        logger.info("Aggregation complete (stub). Pending queue cleared.")
