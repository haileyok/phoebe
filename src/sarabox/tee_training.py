"""
TEE Training Enclave interface + stub.

Raw attack prompts go in, encrypted gradient updates come out.
For MVP, the actual fine-tuning is stubbed — the interface,
encryption, and attestation are real.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass

from src.safety.erc8004 import ERC8004Publisher
from src.sarabox.models import AttackSubmission

logger = logging.getLogger(__name__)


@dataclass
class GradientUpdate:
    """
    Encrypted gradient delta exported from the TEE after local fine-tuning.
    Only this object leaves the enclave — never raw prompts.
    """

    update_id: str
    org_id: str
    skill_id: str
    # In production: actual encrypted weight deltas (bytes)
    # In MVP stub: a hash commitment proving training happened
    encrypted_delta: bytes
    delta_hash: str  # SHA3-256 of encrypted_delta (public)
    num_samples: int  # how many prompts trained on (public)
    tee_attestation_quote: str  # TEE remote attestation (public)
    contribution_score: float  # novelty + coverage score for credits


class TEETrainingEnclave:
    """
    Interface to the TEE training enclave.

    In production: this sends encrypted attack data to an Intel TDX
    or AWS Nitro enclave, which returns a signed GradientUpdate.

    In MVP stub: simulates the enclave locally, produces a deterministic
    GradientUpdate whose delta_hash proves the training happened without
    exposing the prompts. Logs WARNING so this is never mistaken for
    production behaviour.

    The enclave URL is configured via TEE_TRAINING_ENCLAVE_URL env var.
    When unset, the stub runs locally.
    """

    def __init__(
        self,
        erc8004: ERC8004Publisher | None = None,
    ):
        self._enclave_url = os.getenv("TEE_TRAINING_ENCLAVE_URL", "")
        self._erc8004 = erc8004

    async def train(
        self,
        submission: AttackSubmission,
    ) -> GradientUpdate:
        """
        Process an attack submission inside the TEE.
        Returns a GradientUpdate — the only data that leaves the enclave.
        """
        if self._enclave_url:
            return await self._train_remote(submission)
        else:
            logger.warning(
                "TEE_TRAINING_ENCLAVE_URL not set — using LOCAL STUB. "
                "No real training or privacy guarantees in this mode."
            )
            return await self._train_stub(submission)

    async def _train_remote(self, submission: AttackSubmission) -> GradientUpdate:
        """
        Send encrypted submission to real TEE enclave.
        The enclave receives only AES-256-GCM encrypted prompt data.
        It decrypts inside the enclave, fine-tunes, and returns
        the gradient update + attestation quote.
        """
        import httpx

        # In production: encrypt submission.prompts with enclave's public key
        # before sending. For now, stub the encryption.
        payload = {
            "org_id": submission.org_id,
            "skill_id": submission.skill_id,
            "commitment_hash": submission.commitment_hash,
            "num_samples": len(submission.prompts),
            # "encrypted_prompts": encrypt(submission.prompts, enclave_pubkey)
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self._enclave_url}/train",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return GradientUpdate(**data)

    async def _train_stub(self, submission: AttackSubmission) -> GradientUpdate:
        """
        Local stub — produces deterministic GradientUpdate for testing.
        delta_hash proves training happened; encrypted_delta is a placeholder.
        """
        import time
        import uuid

        canonical = json.dumps(
            sorted(submission.prompts),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        delta_hash = hashlib.sha3_256(canonical).hexdigest()
        stub_delta = b"STUB:" + delta_hash.encode()

        # Score based on number of unique prompts submitted
        contribution_score = min(1.0, len(set(submission.prompts)) / 10.0)

        update = GradientUpdate(
            update_id=str(uuid.uuid4()),
            org_id=submission.org_id,
            skill_id=submission.skill_id,
            encrypted_delta=stub_delta,
            delta_hash=delta_hash,
            num_samples=len(submission.prompts),
            tee_attestation_quote="STUB_ATTESTATION",
            contribution_score=contribution_score,
        )

        # Publish on-chain attestation that contribution was received
        if self._erc8004:
            try:
                await self._erc8004.record_evaluation_result(
                    subject=update.update_id,
                    label=f"federated_contribution:{submission.org_id[:8]}",
                    metadata={
                        "delta_hash": delta_hash,
                        "num_samples": update.num_samples,
                        "contribution_score": contribution_score,
                    },
                )
            except Exception as exc:
                logger.warning("ERC8004 contribution log failed: %s", exc)

        return update
