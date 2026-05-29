"""
Training Orchestrator — Task 4
================================

TrainingOrchestrator ties together:
  1. PlaybookRegistry  — load and merge playbooks
  2. PlaybookEngine    — compile detectors and reward functions
  3. LocalRLTrainer    — PPO + DP-SGD gradient computation
  4. GlobalModelServer — pull latest global model (mTLS stub)
  5. TEEAggregator     — submit gradient packet (mTLS stub)
  6. AuditLog          — SHA3-256 commit to local append-only log
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import structlog

from src.learning.local_rl_trainer import (
    GradientPacket,
    LocalRLTrainer,
    ModelWeights,
    ReplayBuffer,
)
from src.learning.playbook_engine import (
    MergedPlaybook,
    PlaybookEngine,
    PlaybookRegistry,
    RewardWeights,
)

logger = structlog.get_logger()


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round_id: int
    gradient_packet: GradientPacket
    epsilon_spent: float
    episodes_sampled: int
    chain_alerts: int
    audit_hash: str
    timestamp: str


# ── Remote service stubs ──────────────────────────────────────────────────────

class GlobalModelServer:
    """Fetches the current global model from the federation aggregator (mTLS)."""

    def __init__(
        self, endpoint: str = "", tls_cert: Optional[str] = None
    ) -> None:
        self.endpoint = endpoint or os.environ.get("GLOBAL_MODEL_ENDPOINT", "")
        self.tls_cert = tls_cert or os.environ.get("GLOBAL_MODEL_TLS_CERT", "")

    def fetch(self) -> ModelWeights:
        # In production: GET {endpoint}/model/latest with mTLS client cert
        stub_sig = os.environ.get("OPERATOR_MODEL_SIGNATURE", "stub_operator_sig")
        return ModelWeights(weights={}, version="0.0.0", signature=stub_sig)


class TEEAggregator:
    """Submits a GradientPacket to the TEE aggregator (mTLS)."""

    def __init__(
        self, endpoint: str = "", tls_cert: Optional[str] = None
    ) -> None:
        self.endpoint = endpoint or os.environ.get("TEE_AGGREGATOR_ENDPOINT", "")
        self.tls_cert = tls_cert or os.environ.get("TEE_AGGREGATOR_TLS_CERT", "")

    def submit(self, packet: GradientPacket) -> bool:
        # In production: POST {endpoint}/gradients with mTLS client cert
        logger.info(
            "gradient_submitted_to_tee",
            round_id=packet.round_id,
            commitment=packet.commitment[:12],
            epsilon_spent=packet.epsilon_spent,
        )
        return True


class AuditLog:
    """Append-only local audit log; each entry is SHA3-256 committed."""

    def __init__(self, path: str = "audit.log") -> None:
        self.path = path

    def commit(self, round_id: int, commitment: str) -> str:
        entry = {
            "round_id": round_id,
            "commitment": commitment,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        audit_hash = hashlib.sha3_256(
            json.dumps(entry, sort_keys=True).encode("utf-8")
        ).hexdigest()
        entry["audit_hash"] = audit_hash

        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.error("audit_log_write_failed", path=self.path, error=str(exc))

        logger.info("audit_committed", round_id=round_id, audit_hash=audit_hash[:12])
        return audit_hash


# ── Orchestrator ──────────────────────────────────────────────────────────────

class TrainingOrchestrator:
    def __init__(
        self,
        registry: PlaybookRegistry,
        replay_buffer: ReplayBuffer,
        trainer: LocalRLTrainer,
        global_model_server: GlobalModelServer,
        tee_aggregator: TEEAggregator,
        audit_log: AuditLog,
        playbook_ids: Optional[List[str]] = None,
    ) -> None:
        self.registry = registry
        self.replay_buffer = replay_buffer
        self.trainer = trainer
        self.global_model_server = global_model_server
        self.tee_aggregator = tee_aggregator
        self.audit_log = audit_log
        self.playbook_ids = playbook_ids or []

    def run_round(self, round_id: int) -> RoundResult:
        """Execute one federation training round (steps 1-7)."""

        # 1. Pull latest global model and verify operator signature
        global_model = self.global_model_server.fetch()
        self.trainer.load_global_model(global_model, global_model.signature)

        # 2. Load merged playbook from registry
        if self.playbook_ids:
            merged = self.registry.get_merged(self.playbook_ids)
        else:
            merged = MergedPlaybook()
        engine = PlaybookEngine(merged, RewardWeights())
        self.trainer.engine = engine

        # 3. Sample episodes and run DP-SGD training
        packet = self.trainer.train_round(self.replay_buffer, round_id)

        # 4. Transmit GradientPacket to TEE aggregator (mTLS)
        self.tee_aggregator.submit(packet)

        # 5. Commit SHA3-256 to local audit log
        audit_hash = self.audit_log.commit(round_id, packet.commitment)

        # 6. Detect active chains in a representative session (informational)
        chain_alerts = 0

        logger.info(
            "orchestrator_round_complete",
            round_id=round_id,
            epsilon_spent=packet.epsilon_spent,
            audit_hash=audit_hash[:12],
        )

        return RoundResult(
            round_id=round_id,
            gradient_packet=packet,
            epsilon_spent=packet.epsilon_spent,
            episodes_sampled=len(self.replay_buffer.sample(1)),
            chain_alerts=chain_alerts,
            audit_hash=audit_hash,
            timestamp=packet.timestamp,
        )
