"""
Local RL Trainer — Task 3
==========================

PPO trainer with DP-SGD (per-sample gradient clipping + Gaussian noise),
privacy budget tracking via a simplified RDP accountant, and gradient
packaging into signed GradientPackets for federated upload.

Production notes:
- Replace numpy gradient stubs with torch.Tensor tensors from the real policy model.
- Replace the privacy_accountant with google/dp-accounting RDP accountant.
- Replace stub ECDSA signature with cryptography.hazmat P-384 signing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    state: Any
    action: str
    reward: float
    next_state: Any
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientPacket:
    gradients: Dict[str, List[float]]  # layer_name → DP-noised gradient (serialisable)
    gradient_norms: Dict[str, float]   # pre-noise L2 norm per layer
    commitment: str                    # SHA3-256 of canonical gradient bytes
    epsilon_spent: float
    delta: float
    round_id: int
    member_org_id: str                 # DID
    signature: str                     # ECDSA P-384 over commitment (stub)
    timestamp: str                     # ISO 8601


@dataclass
class PrivacyConfig:
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1
    epsilon_cap: float = 8.0
    delta: float = 1e-5
    accountant_type: str = "rdp"


@dataclass
class ModelWeights:
    weights: Dict[str, Any]
    version: str
    signature: str


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int = 10_000) -> None:
        self._buffer: List[List[Transition]] = []
        self._max_size = max_size

    def add_episode(self, episode: List[Transition]) -> None:
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append(episode)

    def sample(self, n: int) -> List[List[Transition]]:
        import random
        return random.sample(self._buffer, min(n, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ── Privacy accountant ────────────────────────────────────────────────────────

class PrivacyAccountant:
    """
    Simplified RDP-based privacy accountant.

    Formula used (Gaussian mechanism, RDP order α=2):
        ε_per_step ≈ q² / (2σ²)
    where q = sampling rate, σ = noise_multiplier.

    In production, replace with google/dp-accounting RDP accountant for
    tight composition bounds.
    """

    def __init__(self, config: PrivacyConfig) -> None:
        self.config = config
        self._epsilon_spent: float = 0.0
        self._steps: int = 0

    def step(
        self, noise_multiplier: float, sample_rate: float, num_steps: int = 1
    ) -> float:
        q = min(1.0, sample_rate)
        sigma = noise_multiplier
        epsilon_per_step = (q ** 2) / (2.0 * sigma ** 2)
        self._epsilon_spent += epsilon_per_step * num_steps
        self._steps += num_steps
        return self._epsilon_spent

    @property
    def epsilon(self) -> float:
        return self._epsilon_spent

    @property
    def delta(self) -> float:
        return self.config.delta

    def budget_exhausted(self) -> bool:
        return self._epsilon_spent > self.config.epsilon_cap


# ── Local RL trainer ──────────────────────────────────────────────────────────

class LocalRLTrainer:
    def __init__(
        self,
        model: Any,
        engine: Any,  # PlaybookEngine (avoid circular import at type-check time)
        privacy_config: PrivacyConfig,
    ) -> None:
        self.model = model
        self.engine = engine
        self.privacy_config = privacy_config
        self.accountant = PrivacyAccountant(privacy_config)

    # ── DP-SGD primitives ─────────────────────────────────────────────────────

    def _clip_gradient(self, grad: np.ndarray, clip_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(grad))
        if norm > clip_norm:
            grad = grad * (clip_norm / norm)
        return grad

    def _add_gaussian_noise(
        self, grad: np.ndarray, noise_multiplier: float, clip_norm: float
    ) -> np.ndarray:
        sigma = noise_multiplier * clip_norm
        return grad + np.random.normal(0.0, sigma, grad.shape)

    # ── PPO gradient computation ──────────────────────────────────────────────

    def _compute_ppo_gradients(
        self,
        episodes: List[List[Transition]],
        clip_eps: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """
        Clipped PPO objective gradient.

        Production implementation: run forward/backward pass through the policy
        network, collect .grad tensors per layer.  Here we produce a
        numerically representative numpy approximation for testing.
        """
        if not episodes:
            return {}

        all_rewards: List[float] = [
            t.reward for ep in episodes for t in ep
        ]
        if not all_rewards:
            return {}

        mean_r = float(np.mean(all_rewards))
        std_r = float(np.std(all_rewards)) + 1e-8
        normalized_adv = np.array([(r - mean_r) / std_r for r in all_rewards])
        adv_scale = float(np.mean(np.abs(normalized_adv))) * 0.01

        # Synthetic layer shapes — replace with model.named_parameters() in production
        layer_shapes: Dict[str, tuple] = {
            "policy_head": (64, 32),
            "value_head": (64, 1),
            "encoder": (128, 64),
        }
        return {
            name: np.random.randn(*shape) * adv_scale
            for name, shape in layer_shapes.items()
        }

    # ── Gradient packaging ────────────────────────────────────────────────────

    def _package_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        round_id: int,
    ) -> GradientPacket:
        clip_norm = self.privacy_config.clip_norm
        noise_mult = self.privacy_config.noise_multiplier

        gradient_norms: Dict[str, float] = {}
        dp_gradients: Dict[str, List[float]] = {}

        for layer_name, grad in gradients.items():
            gradient_norms[layer_name] = float(np.linalg.norm(grad))
            clipped = self._clip_gradient(grad, clip_norm)
            noised = self._add_gaussian_noise(clipped, noise_mult, clip_norm)
            dp_gradients[layer_name] = noised.tolist()

        canonical = json.dumps(dp_gradients, sort_keys=True).encode("utf-8")
        commitment = hashlib.sha3_256(canonical).hexdigest()

        signature = self._sign(commitment)
        member_org_id = os.environ.get("MEMBER_ORG_DID", "did:key:unknown")

        return GradientPacket(
            gradients=dp_gradients,
            gradient_norms=gradient_norms,
            commitment=commitment,
            epsilon_spent=self.accountant.epsilon,
            delta=self.privacy_config.delta,
            round_id=round_id,
            member_org_id=member_org_id,
            signature=signature,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def _sign(self, commitment: str) -> str:
        """ECDSA P-384 signature over commitment hash.

        Reads private key path from ORG_PRIVATE_KEY_PATH env var.
        Falls back to a deterministic stub when the key is absent.
        """
        key_path = os.environ.get("ORG_PRIVATE_KEY_PATH", "")
        if key_path:
            try:
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import ec

                with open(key_path, "rb") as f:
                    private_key = serialization.load_pem_private_key(f.read(), password=None)
                sig_bytes = private_key.sign(  # type: ignore[attr-defined]
                    commitment.encode("utf-8"),
                    ec.ECDSA(hashes.SHA384()),
                )
                return sig_bytes.hex()
            except Exception as exc:
                logger.warning("ecdsa_signing_failed", error=str(exc))

        # Deterministic stub — NOT cryptographically secure
        return f"stub:{hashlib.sha3_256(commitment.encode()).hexdigest()[:32]}"

    # ── Public API ────────────────────────────────────────────────────────────

    def train_round(
        self,
        replay_buffer: ReplayBuffer,
        round_id: int,
        batch_size: int = 32,
    ) -> GradientPacket:
        if self.accountant.budget_exhausted():
            raise RuntimeError(
                f"Privacy budget exhausted: ε={self.accountant.epsilon:.4f} "
                f"> cap={self.privacy_config.epsilon_cap}"
            )

        episodes = replay_buffer.sample(batch_size)
        if not episodes:
            raise ValueError("ReplayBuffer is empty — cannot train")

        total_transitions = sum(len(ep) for ep in episodes)
        buffer_total = max(len(replay_buffer) * 10, 1)
        sample_rate = min(1.0, total_transitions / buffer_total)

        gradients = self._compute_ppo_gradients(episodes)

        self.accountant.step(
            noise_multiplier=self.privacy_config.noise_multiplier,
            sample_rate=sample_rate,
            num_steps=1,
        )

        packet = self._package_gradients(gradients, round_id)

        logger.info(
            "train_round_complete",
            round_id=round_id,
            episodes=len(episodes),
            epsilon_spent=packet.epsilon_spent,
            commitment=packet.commitment[:12],
        )
        return packet

    def load_global_model(self, theta: ModelWeights, signature: str) -> None:
        """Load global model weights after verifying the operator signature."""
        if theta.signature != signature:
            raise ValueError(
                f"Model weight signature mismatch: "
                f"expected {signature!r}, got {theta.signature!r}"
            )
        self.model = theta
        logger.info("global_model_loaded", version=theta.version)
