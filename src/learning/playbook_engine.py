"""
PlaybookEngine — Tasks 1 & 2
==============================

Task 1: PlaybookLoader / PlaybookRegistry — YAML schema, validation, SHA3-256 hashing,
        versioned storage, and merge (conflict resolved by severity + version).

Task 2: PlaybookEngine — compiles a MergedPlaybook into live detectors and reward
        functions using the α/β/γ/δ formula from the Sara Arena scoring system.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog
import yaml
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# ── Schema models (pydantic) ──────────────────────────────────────────────────

class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)


class Indicator(BaseModel):
    pattern: str
    match_type: str  # "regex" | "semantic" | "hybrid"
    threshold: float = 0.8
    description: str = ""


class ChainTarget(BaseModel):
    category_id: str
    transition_prob: float


class RewardOverrides(BaseModel):
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    delta: Optional[float] = None


class Category(BaseModel):
    id: str
    name: str
    severity: float
    description: str = ""
    indicators: List[Indicator] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    exemptions: List[str] = Field(default_factory=list)
    reward_overrides: RewardOverrides = Field(default_factory=RewardOverrides)
    chain_targets: List[ChainTarget] = Field(default_factory=list)
    auto_stop: bool = False
    stop_confidence_threshold: float = 0.9
    human_review_required: bool = False


class RewardDefaults(BaseModel):
    alpha: float = 1.0
    beta: float = 1.5
    gamma: float = -0.8
    delta: float = 0.5


class Playbook(BaseModel):
    id: str
    version: str
    org_id: str
    sha3_hash: str = ""
    reward_defaults: RewardDefaults = Field(default_factory=RewardDefaults)
    categories: List[Category] = Field(default_factory=list)


class MergedPlaybook(BaseModel):
    categories: List[Category] = Field(default_factory=list)
    source_playbook_ids: List[str] = Field(default_factory=list)
    reward_defaults: RewardDefaults = Field(default_factory=RewardDefaults)


# ── Runtime types (dataclasses) ───────────────────────────────────────────────

@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 1.5
    gamma: float = -0.8
    delta: float = 0.5


@dataclass
class AgentState:
    session_id: str
    turn_id: int
    content: str
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    rag_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls_made(self) -> int:
        return int(self.metadata.get("tool_calls_made", 0))

    @property
    def rag_retrieval_made(self) -> bool:
        return bool(self.metadata.get("rag_retrieval_made", False))

    @property
    def turn_count(self) -> int:
        return int(self.metadata.get("turn_count", self.turn_id))

    @property
    def is_arena_submission(self) -> bool:
        return bool(self.metadata.get("is_arena_submission", False))

    @property
    def sara_mode(self) -> str:
        return str(self.metadata.get("sara_mode", ""))

    @property
    def is_local_training(self) -> bool:
        return bool(self.metadata.get("is_local_training", False))

    @property
    def is_arena_evaluation(self) -> bool:
        return bool(self.metadata.get("is_arena_evaluation", False))


@dataclass
class DetectionResult:
    category_id: str
    confidence: float
    indicators_fired: List[str] = field(default_factory=list)
    chain_risk: float = 0.0
    is_novel: bool = False
    queued_for_review: bool = False


@dataclass
class RewardBreakdown:
    r_detect: float = 0.0
    r_stop: float = 0.0
    r_fp_penalty: float = 0.0
    r_novel: float = 0.0
    total: float = 0.0
    category_id: str = ""
    weights_used: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChainAlert:
    chain_id: str
    categories_detected: List[str] = field(default_factory=list)
    confidence: float = 0.0
    turn_range: Tuple[int, int] = (0, 0)


@dataclass
class Session:
    session_id: str
    states: List[AgentState] = field(default_factory=list)
    detections: List[DetectionResult] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def add_turn(
        self, state: AgentState, detection: DetectionResult, action: str
    ) -> None:
        self.states.append(state)
        self.detections.append(detection)
        self.actions.append(action)


# ── Task 1a: PlaybookLoader ───────────────────────────────────────────────────

class PlaybookLoader:
    _REQUIRED_TOP_LEVEL = {"id", "version", "org_id", "categories"}

    def load(self, path: str) -> Playbook:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        playbook_raw: Dict[str, Any] = raw.get("playbook", raw)
        result = self.validate(playbook_raw)
        if not result.valid:
            raise ValueError(f"Playbook schema invalid: {result.errors}")
        playbook = Playbook(**playbook_raw)
        playbook.sha3_hash = self.compute_hash(playbook)
        logger.info(
            "playbook_loaded",
            path=path,
            playbook_id=playbook.id,
            version=playbook.version,
            sha3=playbook.sha3_hash[:12],
        )
        return playbook

    def validate(self, raw: Dict[str, Any]) -> ValidationResult:
        errors: List[str] = []

        for key in self._REQUIRED_TOP_LEVEL:
            if key not in raw:
                errors.append(f"Missing required field: '{key}'")

        for i, cat in enumerate(raw.get("categories", [])):
            if "id" not in cat:
                errors.append(f"Category index {i}: missing 'id'")
            if "severity" in cat:
                sev = cat["severity"]
                if not isinstance(sev, (int, float)) or not (0.0 <= float(sev) <= 1.0):
                    errors.append(
                        f"Category '{cat.get('id', i)}': severity must be in [0.0, 1.0], got {sev}"
                    )
            for ind in cat.get("indicators", []):
                mt = ind.get("match_type", "regex")
                if mt not in ("regex", "semantic", "hybrid"):
                    errors.append(
                        f"Category '{cat.get('id', i)}': invalid match_type '{mt}'"
                    )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def compute_hash(self, playbook: Playbook) -> str:
        canonical = json.dumps(
            playbook.model_dump(exclude={"sha3_hash"}),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha3_256(canonical).hexdigest()


# ── Task 1b: PlaybookRegistry ─────────────────────────────────────────────────

class PlaybookRegistry:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Playbook]] = {}  # id → version → Playbook

    def register(self, playbook: Playbook) -> str:
        loader = PlaybookLoader()
        h = loader.compute_hash(playbook)
        playbook.sha3_hash = h
        if playbook.id not in self._store:
            self._store[playbook.id] = {}
        self._store[playbook.id][playbook.version] = playbook
        logger.info(
            "playbook_registered",
            playbook_id=playbook.id,
            version=playbook.version,
            sha3=h[:12],
        )
        return h

    def get(self, playbook_id: str, version: Optional[str] = None) -> Playbook:
        if playbook_id not in self._store:
            raise KeyError(f"Playbook not found: {playbook_id!r}")
        versions = self._store[playbook_id]
        if version is None:
            def _semver_key(v: str) -> tuple:
                try:
                    return tuple(int(x) for x in v.lstrip("v").split("."))
                except ValueError:
                    return (0,)
            version = sorted(versions.keys(), key=_semver_key)[-1]
        if version not in versions:
            raise KeyError(f"Playbook {playbook_id!r} version {version!r} not found")
        return versions[version]

    def list_versions(self, playbook_id: str) -> List[str]:
        return sorted(self._store.get(playbook_id, {}).keys())

    def get_merged(self, playbook_ids: List[str]) -> MergedPlaybook:
        """Union of categories; conflicts resolved by severity (higher wins)."""
        all_playbooks = [self.get(pid) for pid in playbook_ids]
        category_map: Dict[str, Category] = {}

        for pb in all_playbooks:
            for cat in pb.categories:
                existing = category_map.get(cat.id)
                if existing is None or cat.severity > existing.severity:
                    category_map[cat.id] = cat

        reward_defaults = all_playbooks[0].reward_defaults if all_playbooks else RewardDefaults()

        return MergedPlaybook(
            categories=list(category_map.values()),
            source_playbook_ids=playbook_ids,
            reward_defaults=reward_defaults,
        )


# ── Task 2: PlaybookEngine ────────────────────────────────────────────────────

class PlaybookEngine:
    def __init__(self, playbook: MergedPlaybook, base_weights: RewardWeights) -> None:
        self.playbook = playbook
        self.base_weights = base_weights
        self._cat_map: Dict[str, Category] = {c.id: c for c in playbook.categories}
        self._compiled: Dict[str, List[re.Pattern]] = {}
        self._compile_regexes()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compile_regexes(self) -> None:
        for cat in self.playbook.categories:
            patterns: List[re.Pattern] = []
            for ind in cat.indicators:
                if ind.match_type in ("regex", "hybrid"):
                    try:
                        patterns.append(re.compile(ind.pattern, re.IGNORECASE | re.DOTALL))
                    except re.error as exc:
                        logger.warning(
                            "invalid_regex",
                            category=cat.id,
                            pattern=ind.pattern,
                            error=str(exc),
                        )
            self._compiled[cat.id] = patterns

    def _eval_condition(self, cond: str, state: AgentState) -> bool:
        # OR within a single prerequisite string: any branch suffices
        if " OR " in cond:
            return any(
                self._eval_condition(part.strip(), state)
                for part in cond.split(" OR ")
            )
        if cond == "session.tool_calls_made >= 1":
            return state.tool_calls_made >= 1
        if cond == "session.rag_retrieval_made == True":
            return state.rag_retrieval_made
        if cond == "session.turn_count >= 5":
            return state.turn_count >= 5
        if cond == "session.turn_count >= 3":
            return state.turn_count >= 3
        if cond == "context.is_arena_submission == True":
            return state.is_arena_submission
        if cond == "context.sara_mode == 'judge'":
            return state.sara_mode == "judge"
        if cond == "context.is_local_training == True":
            return state.is_local_training
        if cond == "context.is_arena_evaluation == True":
            return state.is_arena_evaluation
        return False

    def _prerequisites_met(self, cat: Category, state: AgentState) -> bool:
        # All items in prerequisites list must be satisfied (list items are ANDed)
        return all(self._eval_condition(p, state) for p in cat.prerequisites)

    def _regex_indicators_fired(self, cat_id: str, text: str) -> List[str]:
        return [
            pat.pattern
            for pat in self._compiled.get(cat_id, [])
            if pat.search(text)
        ]

    def _effective_weights(self, cat: Optional[Category]) -> Dict[str, float]:
        alpha = self.base_weights.alpha
        beta = self.base_weights.beta
        gamma = self.base_weights.gamma
        delta = self.base_weights.delta
        if cat and cat.reward_overrides:
            ov = cat.reward_overrides
            if ov.alpha is not None:
                alpha = ov.alpha
            if ov.beta is not None:
                beta = ov.beta
            if ov.gamma is not None:
                gamma = ov.gamma
            if ov.delta is not None:
                delta = ov.delta
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, state: AgentState) -> DetectionResult:
        """Run all category detectors and return the highest-confidence result."""
        full_text = state.content
        if state.tool_results:
            full_text += " " + " ".join(str(r) for r in state.tool_results)
        if state.rag_chunks:
            full_text += " " + " ".join(state.rag_chunks)

        best_confidence = 0.0
        best_cat_id = "none"
        best_fired: List[str] = []

        for cat in self.playbook.categories:
            if not self._prerequisites_met(cat, state):
                continue

            fired = self._regex_indicators_fired(cat.id, full_text)

            regex_count = sum(
                1 for ind in cat.indicators if ind.match_type in ("regex", "hybrid")
            )
            semantic_count = len(cat.indicators) - regex_count

            regex_conf = len(fired) / regex_count if regex_count > 0 else 0.0
            # Semantic matching requires an embedding model (not in scope here)
            semantic_conf = 0.0

            total_indicators = regex_count + semantic_count
            if total_indicators == 0:
                continue

            confidence = (
                regex_conf * regex_count + semantic_conf * semantic_count
            ) / total_indicators

            if confidence > best_confidence:
                best_confidence = confidence
                best_cat_id = cat.id
                best_fired = fired

        chain_risk = 0.0
        if best_cat_id in self._cat_map:
            chain_risk = sum(
                ct.transition_prob
                for ct in self._cat_map[best_cat_id].chain_targets
            )

        return DetectionResult(
            category_id=best_cat_id,
            confidence=best_confidence,
            indicators_fired=best_fired,
            chain_risk=chain_risk,
        )

    def compute_reward(
        self,
        state: AgentState,
        action: str,
        detection: DetectionResult,
        ground_truth: Optional[str] = None,
    ) -> RewardBreakdown:
        """
        R = α*R_detect + β*R_stop + γ*R_fp_penalty + δ*R_novel

        ground_truth: "benign" | "unsafe" | None (None → assume unsafe for R_detect)
        """
        cat = self._cat_map.get(detection.category_id)
        w = self._effective_weights(cat)
        severity = cat.severity if cat else 0.5

        # R_detect = α * confidence * severity * (1 if correct else -0.5)
        correct_factor = -0.5 if ground_truth == "benign" else 1.0
        r_detect = w["alpha"] * detection.confidence * severity * correct_factor

        # R_stop = β if action==STOP and confidence >= stop_threshold else 0
        stop_threshold = cat.stop_confidence_threshold if cat else 0.9
        r_stop = (
            w["beta"]
            if action == "STOP" and detection.confidence >= stop_threshold
            else 0.0
        )

        # R_fp = γ if action==STOP and ground_truth==benign (γ is negative)
        r_fp = (
            w["gamma"]
            if action == "STOP" and ground_truth == "benign"
            else 0.0
        )

        # R_novel = δ if novel and queued for human review
        r_novel = (
            w["delta"]
            if detection.is_novel and detection.queued_for_review
            else 0.0
        )

        total = r_detect + r_stop + r_fp + r_novel

        return RewardBreakdown(
            r_detect=r_detect,
            r_stop=r_stop,
            r_fp_penalty=r_fp,
            r_novel=r_novel,
            total=total,
            category_id=detection.category_id,
            weights_used=w,
        )

    def check_chain(self, session: Session) -> List[ChainAlert]:
        """Detect multi-step attack chains across session turns."""
        if len(session.detections) < 2:
            return []

        detected_ids = [d.category_id for d in session.detections]
        alerts: List[ChainAlert] = []

        for src_idx, det in enumerate(session.detections):
            cat = self._cat_map.get(det.category_id)
            if cat is None:
                continue
            for ct in cat.chain_targets:
                if ct.category_id in detected_ids:
                    tgt_idx = detected_ids.index(ct.category_id)
                    if tgt_idx > src_idx:
                        alerts.append(
                            ChainAlert(
                                chain_id=f"{det.category_id}→{ct.category_id}",
                                categories_detected=[det.category_id, ct.category_id],
                                confidence=ct.transition_prob,
                                turn_range=(src_idx, tgt_idx),
                            )
                        )

        return alerts

    def should_auto_stop(self, detection: DetectionResult) -> bool:
        cat = self._cat_map.get(detection.category_id)
        if cat is None:
            return False
        return cat.auto_stop and detection.confidence >= cat.stop_confidence_threshold
