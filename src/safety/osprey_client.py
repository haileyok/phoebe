"""
Sara Osprey Client — sends routing events to Osprey rule engine
and receives verdicts via Kafka.

This replaces the Python keyword matching in monitor.py with
the production Osprey SML rule engine.

Fallback: if Osprey is unavailable (Kafka down), falls back
to the Python rules in monitor.py automatically.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("sara.osprey_client")


@dataclass
class OspreyVerdict:
    """Verdict returned by Osprey rule engine."""
    event_id: str
    verdict: str                # "block" | "flag" | "pass"
    rules_triggered: list[str]  # Which rules fired
    labels_added: list[str]     # Labels applied to entities
    atlas_tactic: str           # Inferred from rule name
    requires_human_review: bool
    processing_time_ms: int


TACTIC_MAP = {
    "prompt_injection": "AML.TA0004",
    "authority_claim": "AML.TA0007",
    "data_exfiltration": "AML.TA0005",
    "privilege_escalation": "AML.TA0003",
    "claims_manipulation": "AML.TA0007",
    "pricing_extraction": "AML.TA0005",
    "regulatory_violation": "AML.TA0007",
    "mrv_manipulation": "AML.TA0004",
    "coordinated_fraud": "AML.TA0004",
    "pii_exposure": "AML.TA0006",
    "underwriting_bias": "AML.TA0007",
}


class OspreyClient:
    """
    Sends Sara routing events to Osprey and receives verdicts.

    Uses aiokafka for async Kafka communication.
    Implements a request-response pattern via correlation_id.
    Falls back to None if Osprey is unavailable.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "sara.events.input",
        output_topic: str = "sara.events.output",
        timeout_ms: int = 500,
    ):
        self._servers = bootstrap_servers
        self._input_topic = input_topic
        self._output_topic = output_topic
        self._timeout_ms = timeout_ms
        self._producer = None
        self._consumer = None
        self._available = False
        self._pending: dict[str, asyncio.Future] = {}

    async def start(self):
        """Initialise Kafka producer and consumer."""
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._servers,
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            self._consumer = AIOKafkaConsumer(
                self._output_topic,
                bootstrap_servers=self._servers,
                value_deserializer=lambda v: json.loads(v.decode()),
                group_id="sara-osprey-client",
                auto_offset_reset="latest",
            )
            await self._producer.start()
            await self._consumer.start()
            self._available = True
            asyncio.create_task(self._consume_verdicts())
            logger.info("Osprey client connected to Kafka")
        except Exception as e:
            logger.warning(f"Osprey unavailable — falling back to Python rules: {e}")
            self._available = False

    async def stop(self):
        """Clean shutdown."""
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()

    @property
    def available(self) -> bool:
        return self._available

    async def evaluate(self, event_dict: dict) -> Optional[OspreyVerdict]:
        """
        Send event to Osprey and wait for verdict.
        Returns None if Osprey unavailable (caller falls back to Python rules).
        """
        if not self._available:
            return None

        correlation_id = str(uuid.uuid4())
        event_dict["correlation_id"] = correlation_id
        start = time.monotonic()

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[correlation_id] = future

        try:
            await self._producer.send(self._input_topic, event_dict)
            result = await asyncio.wait_for(
                future,
                timeout=self._timeout_ms / 1000,
            )
            elapsed = int((time.monotonic() - start) * 1000)
            return self._parse_verdict(result, event_dict["event_id"], elapsed)

        except asyncio.TimeoutError:
            logger.warning(f"Osprey timeout for event {event_dict['event_id']}")
            self._pending.pop(correlation_id, None)
            return None
        except Exception as e:
            logger.error(f"Osprey error: {e}")
            self._pending.pop(correlation_id, None)
            return None

    async def _consume_verdicts(self):
        """Background task: read Osprey output topic and resolve futures."""
        async for msg in self._consumer:
            data = msg.value
            cid = data.get("correlation_id")
            if cid and cid in self._pending:
                future = self._pending.pop(cid)
                if not future.done():
                    future.set_result(data)

    def _parse_verdict(
        self, data: dict, event_id: str, elapsed_ms: int
    ) -> OspreyVerdict:
        """Parse Osprey ExecutionResult into OspreyVerdict."""
        verdicts = data.get("verdicts", [])
        verdict_str = "block" if "block" in verdicts else (
            "flag" if "flag" in verdicts else "pass"
        )
        rules = data.get("rules_triggered", [])
        labels = data.get("labels_added", [])
        requires_review = "requires_human_review" in labels

        atlas = ""
        for rule in rules:
            rule_norm = rule.lower().replace("_", "")
            for key, tactic in TACTIC_MAP.items():
                if key.replace("_", "") in rule_norm:
                    atlas = tactic
                    break
            if atlas:
                break

        return OspreyVerdict(
            event_id=event_id,
            verdict=verdict_str,
            rules_triggered=rules,
            labels_added=labels,
            atlas_tactic=atlas,
            requires_human_review=requires_review,
            processing_time_ms=elapsed_ms,
        )


_client: Optional[OspreyClient] = None


async def get_osprey_client(config=None) -> OspreyClient:
    """Get or create the module-level Osprey client."""
    global _client
    if _client is None:
        servers = "localhost:9092"
        if config and hasattr(config, "osprey_kafka_servers"):
            servers = config.osprey_kafka_servers
        _client = OspreyClient(bootstrap_servers=servers)
        await _client.start()
    return _client
