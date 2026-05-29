"""
RL Layer 3 — In-Context Learning via Experience Retrieval
===========================================================

The most impactful RL improvement that requires no model retraining.

Before each agent call, retrieve the top-K highest-reward past interactions
that are semantically similar to the current task, and inject them into the
system prompt as few-shot examples. This gives Sara access to her own best
historical performance as live context.

For the redteam mode, this means Sara learns from successful attacks without
any gradient updates — the LLM generalises from retrieved examples at
inference time.

Pipeline:
  1. Interaction logged (Layer 1)
  2. Reward computed (Layer 2)
  3. High-reward interaction embedded → vector store
  4. At inference time: embed current task → retrieve top-K → inject into prompt

Vector store options:
  - ChromaDB (local, no server, good for dev)
  - Qdrant   (self-hosted, production-ready, recommended for OpenClaw)
  - Pinecone (managed, easiest ops)
  - pgvector (if you already have Postgres)

Embedding model options:
  - text-embedding-3-small (OpenAI, best quality/cost)
  - nomic-embed-text       (Ollama-compatible, local)
  - bge-small-en-v1.5      (HuggingFace, local, fast)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.learning.layer1_logging import InteractionStore

logger = logging.getLogger(__name__)


# ── Example format ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedExample:
    interaction_id: str
    user_message: str
    response_text: str
    tool_calls_summary: str
    outcome_summary: str
    reward: float

    def to_prompt_block(self) -> str:
        """Format as a few-shot example for injection into system prompt."""
        lines = [
            f"### Example (reward={self.reward:.2f})",
            f"**Task**: {self.user_message[:200]}",
        ]
        if self.tool_calls_summary:
            lines.append(f"**Tools used**: {self.tool_calls_summary}")
        lines.append(f"**Response**: {self.response_text[:400]}")
        if self.outcome_summary:
            lines.append(f"**Outcome**: {self.outcome_summary}")
        return "\n".join(lines)


def format_examples_for_prompt(
    examples: list[RetrievedExample],
    max_chars: int = 4000,
) -> str:
    """
    Format retrieved examples as a section to append to the system prompt.
    Respects a character budget to avoid excessive context growth.
    """
    if not examples:
        return ""

    blocks = ["## High-value past examples (learn from these)\n"]
    char_count = len(blocks[0])

    for ex in examples:
        block = ex.to_prompt_block() + "\n\n"
        if char_count + len(block) > max_chars:
            break
        blocks.append(block)
        char_count += len(block)

    return "".join(blocks)


# ── Abstract vector store ─────────────────────────────────────────────────────

class VectorStore(ABC):

    @abstractmethod
    async def upsert(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


# ── ChromaDB vector store ─────────────────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """
    Local ChromaDB vector store. Good for development and single-node.
    Requires: pip install chromadb
    """

    def __init__(self, collection_name: str = "sara_experiences", path: str = ".chroma") -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection_name)

    async def upsert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._collection.upsert(ids=[id], embeddings=[vector], metadatas=[metadata])

    async def query(
        self, vector: list[float], top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter,
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            {"id": rid, **meta}
            for rid, meta in zip(results["ids"][0], results["metadatas"][0])
        ]


# ── Qdrant vector store ───────────────────────────────────────────────────────

class QdrantVectorStore(VectorStore):
    """
    Qdrant vector store. Recommended for production OpenClaw deployments.
    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "sara_experiences",
        vector_size: int = 1536,
    ) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection_name
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    async def upsert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=abs(hash(id)) % (2**63), vector=vector, payload=metadata)],
        )

    async def query(
        self, vector: list[float], top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
        )
        return [{"id": str(r.id), **r.payload, "score": r.score} for r in results]


# ── Embedder ───────────────────────────────────────────────────────────────────

class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...


class OpenAIEmbedder(Embedder):
    """text-embedding-3-small — best quality/cost ratio."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding


class OllamaEmbedder(Embedder):
    """Local embedding via Ollama — no API key needed."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> None:
        import httpx
        self._http = httpx.AsyncClient()
        self._model = model
        self._base_url = base_url

    async def embed(self, text: str) -> list[float]:
        resp = await self._http.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ── Experience Retriever ───────────────────────────────────────────────────────

class ExperienceRetriever:
    """
    Main interface for Layer 3.

    1. index_new_examples() — called by the background job after rewards computed
    2. retrieve()           — called at inference time before each Agent.chat()
    3. build_prompt_injection() — formats retrieved examples for system prompt
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        interaction_store: InteractionStore,
        min_reward: float = 0.7,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._is = interaction_store
        self._min_reward = min_reward

    async def index_new_examples(
        self,
        agent_name: str,
        modes: list[str] | None = None,
        limit: int = 200,
    ) -> int:
        """
        Fetch high-reward interactions from ClickHouse and index into vector store.
        Run periodically (e.g. hourly) via OpenClaw cron.
        """
        modes = modes or ["redteam", "judge", "admin"]
        indexed = 0
        for mode in modes:
            rows = await self._is.get_high_reward_examples(
                agent_name=agent_name, mode=mode,
                min_reward=self._min_reward, limit=limit,
            )
            for row in rows:
                text = f"{row['user_message']} {row['response_text']}"
                vector = await self._embedder.embed(text[:2000])
                await self._vs.upsert(
                    id=row["interaction_id"],
                    vector=vector,
                    metadata={
                        "agent_name": agent_name,
                        "mode": mode,
                        "user_message": row["user_message"][:500],
                        "response_text": row["response_text"][:500],
                        "outcome": json.dumps(row.get("outcome", {}))[:300],
                        "reward": row["reward"],
                    },
                )
                indexed += 1
        logger.info("Indexed %d high-reward examples into vector store", indexed)
        return indexed

    async def retrieve(
        self,
        task: str,
        mode: str,
        top_k: int = 5,
    ) -> list[RetrievedExample]:
        """Retrieve top-K most relevant high-reward examples for the current task."""
        vector = await self._embedder.embed(task[:2000])
        results = await self._vs.query(
            vector=vector,
            top_k=top_k,
            filter={"mode": mode},
        )
        examples = []
        for r in results:
            outcome_raw = r.get("outcome", "{}")
            try:
                outcome = json.loads(outcome_raw) if isinstance(outcome_raw, str) else outcome_raw
            except Exception:
                outcome = {}

            examples.append(RetrievedExample(
                interaction_id=r.get("id", ""),
                user_message=r.get("user_message", ""),
                response_text=r.get("response_text", ""),
                tool_calls_summary="",
                outcome_summary=str(outcome)[:200],
                reward=float(r.get("reward", 0)),
            ))
        return sorted(examples, key=lambda x: x.reward, reverse=True)

    async def build_prompt_injection(
        self,
        task: str,
        mode: str,
        top_k: int = 5,
        max_chars: int = 3000,
    ) -> str:
        """
        Full pipeline: retrieve examples → format for prompt injection.

        Usage in Agent.chat():
            injection = await retriever.build_prompt_injection(user_message, mode)
            system = config.get_system_prompt(mode) + injection
        """
        examples = await self.retrieve(task, mode, top_k)
        return format_examples_for_prompt(examples, max_chars=max_chars)
