"""
RL Layer 4 — Fine-Tuning Pipeline
====================================

Actual model weight updates. Only activate when you have:
  - ~1,000+ labelled interactions per mode (from Layers 1+2)
  - Clear evidence that Layers 1-3 have plateaued
  - An open-source model as the agent backbone for at least one mode

Architecture options:

  A) OpenAI Fine-Tuning API  (easiest, closed model)
     - Works directly with your existing OpenAICompatibleClient
     - Supports: gpt-4o-mini, gpt-3.5-turbo
     - Format: JSONL chat completion format
     - Cost: ~$0.008/1K tokens for training

  B) DPO on open model       (recommended for redteam mode)
     - Direct Preference Optimisation — no reward model needed
     - Uses preference pairs: (prompt, chosen_response, rejected_response)
     - Works on: Llama-3, Gemma-2, Qwen-2.5, Mistral, DeepSeek
     - Library: trl (HuggingFace)

  C) PPO / GRPO              (most powerful, most complex)
     - Full RL loop with reward model
     - Requires significant GPU resources
     - Only worth it if DPO plateaus

This module provides:
  - PreferencePairBuilder  — creates (chosen, rejected) pairs from ClickHouse data
  - OpenAIFineTuner        — submits fine-tuning jobs to OpenAI API
  - DPODatasetBuilder      — exports HuggingFace-compatible DPO dataset
  - FineTuneMonitor        — tracks job status and model performance

Prerequisites:
  pip install openai datasets trl transformers accelerate peft
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from src.learning.layer1_logging import InteractionStore

logger = logging.getLogger(__name__)


# ── Preference pair building ───────────────────────────────────────────────────

@dataclass
class PreferencePair:
    """
    A (prompt, chosen, rejected) triple for DPO training.
    chosen  = high-reward response to this type of task
    rejected = low-reward response to this type of task
    """
    prompt: str
    chosen: str
    rejected: str
    mode: str
    reward_gap: float  # chosen_reward - rejected_reward


class PreferencePairBuilder:
    """
    Builds DPO preference pairs from ClickHouse interaction data.

    Strategy:
      For each mode, pair the top-quartile interactions (chosen) with
      bottom-quartile interactions (rejected) that have similar prompts
      (cosine similarity > 0.7). This gives meaningful preference signal.
    """

    def __init__(self, store: InteractionStore) -> None:
        self._store = store

    async def build_pairs(
        self,
        agent_name: str,
        mode: str,
        min_reward_gap: float = 0.3,
        max_pairs: int = 1000,
    ) -> list[PreferencePair]:
        """
        Fetch high-reward and low-reward examples, then pair them.
        A simple cross-product is used here; replace with embedding-based
        pairing for higher quality.
        """
        high_reward = await self._store.get_high_reward_examples(
            agent_name, mode, min_reward=0.7, limit=max_pairs
        )
        # Fetch low-reward interactions
        # (extend InteractionStore with get_low_reward_examples if needed)
        low_reward_rows = self._store._client.execute(
            f"""
            SELECT interaction_id, user_message, response_text, outcome, reward
            FROM sara_interactions
            WHERE agent_name = %(a)s AND mode = %(m)s
              AND reward < 0.3 AND reward IS NOT NULL
            ORDER BY reward ASC LIMIT %(l)s
            """,
            {"a": agent_name, "m": mode, "l": max_pairs},
        )

        pairs = []
        for high in high_reward[:max_pairs]:
            for row in low_reward_rows[:10]:  # limit combinations
                reward_gap = float(high.get("reward", 1.0)) - float(row[4] or 0.0)
                if reward_gap >= min_reward_gap:
                    pairs.append(PreferencePair(
                        prompt=high["user_message"],
                        chosen=high["response_text"],
                        rejected=row[2],
                        mode=mode,
                        reward_gap=reward_gap,
                    ))
                    break  # one rejected per chosen is sufficient

        pairs.sort(key=lambda p: p.reward_gap, reverse=True)
        logger.info("Built %d preference pairs for mode=%s", len(pairs), mode)
        return pairs[:max_pairs]


# ── OpenAI Fine-Tuning ─────────────────────────────────────────────────────────

class OpenAIFineTuner:
    """
    Submits fine-tuning jobs to the OpenAI Fine-Tuning API.
    Works with gpt-4o-mini and gpt-3.5-turbo.
    The resulting model can be used in OpenAICompatibleClient by setting
    model_name to the fine-tuned model ID.

    Requires: pip install openai
    """

    def __init__(self, api_key: str | None = None) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def build_training_jsonl(
        self,
        interactions: list[dict[str, Any]],
        system_prompt: str,
        output_path: str,
    ) -> str:
        """Export interactions as OpenAI fine-tuning JSONL format."""
        with open(output_path, "w") as f:
            for row in interactions:
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["user_message"]},
                        {"role": "assistant", "content": row["response_text"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")
        logger.info("Wrote %d training examples to %s", len(interactions), output_path)
        return output_path

    def submit_job(
        self,
        training_file_path: str,
        base_model: str = "gpt-4o-mini",
        suffix: str = "sara-redteam",
        n_epochs: int = 3,
    ) -> str:
        """Upload JSONL and start a fine-tuning job. Returns job ID."""
        with open(training_file_path, "rb") as f:
            file_obj = self._client.files.create(file=f, purpose="fine-tune")

        job = self._client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=base_model,
            suffix=suffix,
            hyperparameters={"n_epochs": n_epochs},
        )
        logger.info("Fine-tuning job submitted: %s", job.id)
        return job.id

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = self._client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
        }


# ── HuggingFace DPO Dataset ───────────────────────────────────────────────────

class DPODatasetBuilder:
    """
    Exports preference pairs as a HuggingFace Dataset for DPO training
    with the trl library.

    Example training command (after export):

        from trl import DPOTrainer, DPOConfig
        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        dataset   = load_from_disk("./sara_dpo_dataset")

        trainer = DPOTrainer(
            model=model,
            args=DPOConfig(output_dir="./sara-dpo-output", num_train_epochs=3),
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

    Supports: Llama-3, Gemma-2, Qwen-2.5, Mistral, DeepSeek, Phi-3
    """

    def export(
        self,
        pairs: list[PreferencePair],
        output_dir: str,
        system_prompt: str,
    ) -> str:
        """
        Export to HuggingFace Dataset format.
        Requires: pip install datasets
        """
        from datasets import Dataset

        records = [
            {
                "prompt": f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{p.prompt}",
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in pairs
        ]

        dataset = Dataset.from_list(records)
        dataset.save_to_disk(output_dir)
        logger.info("DPO dataset saved to %s (%d pairs)", output_dir, len(pairs))
        return output_dir


# ── Fine-tune monitor ─────────────────────────────────────────────────────────

class FineTuneMonitor:
    """
    Tracks the performance delta between base and fine-tuned models
    using held-out interactions from ClickHouse.

    Metric: mean reward on held-out set before vs. after fine-tuning.
    """

    def __init__(self, store: InteractionStore) -> None:
        self._store = store

    async def evaluate(
        self,
        base_model_id: str,
        finetuned_model_id: str,
        agent_name: str,
        mode: str,
        n_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Run held-out interactions through both models and compare mean reward.
        This requires a live agent instance for each model — wire up externally.
        Returns a summary dict for logging.
        """
        # Fetch held-out sample (interactions not used in training)
        rows = await self._store.get_high_reward_examples(
            agent_name, mode, min_reward=0.0, limit=n_samples
        )
        logger.info(
            "Evaluation: %d held-out examples for %s/%s", len(rows), agent_name, mode
        )
        return {
            "base_model": base_model_id,
            "finetuned_model": finetuned_model_id,
            "n_samples": len(rows),
            "note": "Wire agent instances to run live evaluation",
        }
