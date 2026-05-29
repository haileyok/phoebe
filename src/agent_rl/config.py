"""
Agent configuration — separates persona from framework.

Each agent directory (agents/<name>/config.py) defines an AgentConfig.
The Agent class consumes this config and knows nothing about Phoebe, Sara,
or any other specific persona.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Provider + model routing for any OpenAI-compatible or Anthropic endpoint."""

    provider: str  # "anthropic" | "openai" | "openapi"
    model_name: str
    api_key_env: str  # env var name, e.g. "ANTHROPIC_API_KEY"
    endpoint: str | None = None  # required for "openapi" provider

    # Well-known provider presets — override endpoint/model_name as needed
    PRESETS: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @classmethod
    def anthropic(cls, model: str = "claude-sonnet-4-5-20250929") -> "ModelConfig":
        return cls(provider="anthropic", model_name=model, api_key_env="ANTHROPIC_API_KEY")

    @classmethod
    def openai(cls, model: str = "gpt-4o") -> "ModelConfig":
        return cls(provider="openai", model_name=model, api_key_env="OPENAI_API_KEY")

    @classmethod
    def kimi(cls, model: str = "moonshot-v1-8k") -> "ModelConfig":
        return cls(
            provider="openapi",
            model_name=model,
            api_key_env="MOONSHOT_API_KEY",
            endpoint="https://api.moonshot.cn/v1",
        )

    @classmethod
    def qwen(cls, model: str = "qwen-plus") -> "ModelConfig":
        return cls(
            provider="openapi",
            model_name=model,
            api_key_env="DASHSCOPE_API_KEY",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @classmethod
    def gemini(cls, model: str = "gemini-2.0-flash") -> "ModelConfig":
        return cls(
            provider="openapi",
            model_name=model,
            api_key_env="GEMINI_API_KEY",
            endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
        )

    @classmethod
    def deepseek(cls, model: str = "deepseek-chat") -> "ModelConfig":
        return cls(
            provider="openapi",
            model_name=model,
            api_key_env="DEEPSEEK_API_KEY",
            endpoint="https://api.deepseek.com/v1",
        )

    @classmethod
    def ollama(cls, model: str = "llama3.2") -> "ModelConfig":
        """Local Ollama — no API key required."""
        return cls(
            provider="openapi",
            model_name=model,
            api_key_env="OLLAMA_API_KEY",  # set to "ollama" or any string
            endpoint="http://localhost:11434/v1",
        )

    @classmethod
    def custom(
        cls,
        model: str,
        endpoint: str,
        api_key_env: str,
    ) -> "ModelConfig":
        """Any OpenAI-compatible endpoint."""
        return cls(
            provider="openapi",
            model_name=model,
            api_key_env=api_key_env,
            endpoint=endpoint,
        )


@dataclass
class AgentConfig:
    """
    Full persona definition for a named agent.

    Separate this from the Agent runtime class so that:
    - New agents need only a config file, not code changes
    - The same framework can host Phoebe, Sara, a governance bot, etc.
    - Modes map to system prompts without any hardcoded if/elif chains
    """

    name: str
    description: str
    default_mode: str
    modes: dict[str, str]           # mode_name -> system prompt string
    default_model: ModelConfig
    tool_tags: list[str] = field(default_factory=list)   # e.g. ["code", "search", "memory"]
    max_tokens: int = 16_000
    max_tool_result_length: int = 10_000
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_system_prompt(self, mode: str | None = None) -> str:
        m = mode or self.default_mode
        if m not in self.modes:
            raise ValueError(
                f"Agent '{self.name}' has no mode '{m}'. "
                f"Available: {list(self.modes.keys())}"
            )
        return self.modes[m]

    def available_modes(self) -> list[str]:
        return list(self.modes.keys())
