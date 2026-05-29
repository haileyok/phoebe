"""
Tests for new model providers: Kimi (Moonshot), GLM (Zhipu), DeepSeek.

All tests run without live API calls — only config and client construction.
"""

import pytest

from src.agent.agent import AnthropicClient, OpenAICompatibleClient, _build_client
from src.agent.config import ModelConfig


# ── ModelConfig classmethods ──────────────────────────────────────────────────


def test_kimi_config_defaults():
    mc = ModelConfig.kimi()
    assert mc.provider == "kimi"
    assert mc.model_name == "kimi-k2"
    assert mc.api_key_env == "MOONSHOT_API_KEY"
    assert "moonshot.ai" in mc.endpoint


def test_kimi_config_custom_model():
    mc = ModelConfig.kimi("kimi-k2.5")
    assert mc.model_name == "kimi-k2.5"
    assert mc.provider == "kimi"


def test_glm_config_defaults():
    mc = ModelConfig.glm()
    assert mc.provider == "glm"
    assert mc.model_name == "glm-4.5"
    assert mc.api_key_env == "ZHIPU_API_KEY"
    assert "bigmodel.cn" in mc.endpoint


def test_glm_config_flash():
    mc = ModelConfig.glm("glm-4.5-flash")
    assert mc.model_name == "glm-4.5-flash"
    assert mc.provider == "glm"


def test_deepseek_config_defaults():
    mc = ModelConfig.deepseek()
    assert mc.provider == "deepseek"
    assert mc.model_name == "deepseek-v4-flash"
    assert mc.api_key_env == "DEEPSEEK_API_KEY"
    assert "deepseek.com" in mc.endpoint


def test_deepseek_config_pro():
    mc = ModelConfig.deepseek("deepseek-v4-pro")
    assert mc.model_name == "deepseek-v4-pro"
    assert mc.provider == "deepseek"


# ── _build_client() returns correct client type ───────────────────────────────


def test_build_client_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    mc = ModelConfig.anthropic()
    client = _build_client(mc)
    assert isinstance(client, AnthropicClient)


def test_build_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    mc = ModelConfig.openai()
    client = _build_client(mc)
    assert isinstance(client, OpenAICompatibleClient)
    assert "openai.com" in client._endpoint


def test_build_client_kimi(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-kimi-key")
    mc = ModelConfig.kimi()
    client = _build_client(mc)
    assert isinstance(client, OpenAICompatibleClient)
    assert "moonshot.ai" in client._endpoint
    assert client._model_name == "kimi-k2"


def test_build_client_glm(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "test-zhipu-key")
    mc = ModelConfig.glm()
    client = _build_client(mc)
    assert isinstance(client, OpenAICompatibleClient)
    assert "bigmodel.cn" in client._endpoint
    assert client._model_name == "glm-4.5"


def test_build_client_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-ds-key")
    mc = ModelConfig.deepseek()
    client = _build_client(mc)
    assert isinstance(client, OpenAICompatibleClient)
    assert "deepseek.com" in client._endpoint
    assert client._model_name == "deepseek-v4-flash"


def test_build_client_custom_openapi(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "test-custom-key")
    mc = ModelConfig.custom("my-model", "https://custom.api/v1", "CUSTOM_API_KEY")
    client = _build_client(mc)
    assert isinstance(client, OpenAICompatibleClient)
    assert client._endpoint == "https://custom.api/v1"


def test_build_client_unknown_provider():
    mc = ModelConfig(
        provider="unknown_provider",
        model_name="some-model",
        api_key_env="SOME_KEY",
        endpoint="https://example.com",
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        _build_client(mc)


def test_build_client_openapi_missing_endpoint():
    mc = ModelConfig(
        provider="openapi",
        model_name="some-model",
        api_key_env="SOME_KEY",
        endpoint=None,
    )
    with pytest.raises(ValueError, match="endpoint is required"):
        _build_client(mc)


# ── Message format conversion (shared across all OAI-compat providers) ────────


def test_convert_messages_tool_use_kimi():
    """Kimi uses OAI tool_calls format — verify conversion is correct."""
    client = OpenAICompatibleClient("key", "kimi-k2", "https://api.moonshot.ai/v1")
    msgs = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking..."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "execute_code",
                    "input": {"code": "print(1)"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "1"}
            ],
        },
    ]
    result = client._convert_messages(msgs, "system prompt")
    # Verify system message
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "system prompt"
    # Verify user message
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "test"
    # Verify assistant message with tool_calls
    assistant_msg = result[2]
    assert assistant_msg["role"] == "assistant"
    assert "tool_calls" in assistant_msg
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "execute_code"
    # Verify tool result message
    tool_msg = result[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "t1"
    assert tool_msg["content"] == "1"


def test_convert_messages_reasoning_content_deepseek():
    """DeepSeek thinking mode returns reasoning_content — verify it passes through."""
    client = OpenAICompatibleClient("key", "deepseek-v4-pro", "https://api.deepseek.com")
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "final answer"}],
            "reasoning_content": "step-by-step reasoning",
        },
    ]
    result = client._convert_messages(msgs, "system")
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_content"] == "step-by-step reasoning"
    assert assistant_msg["content"] == "final answer"
