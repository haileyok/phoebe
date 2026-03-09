from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # clickhouse config
    clickhouse_host: str = "localhost"
    """host for the clickhouse server"""
    clickhouse_port: int = 8123
    """port for the clickhouse server"""
    clickhouse_user: str = "default"
    """username for the clickhouse server"""
    clickhouse_password: str = "clickhouse"
    """password for the clickhouse server"""
    clickhouse_database: str = "default"
    """default database for the clickhouse server"""

    # model config — used for Phoebe's own LLM reasoning + safety classifier
    model_api: Literal["anthropic", "openai", "openapi"] = "anthropic"
    """the model api to use. must be one of `anthropic`, `openai`, or `openapi`"""
    model_name: str = "claude-sonnet-4-5-20250929"
    """the model to use with the given api"""
    model_api_key: str = ""
    """the model api key"""
    model_endpoint: str = ""
    """for openapi model apis, the endpoint to use"""

    # x402 payment config
    x402_wallet_private_key: str = ""
    """private key for signing x402 USDC payments (EVM or Solana)"""
    x402_wallet_address: str = ""
    """wallet address for x402 payments"""
    x402_chain: str = "base"
    """blockchain to use for x402 payments (base, solana, etc.)"""
    x402_facilitator_url: str = ""
    """x402 facilitator URL for payment settlement"""
    x402_max_auto_pay: float = 1.0
    """maximum USDC amount to auto-pay per x402 request"""

    # arena config
    arena_host: str = "0.0.0.0"
    """host for the arena HTTP server"""
    arena_port: int = 8080
    """port for the arena HTTP server"""
    arena_submission_fee: float = 0.01
    """USDC fee per attack submission (anti-spam)"""
    arena_scoring_alpha: float = 0.4
    """scoring weight for attack success"""
    arena_scoring_beta: float = 0.3
    """scoring weight for novelty"""
    arena_scoring_gamma: float = 0.2
    """scoring weight for category coverage"""
    arena_scoring_delta: float = 0.1
    """scoring weight for duplicate penalty"""
    arena_payout_rate: float = 1.0
    """score-to-USDC multiplier for payouts"""
    arena_wallet: str = "arena.sandbox.eth"
    """arena's wallet address for receiving bounty funds and submission fees"""
    arena_dev_mode: bool = True
    """run arena in dev mode (accept DevWallet HMAC signatures, skip real EVM verification)"""

    # spending limits
    x402_spending_limit: float = 100.0
    """cumulative USDC spending limit for x402 client"""

    # safety classifier config
    safety_classifier_model: str = "claude-sonnet-4-5-20250929"
    """model to use for the LLM-as-judge safety classifier"""
    safety_classifier_endpoint: str = "https://api.anthropic.com"
    """API endpoint for the safety classifier"""

    model_config = SettingsConfigDict(env_file=".env")


CONFIG = Config()
