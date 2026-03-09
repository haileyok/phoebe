"""
Self-contained configuration for the Phala TEE + ERC-8004 feature.

All TEE/attestation config lives here — not in the main Config class —
so the feature stays on its own branch without touching core code.

Usage:
    from src.safety.tee_config import TEE_CONFIG, build_tee_classifier

    # Wrap an existing SafetyClassifier with TEE if enabled
    classifier = build_tee_classifier(base_classifier)
"""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.safety.classifier import SafetyClassifier

logger = logging.getLogger(__name__)


class TEEConfig(BaseSettings):
    """Configuration for Phala TEE proxy and ERC-8004 attestation."""

    # Phala TEE
    tee_enabled: bool = False
    """route safety classifications through Phala TEE enclave"""
    tee_endpoint: str = ""
    """Phala TEE worker endpoint (e.g. https://your-worker.phala.network)"""
    tee_verify_attestation: bool = True
    """verify Intel DCAP attestation before sending data to enclave"""
    tee_fallback_on_failure: bool = True
    """fall back to direct classifier if TEE is unreachable"""

    # ERC-8004 on-chain attestation
    erc8004_enabled: bool = False
    """publish TEE attestation results as ERC-8004 tokens on-chain"""
    erc8004_contract: str = ""
    """ERC-8004 contract address"""
    erc8004_chain: str = "base"
    """chain for ERC-8004 attestation tokens"""
    erc8004_rpc_url: str = ""
    """JSON-RPC endpoint for direct on-chain publishing"""
    erc8004_relayer_url: str = ""
    """HTTP relayer URL (alternative to direct RPC, handles gas)"""
    erc8004_publisher_address: str = ""
    """address to receive minted attestation tokens"""
    erc8004_private_key: str = ""
    """private key for signing ERC-8004 mint transactions (can reuse x402 key)"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


TEE_CONFIG = TEEConfig()


def build_tee_classifier(base: SafetyClassifier) -> SafetyClassifier:
    """
    Wrap a SafetyClassifier with TEE proxy if TEE is enabled.

    Returns the base classifier unchanged if TEE is disabled.
    This is the only integration point — call it from your own code
    or monkey-patch build_safety_classifier() in main.py.
    """
    if not TEE_CONFIG.tee_enabled or not TEE_CONFIG.tee_endpoint:
        return base

    # Build optional ERC-8004 publisher
    erc8004_publisher = None
    if TEE_CONFIG.erc8004_enabled and TEE_CONFIG.erc8004_contract:
        from src.safety.erc8004 import ERC8004Publisher

        erc8004_publisher = ERC8004Publisher(
            contract_address=TEE_CONFIG.erc8004_contract,
            chain=TEE_CONFIG.erc8004_chain,
            rpc_url=TEE_CONFIG.erc8004_rpc_url,
            relayer_url=TEE_CONFIG.erc8004_relayer_url,
            publisher_address=TEE_CONFIG.erc8004_publisher_address,
            private_key=TEE_CONFIG.erc8004_private_key,
        )

    from src.safety.tee_classifier import TEEClassifier

    logger.info("TEE classifier enabled — endpoint: %s", TEE_CONFIG.tee_endpoint)
    return TEEClassifier(
        inner=base,
        tee_endpoint=TEE_CONFIG.tee_endpoint,
        verify_attestation=TEE_CONFIG.tee_verify_attestation,
        fallback_on_failure=TEE_CONFIG.tee_fallback_on_failure,
        erc8004_publisher=erc8004_publisher,
    )
