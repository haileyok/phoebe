"""
ERC-8004 on-chain attestation publisher.

ERC-8004 defines a standard for publishing verifiable attestation results
as on-chain tokens. Each token represents a cryptographic proof that:

1. A specific piece of code (identified by MRENCLAVE) ran inside genuine
   Intel SGX/TDX hardware
2. The enclave processed data without exposing it to the host
3. The attestation was verified against Intel's DCAP collateral

This lets data partners (e.g. Indeed) independently verify that Phoebe's
safety classifier ran inside a TEE — without trusting Phoebe's operator.

Contract interface (ERC-8004):
    - mint(to, attestationData) → tokenId
    - verify(tokenId) → bool
    - attestationOf(tokenId) → AttestationData struct

The attestation data includes:
    - enclaveHash (bytes32): MRENCLAVE measurement
    - signerHash (bytes32): MRSIGNER measurement
    - quoteHash (bytes32): SHA-256 of the raw DCAP quote
    - timestamp (uint256): when the attestation was produced
    - metadata (bytes): arbitrary data (e.g. classification event hash)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# Minimal ERC-8004 ABI — only the functions we call
ERC8004_ABI = [
    {
        "name": "mintAttestation",
        "type": "function",
        "inputs": [
            {"name": "to", "type": "address"},
            {
                "name": "data",
                "type": "tuple",
                "components": [
                    {"name": "enclaveHash", "type": "bytes32"},
                    {"name": "signerHash", "type": "bytes32"},
                    {"name": "quoteHash", "type": "bytes32"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "metadata", "type": "bytes"},
                ],
            },
        ],
        "outputs": [{"name": "tokenId", "type": "uint256"}],
    },
    {
        "name": "verifyAttestation",
        "type": "function",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "valid", "type": "bool"}],
    },
    {
        "name": "attestationOf",
        "type": "function",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [
            {
                "name": "data",
                "type": "tuple",
                "components": [
                    {"name": "enclaveHash", "type": "bytes32"},
                    {"name": "signerHash", "type": "bytes32"},
                    {"name": "quoteHash", "type": "bytes32"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "metadata", "type": "bytes"},
                ],
            },
        ],
    },
]


@dataclass
class AttestationRecord:
    """A published ERC-8004 attestation token."""

    token_id: int
    tx_hash: str
    chain: str
    contract_address: str
    enclave_hash: str
    timestamp: int


class ERC8004Publisher:
    """
    Publishes TEE attestation results as ERC-8004 tokens on-chain.

    This provides a tamper-proof, publicly verifiable record that safety
    classifications were performed inside a genuine TEE enclave. Data
    partners can verify the token on-chain without trusting the operator.

    Supports two modes:
    - Direct: uses a local wallet (eth_account) to sign and submit txns
    - Relayer: posts attestation data to an HTTP relayer that handles gas
    """

    def __init__(
        self,
        contract_address: str,
        chain: str = "base",
        rpc_url: str = "",
        relayer_url: str = "",
        publisher_address: str = "",
        private_key: str = "",
    ) -> None:
        self._contract = contract_address
        self._chain = chain
        self._rpc_url = rpc_url
        self._relayer_url = relayer_url
        self._publisher_address = publisher_address
        self._private_key = private_key
        self._http = httpx.AsyncClient(timeout=30.0)
        self._records: list[AttestationRecord] = []

    async def publish_attestation(self, quote: Any) -> AttestationRecord:
        """
        Publish a TEE attestation quote as an ERC-8004 token.

        Args:
            quote: An AttestationQuote from the TEE classifier.

        Returns:
            AttestationRecord with the on-chain token ID and tx hash.
        """
        quote_hash = hashlib.sha256(
            quote.quote_bytes.encode() if isinstance(quote.quote_bytes, str)
            else quote.quote_bytes
        ).hexdigest()

        attestation_data = {
            "enclaveHash": _to_bytes32(quote.enclave_hash),
            "signerHash": _to_bytes32(quote.signer_hash),
            "quoteHash": _to_bytes32(quote_hash),
            "timestamp": quote.timestamp,
            "metadata": "0x",  # no extra metadata for attestation-only mints
        }

        if self._relayer_url:
            return await self._publish_via_relayer(attestation_data, quote)
        return await self._publish_direct(attestation_data, quote)

    async def record_classification_event(
        self,
        session: Any,
        category: str,
        result_hash: str,
    ) -> AttestationRecord | None:
        """
        Record a classification event as metadata on an existing attestation,
        or mint a new lightweight attestation token with event data.

        This creates an on-chain audit trail: for each classification batch,
        there's a token proving it happened inside the attested enclave.
        The result_hash is a SHA-256 of the classification output — the
        actual content stays private, but the hash is publicly verifiable.
        """
        metadata = json.dumps({
            "type": "classification_event",
            "category": category,
            "result_hash": result_hash,
            "session_id": session.session_id,
            "enclave_hash": session.attestation.enclave_hash,
            "timestamp": int(time.time()),
        }).encode().hex()

        attestation_data = {
            "enclaveHash": _to_bytes32(session.attestation.enclave_hash),
            "signerHash": _to_bytes32(session.attestation.signer_hash),
            "quoteHash": _to_bytes32(result_hash),
            "timestamp": int(time.time()),
            "metadata": f"0x{metadata}",
        }

        if self._relayer_url:
            return await self._publish_via_relayer(attestation_data, session.attestation)
        return await self._publish_direct(attestation_data, session.attestation)

    async def verify_attestation(self, token_id: int) -> bool:
        """
        Verify an ERC-8004 attestation token on-chain.

        Calls the contract's verifyAttestation(tokenId) to check that:
        - The token exists and hasn't been revoked
        - The attestation data matches the original DCAP quote
        """
        if self._relayer_url:
            resp = await self._http.get(
                f"{self._relayer_url}/verify/{token_id}",
            )
            resp.raise_for_status()
            return bool(resp.json().get("valid", False))

        # Direct RPC call (eth_call)
        return await self._eth_call_verify(token_id)

    async def get_attestation(self, token_id: int) -> dict[str, Any]:
        """Read attestation data for a given token from the chain."""
        if self._relayer_url:
            resp = await self._http.get(
                f"{self._relayer_url}/attestation/{token_id}",
            )
            resp.raise_for_status()
            return resp.json()

        # Direct RPC read
        return await self._eth_call_attestation_of(token_id)

    @property
    def records(self) -> list[AttestationRecord]:
        """All attestation records published during this session."""
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal: relayer mode
    # ------------------------------------------------------------------

    async def _publish_via_relayer(
        self, attestation_data: dict, quote: Any
    ) -> AttestationRecord:
        """Submit attestation to an HTTP relayer that handles gas and signing."""
        resp = await self._http.post(
            f"{self._relayer_url}/mint",
            json={
                "contract": self._contract,
                "chain": self._chain,
                "to": self._publisher_address,
                "attestation": attestation_data,
            },
        )
        resp.raise_for_status()
        result = resp.json()

        record = AttestationRecord(
            token_id=result["tokenId"],
            tx_hash=result["txHash"],
            chain=self._chain,
            contract_address=self._contract,
            enclave_hash=quote.enclave_hash,
            timestamp=attestation_data["timestamp"],
        )
        self._records.append(record)
        logger.info(
            "ERC-8004 attestation minted: token=%d tx=%s",
            record.token_id, record.tx_hash[:16],
        )
        return record

    # ------------------------------------------------------------------
    # Internal: direct on-chain mode
    # ------------------------------------------------------------------

    async def _publish_direct(
        self, attestation_data: dict, quote: Any
    ) -> AttestationRecord:
        """
        Publish directly via JSON-RPC (eth_sendTransaction).

        Encodes the mintAttestation call data and sends it to the RPC endpoint.
        """
        if not self._rpc_url:
            raise RuntimeError("No RPC URL configured for direct ERC-8004 publishing")
        if not self._private_key:
            raise RuntimeError("No private key configured for direct ERC-8004 publishing")

        from eth_account import Account

        account = Account.from_key(self._private_key)

        # Encode the mintAttestation function call
        call_data = _encode_mint_attestation(
            to=self._publisher_address or account.address,
            attestation=attestation_data,
        )

        # Get nonce
        nonce_resp = await self._http.post(
            self._rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_getTransactionCount",
                "params": [account.address, "latest"],
                "id": 1,
            },
        )
        nonce = int(nonce_resp.json()["result"], 16)

        # Build and sign transaction
        tx = {
            "to": self._contract,
            "data": call_data,
            "nonce": nonce,
            "gas": 200_000,
            "maxFeePerGas": 1_000_000_000,  # 1 gwei — adjust for chain
            "maxPriorityFeePerGas": 100_000_000,
            "chainId": _chain_id(self._chain),
            "type": 2,
        }

        signed = account.sign_transaction(tx)

        # Send
        send_resp = await self._http.post(
            self._rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_sendRawTransaction",
                "params": [signed.raw_transaction.hex()],
                "id": 2,
            },
        )
        tx_hash = send_resp.json()["result"]

        record = AttestationRecord(
            token_id=0,  # resolved after receipt
            tx_hash=tx_hash,
            chain=self._chain,
            contract_address=self._contract,
            enclave_hash=quote.enclave_hash,
            timestamp=attestation_data["timestamp"],
        )
        self._records.append(record)
        logger.info("ERC-8004 tx submitted: %s", tx_hash)
        return record

    async def _eth_call_verify(self, token_id: int) -> bool:
        """Call verifyAttestation(uint256) as a read-only eth_call."""
        # Function selector: keccak256("verifyAttestation(uint256)")[:4]
        selector = "0x1f3302a9"
        encoded_id = hex(token_id)[2:].zfill(64)

        resp = await self._http.post(
            self._rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [
                    {"to": self._contract, "data": f"{selector}{encoded_id}"},
                    "latest",
                ],
                "id": 1,
            },
        )
        result = resp.json().get("result", "0x0")
        return int(result, 16) == 1

    async def _eth_call_attestation_of(self, token_id: int) -> dict[str, Any]:
        """Call attestationOf(uint256) as a read-only eth_call."""
        selector = "0x8c2a993e"
        encoded_id = hex(token_id)[2:].zfill(64)

        resp = await self._http.post(
            self._rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [
                    {"to": self._contract, "data": f"{selector}{encoded_id}"},
                    "latest",
                ],
                "id": 1,
            },
        )
        raw = resp.json().get("result", "0x")
        # Return raw for now — full ABI decoding would use eth_abi
        return {"raw": raw, "token_id": token_id, "contract": self._contract}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_bytes32(hex_str: str) -> str:
    """Normalize a hex string to a 0x-prefixed bytes32."""
    clean = hex_str.replace("0x", "")
    # If it's not already a hex hash, SHA-256 it
    if len(clean) != 64:
        clean = hashlib.sha256(clean.encode()).hexdigest()
    return f"0x{clean}"


def _chain_id(chain: str) -> int:
    """Map chain name to EIP-155 chain ID."""
    return {
        "base": 8453,
        "ethereum": 1,
        "polygon": 137,
        "arbitrum": 42161,
        "optimism": 10,
        "base-sepolia": 84532,
        "sepolia": 11155111,
    }.get(chain, 8453)


def _encode_mint_attestation(to: str, attestation: dict) -> str:
    """
    ABI-encode a mintAttestation(address, AttestationData) call.

    This is a simplified encoder — in production use eth_abi.encode.
    """
    # Function selector: keccak256("mintAttestation(address,(bytes32,bytes32,bytes32,uint256,bytes))")[:4]
    selector = "0xa1448194"

    # Pad address
    addr = to.replace("0x", "").zfill(64)

    # Offset to tuple data (64 bytes = 0x40)
    tuple_offset = "0" * 62 + "40"

    # Tuple fields (each 32 bytes except metadata which is dynamic)
    enclave = attestation["enclaveHash"].replace("0x", "")
    signer = attestation["signerHash"].replace("0x", "")
    quote = attestation["quoteHash"].replace("0x", "")
    ts = hex(attestation["timestamp"])[2:].zfill(64)

    # Metadata offset (5 * 32 = 160 = 0xa0)
    meta_offset = "0" * 62 + "a0"

    # Metadata bytes
    meta_hex = attestation.get("metadata", "0x").replace("0x", "")
    meta_len = hex(len(meta_hex) // 2)[2:].zfill(64)
    # Pad to 32 bytes
    meta_padded = meta_hex + "0" * (64 - len(meta_hex) % 64) if meta_hex else ""

    return (
        selector + addr + tuple_offset
        + enclave + signer + quote + ts + meta_offset
        + meta_len + meta_padded
    )
