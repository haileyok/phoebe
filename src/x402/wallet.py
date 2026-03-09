"""
Production wallet for signing x402 payments on EVM chains.

Uses eth_account for real EIP-191 message signing and EIP-712 typed data.
The signed authorization is what the x402 facilitator verifies before
settling the USDC transfer on-chain.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass

from eth_account import Account
from eth_account.messages import encode_defunct

logger = logging.getLogger(__name__)


@dataclass
class SignedPayment:
    """A signed x402 payment ready to be sent as an X-PAYMENT header."""

    chain: str
    token: str
    amount: str
    recipient: str
    sender: str
    signature: str
    timestamp: int
    nonce: int

    def to_header(self) -> str:
        """Encode as base64 JSON for the X-PAYMENT header."""
        payload = json.dumps(
            {
                "chain": self.chain,
                "token": self.token,
                "amount": self.amount,
                "recipient": self.recipient,
                "sender": self.sender,
                "signature": self.signature,
                "timestamp": self.timestamp,
                "nonce": self.nonce,
            }
        )
        return base64.b64encode(payload.encode()).decode()


# USDC contract addresses per chain
USDC_CONTRACTS: dict[str, str] = {
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "polygon": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    "optimism": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
}


class Wallet:
    """
    EVM wallet for signing x402 payments.

    Signs EIP-191 personal messages that the x402 facilitator verifies
    on-chain. The facilitator then executes the USDC transfer from the
    sender's pre-approved allowance.

    Requires: The wallet must have pre-approved USDC spending allowance
    to the x402 facilitator contract.
    """

    def __init__(self, private_key: str, chain: str = "base") -> None:
        if not private_key or private_key in ("dev-key", ""):
            raise ValueError(
                "A real EVM private key is required. "
                "Set X402_WALLET_PRIVATE_KEY in .env"
            )

        self._account = Account.from_key(private_key)
        self._chain = chain
        self._nonce = int(time.time() * 1000)  # monotonic nonce

    @property
    def address(self) -> str:
        return self._account.address

    @property
    def chain(self) -> str:
        return self._chain

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def sign_payment(
        self,
        recipient: str,
        amount: str,
        token: str = "USDC",
    ) -> SignedPayment:
        """
        Sign a payment authorization.

        Produces an EIP-191 personal_sign over a canonical message that
        the x402 facilitator can verify to authorize the USDC transfer.
        """
        ts = int(time.time())
        nonce = self._next_nonce()

        # canonical message format per x402 spec
        usdc_contract = USDC_CONTRACTS.get(self._chain, "0x0")
        message = (
            f"x402 Payment Authorization\n"
            f"Chain: {self._chain}\n"
            f"Token: {usdc_contract}\n"
            f"Amount: {amount}\n"
            f"Recipient: {recipient}\n"
            f"Sender: {self._account.address}\n"
            f"Timestamp: {ts}\n"
            f"Nonce: {nonce}"
        )

        signable = encode_defunct(text=message)
        signed = self._account.sign_message(signable)

        logger.debug(
            "Signed x402 payment: %s %s to %s (nonce=%d)",
            amount, token, recipient[:16], nonce,
        )

        return SignedPayment(
            chain=self._chain,
            token=token,
            amount=amount,
            recipient=recipient,
            sender=self._account.address,
            signature=signed.signature.hex(),
            timestamp=ts,
            nonce=nonce,
        )


class DevWallet:
    """
    Development-mode wallet that uses HMAC signing instead of real EVM keys.

    Use this for local testing without real funds. The arena server's
    _verify_payment will accept these in dev mode.
    """

    def __init__(self, address: str = "0xdev", chain: str = "base") -> None:
        import hashlib
        import hmac as _hmac

        self._address = address
        self._chain = chain
        self._hmac = _hmac
        self._hashlib = hashlib
        self._nonce = int(time.time() * 1000)

    @property
    def address(self) -> str:
        return self._address

    @property
    def chain(self) -> str:
        return self._chain

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def sign_payment(
        self,
        recipient: str,
        amount: str,
        token: str = "USDC",
    ) -> SignedPayment:
        ts = int(time.time())
        nonce = self._next_nonce()
        message = f"{self._chain}:{token}:{amount}:{recipient}:{self._address}:{ts}:{nonce}"
        sig = self._hmac.new(
            b"dev-key", message.encode(), self._hashlib.sha256
        ).hexdigest()

        return SignedPayment(
            chain=self._chain,
            token=token,
            amount=amount,
            recipient=recipient,
            sender=self._address,
            signature=sig,
            timestamp=ts,
            nonce=nonce,
        )
