"""
Minimal wallet abstraction for signing x402 payments.

In production this would use a proper crypto signing library (e.g. eth_account
for EVM chains, solders for Solana). This prototype provides the interface and
uses a placeholder signer so the rest of the system can be built and tested
end-to-end without requiring real chain access.
"""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass


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

    def to_header(self) -> str:
        """Encode as base64 JSON for the X-PAYMENT header."""
        import base64

        payload = json.dumps(
            {
                "chain": self.chain,
                "token": self.token,
                "amount": self.amount,
                "recipient": self.recipient,
                "sender": self.sender,
                "signature": self.signature,
                "timestamp": self.timestamp,
            }
        )
        return base64.b64encode(payload.encode()).decode()


class Wallet:
    """
    Wallet for signing x402 payments.

    In a real deployment, `private_key` would be an EVM/Solana private key
    and `sign_payment` would produce a real on-chain transaction signature.
    This prototype uses HMAC as a placeholder.
    """

    def __init__(self, private_key: str, address: str, chain: str = "base") -> None:
        self._private_key = private_key
        self._address = address
        self._chain = chain

    @property
    def address(self) -> str:
        return self._address

    @property
    def chain(self) -> str:
        return self._chain

    def sign_payment(
        self,
        recipient: str,
        amount: str,
        token: str = "USDC",
    ) -> SignedPayment:
        """
        Sign a payment. In production this creates a real blockchain transaction.
        For the prototype, we produce an HMAC signature that a local facilitator
        can verify.
        """
        ts = int(time.time())
        message = f"{self._chain}:{token}:{amount}:{recipient}:{self._address}:{ts}"
        sig = hmac.new(
            self._private_key.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

        return SignedPayment(
            chain=self._chain,
            token=token,
            amount=amount,
            recipient=recipient,
            sender=self._address,
            signature=sig,
            timestamp=ts,
        )
