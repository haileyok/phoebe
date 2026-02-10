"""
x402-aware HTTP client.

Wraps httpx.AsyncClient to transparently handle HTTP 402 Payment Required
responses per the x402 protocol:

  1. Client makes a normal request.
  2. Server returns 402 + PAYMENT-REQUIRED header with payment terms.
  3. Client signs a payment via its Wallet.
  4. Client resends the original request with the X-PAYMENT header.
  5. Server settles payment (via facilitator) and returns 200.

This allows any outbound HTTP call in Phoebe to auto-pay via x402 with zero
call-site changes — just swap `httpx.AsyncClient` for `X402Client`.
"""

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from src.x402.wallet import Wallet

logger = logging.getLogger(__name__)


class X402PaymentError(Exception):
    """Raised when an x402 payment fails."""

    def __init__(self, message: str, cost: str | None = None) -> None:
        super().__init__(message)
        self.cost = cost


@dataclass
class PaymentTerms:
    """Parsed from the PAYMENT-REQUIRED header on a 402 response."""

    recipient: str
    amount: str
    token: str
    chain: str
    description: str
    facilitator_url: str

    @classmethod
    def from_header(cls, header_value: str) -> "PaymentTerms":
        """Parse the base64-encoded PAYMENT-REQUIRED header."""
        try:
            decoded = base64.b64decode(header_value)
            data = json.loads(decoded)
        except Exception as e:
            raise X402PaymentError(f"Invalid PAYMENT-REQUIRED header: {e}")

        return cls(
            recipient=data.get("recipient", ""),
            amount=data.get("amount", "0"),
            token=data.get("token", "USDC"),
            chain=data.get("chain", "base"),
            description=data.get("description", ""),
            facilitator_url=data.get("facilitatorUrl", ""),
        )


class X402Client:
    """
    HTTP client that auto-handles x402 payments.

    Usage:
        wallet = Wallet(private_key="...", address="0x...")
        client = X402Client(wallet=wallet)
        resp = await client.post("https://api.example.com/inference", json={...})
        # If server returns 402, payment is signed and retried automatically.
        # resp.x402_cost contains the USDC amount paid (or None if no payment).
    """

    def __init__(
        self,
        wallet: Wallet,
        facilitator_url: str = "",
        max_auto_pay: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self._wallet = wallet
        self._facilitator_url = facilitator_url
        self._max_auto_pay = max_auto_pay
        self._http = httpx.AsyncClient(timeout=timeout)
        self._total_spent: float = 0.0

    @property
    def total_spent(self) -> float:
        return self._total_spent

    @property
    def wallet_address(self) -> str:
        return self._wallet.address

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: Any = None,
        content: bytes | None = None,
    ) -> "X402Response":
        """Make an HTTP request, auto-paying via x402 if a 402 is returned."""
        headers = dict(headers or {})

        resp = await self._http.request(
            method, url, headers=headers, json=json, content=content
        )

        if resp.status_code != 402:
            return X402Response(response=resp, payment_cost=None)

        # parse payment terms from the 402 response
        payment_header = resp.headers.get("payment-required", "")
        if not payment_header:
            raise X402PaymentError("Server returned 402 but no PAYMENT-REQUIRED header")

        terms = PaymentTerms.from_header(payment_header)

        # safety check: don't auto-pay more than max
        try:
            amount_float = float(terms.amount)
        except ValueError:
            amount_float = 0.0

        if amount_float > self._max_auto_pay:
            raise X402PaymentError(
                f"Payment of {terms.amount} {terms.token} exceeds max_auto_pay "
                f"({self._max_auto_pay})",
                cost=terms.amount,
            )

        # sign and retry
        signed = self._wallet.sign_payment(
            recipient=terms.recipient,
            amount=terms.amount,
            token=terms.token,
        )

        logger.info(
            "x402: paying %s %s to %s for %s",
            terms.amount,
            terms.token,
            terms.recipient[:16] + "...",
            url[:60],
        )

        headers["X-PAYMENT"] = signed.to_header()
        retry_resp = await self._http.request(
            method, url, headers=headers, json=json, content=content
        )

        self._total_spent += amount_float

        if retry_resp.status_code == 402:
            raise X402PaymentError(
                "Payment rejected by server (still 402 after X-PAYMENT)",
                cost=terms.amount,
            )

        return X402Response(response=retry_resp, payment_cost=terms.amount)

    async def get(self, url: str, **kwargs: Any) -> "X402Response":
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> "X402Response":
        return await self.request("POST", url, **kwargs)

    async def close(self) -> None:
        await self._http.aclose()


class X402Response:
    """Wraps an httpx.Response with optional x402 payment metadata."""

    def __init__(
        self, response: httpx.Response, payment_cost: str | None
    ) -> None:
        self._response = response
        self.payment_cost = payment_cost

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def is_success(self) -> bool:
        return self._response.is_success

    @property
    def text(self) -> str:
        return self._response.text

    @property
    def headers(self) -> httpx.Headers:
        return self._response.headers

    def json(self) -> Any:
        return self._response.json()

    def raise_for_status(self) -> None:
        self._response.raise_for_status()
