"""
Production x402 HTTP client with facilitator-based payment settlement.

Handles the full x402 flow:
  1. Client makes a normal request.
  2. Server returns 402 + PAYMENT-REQUIRED header.
  3. Client signs a payment via its Wallet (real EVM signature).
  4. Client sends the signed payment to the facilitator for settlement.
  5. Facilitator settles on-chain, returns a settlement receipt.
  6. Client resends the original request with X-PAYMENT header containing
     the settlement receipt.
  7. Server verifies receipt and returns 200.

If no facilitator URL is configured, falls back to direct mode (sends
the signed payment directly to the server, letting the server settle).
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.x402.wallet import DevWallet, SignedPayment, Wallet

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
    resource_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_header(cls, header_value: str) -> "PaymentTerms":
        try:
            decoded = base64.b64decode(header_value)
            data = json.loads(decoded)
        except Exception as e:
            raise X402PaymentError(f"Invalid PAYMENT-REQUIRED header: {e}")

        return cls(
            recipient=data.get("recipient", ""),
            amount=str(data.get("amount", "0")),
            token=data.get("token", "USDC"),
            chain=data.get("chain", "base"),
            description=data.get("description", ""),
            facilitator_url=data.get("facilitatorUrl", ""),
            resource_url=data.get("resourceUrl", ""),
            extra={k: v for k, v in data.items() if k not in {
                "recipient", "amount", "token", "chain",
                "description", "facilitatorUrl", "resourceUrl",
            }},
        )


@dataclass
class PaymentRecord:
    """Record of a settled payment for auditing."""

    timestamp: float
    url: str
    amount: float
    token: str
    recipient: str
    tx_hash: str
    facilitator_url: str


class X402Client:
    """
    Production HTTP client with x402 payment handling.

    Supports two settlement modes:
    1. Facilitator mode (default): Sends signed payment to a facilitator
       service that settles on-chain and returns a receipt.
    2. Direct mode: Sends signed payment directly to the resource server
       (server is responsible for settlement).
    """

    def __init__(
        self,
        wallet: Wallet | DevWallet,
        facilitator_url: str = "",
        max_auto_pay: float = 1.0,
        timeout: float = 120.0,
        spending_limit: float = 100.0,
    ) -> None:
        self._wallet = wallet
        self._facilitator_url = facilitator_url
        self._max_auto_pay = max_auto_pay
        self._spending_limit = spending_limit
        self._http = httpx.AsyncClient(timeout=timeout)
        self._total_spent: float = 0.0
        self._payment_log: list[PaymentRecord] = []

    @property
    def total_spent(self) -> float:
        return self._total_spent

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self._spending_limit - self._total_spent)

    @property
    def wallet_address(self) -> str:
        return self._wallet.address

    @property
    def payment_log(self) -> list[PaymentRecord]:
        return list(self._payment_log)

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
            return X402Response(response=resp, payment_cost=None, tx_hash=None)

        # --- x402 flow ---
        payment_header = resp.headers.get("payment-required", "")
        if not payment_header:
            raise X402PaymentError("Server returned 402 but no PAYMENT-REQUIRED header")

        terms = PaymentTerms.from_header(payment_header)

        # safety: check single-payment limit
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

        # safety: check cumulative spending limit
        if self._total_spent + amount_float > self._spending_limit:
            raise X402PaymentError(
                f"Payment of {terms.amount} {terms.token} would exceed spending "
                f"limit ({self._spending_limit}). Spent so far: {self._total_spent}",
                cost=terms.amount,
            )

        # sign the payment
        signed = self._wallet.sign_payment(
            recipient=terms.recipient,
            amount=terms.amount,
            token=terms.token,
        )

        # settle via facilitator or direct
        facilitator = terms.facilitator_url or self._facilitator_url
        tx_hash = ""

        if facilitator:
            tx_hash = await self._settle_via_facilitator(facilitator, signed, terms)
        else:
            logger.debug("No facilitator configured; using direct payment mode")

        logger.info(
            "x402: paying %s %s to %s for %s (tx: %s)",
            terms.amount, terms.token,
            terms.recipient[:16] + "...",
            url[:60],
            tx_hash[:16] + "..." if tx_hash else "direct",
        )

        # resend with payment proof
        headers["X-PAYMENT"] = signed.to_header()
        if tx_hash:
            headers["X-PAYMENT-TX"] = tx_hash

        retry_resp = await self._http.request(
            method, url, headers=headers, json=json, content=content
        )

        if retry_resp.status_code == 402:
            raise X402PaymentError(
                "Payment rejected by server (still 402 after settlement)",
                cost=terms.amount,
            )

        # record successful payment
        self._total_spent += amount_float
        self._payment_log.append(PaymentRecord(
            timestamp=time.time(),
            url=url,
            amount=amount_float,
            token=terms.token,
            recipient=terms.recipient,
            tx_hash=tx_hash,
            facilitator_url=facilitator,
        ))

        return X402Response(
            response=retry_resp, payment_cost=terms.amount, tx_hash=tx_hash,
        )

    async def _settle_via_facilitator(
        self,
        facilitator_url: str,
        signed: SignedPayment,
        terms: PaymentTerms,
    ) -> str:
        """
        Send the signed payment to the x402 facilitator for on-chain settlement.

        The facilitator:
        1. Verifies the EIP-191 signature
        2. Submits the USDC transfer on-chain
        3. Waits for confirmation
        4. Returns the transaction hash

        Returns the tx hash on success.
        """
        settle_url = f"{facilitator_url.rstrip('/')}/settle"

        payload = {
            "payment": {
                "chain": signed.chain,
                "token": signed.token,
                "amount": signed.amount,
                "recipient": signed.recipient,
                "sender": signed.sender,
                "signature": signed.signature,
                "timestamp": signed.timestamp,
                "nonce": signed.nonce,
            },
            "resource": {
                "url": terms.resource_url,
                "description": terms.description,
            },
        }

        try:
            resp = await self._http.post(
                settle_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if not resp.is_success:
                error_text = resp.text[:500]
                raise X402PaymentError(
                    f"Facilitator settlement failed (HTTP {resp.status_code}): {error_text}",
                    cost=terms.amount,
                )

            data = resp.json()
            tx_hash = data.get("txHash", data.get("transactionHash", ""))

            if not tx_hash:
                logger.warning("Facilitator returned success but no tx hash: %s", data)

            return tx_hash

        except httpx.HTTPError as e:
            raise X402PaymentError(
                f"Facilitator HTTP error: {e}", cost=terms.amount,
            )

    async def get(self, url: str, **kwargs: Any) -> "X402Response":
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> "X402Response":
        return await self.request("POST", url, **kwargs)

    async def close(self) -> None:
        await self._http.aclose()


class X402Response:
    """Wraps an httpx.Response with x402 payment metadata."""

    def __init__(
        self,
        response: httpx.Response,
        payment_cost: str | None,
        tx_hash: str | None = None,
    ) -> None:
        self._response = response
        self.payment_cost = payment_cost
        self.tx_hash = tx_hash

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
