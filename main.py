import logging
from collections.abc import Callable
from typing import Literal

import click
from aiokafka.client import asyncio

from src.agent.agent import Agent
from src.arena.context import ArenaContext
from src.arena.scorer import Scorer, ScoringConfig
from src.arena.server import ArenaServer
from src.arena.store import ArenaStore
from src.clickhouse.clickhouse import Clickhouse
from src.config import CONFIG
from src.safety.classifier import SafetyClassifier
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOL_REGISTRY, ToolContext
from src.x402.client import X402Client
from src.x402.wallet import DevWallet, Wallet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# disable httpx verbose logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI option groups
# ---------------------------------------------------------------------------

SHARED_OPTIONS: list[Callable[..., Callable[..., object]]] = [
    click.option("--clickhouse-host"),
    click.option("--clickhouse-port"),
    click.option("--clickhouse-user"),
    click.option("--clickhouse-password"),
    click.option("--clickhouse-database"),
    click.option("--model-api"),
    click.option("--model-name"),
    click.option("--model-api-key"),
    click.option("--model-endpoint"),
]

ARENA_OPTIONS: list[Callable[..., Callable[..., object]]] = [
    click.option("--arena-host"),
    click.option("--arena-port", type=int),
    click.option("--x402-wallet-key"),
    click.option("--x402-wallet-address"),
    click.option("--dev-mode/--no-dev-mode", default=None, help="Run in dev mode (HMAC wallets, skip EVM verification)"),
]


def shared_options[F: Callable[..., object]](func: F) -> F:
    for option in reversed(SHARED_OPTIONS):
        func = option(func)  # type: ignore[assignment]
    return func


def arena_options[F: Callable[..., object]](func: F) -> F:
    for option in reversed(ARENA_OPTIONS):
        func = option(func)  # type: ignore[assignment]
    return func


# ---------------------------------------------------------------------------
# Service builders
# ---------------------------------------------------------------------------


def build_clickhouse(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
) -> Clickhouse:
    return Clickhouse(
        host=clickhouse_host or CONFIG.clickhouse_host,
        port=clickhouse_port or CONFIG.clickhouse_port,
        user=clickhouse_user or CONFIG.clickhouse_user,
        password=clickhouse_password or CONFIG.clickhouse_password,
        database=clickhouse_database or CONFIG.clickhouse_database,
    )


def build_x402(
    wallet_key: str | None = None,
    wallet_address: str | None = None,
    dev_mode: bool | None = None,
) -> X402Client:
    is_dev = dev_mode if dev_mode is not None else CONFIG.arena_dev_mode

    if is_dev:
        wallet = DevWallet(
            address=wallet_address or CONFIG.x402_wallet_address or "0xdev",
            chain=CONFIG.x402_chain,
        )
    else:
        wallet = Wallet(
            private_key=wallet_key or CONFIG.x402_wallet_private_key,
            chain=CONFIG.x402_chain,
        )

    return X402Client(
        wallet=wallet,
        facilitator_url=CONFIG.x402_facilitator_url,
        max_auto_pay=CONFIG.x402_max_auto_pay,
        spending_limit=CONFIG.x402_spending_limit,
    )


def build_safety_classifier() -> SafetyClassifier:
    return SafetyClassifier(
        api_key=CONFIG.model_api_key,
        model_name=CONFIG.safety_classifier_model,
        endpoint=CONFIG.safety_classifier_endpoint,
    )


def build_arena_services(
    clickhouse: Clickhouse,
    x402_client: X402Client,
    safety_classifier: SafetyClassifier,
    store: ArenaStore,
    model_api: str | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
) -> tuple[ArenaContext, ToolExecutor, Agent, Scorer]:
    arena_ctx = ArenaContext(store=store)

    tool_context = ToolContext(
        clickhouse=clickhouse,
        x402_client=x402_client,
        safety_classifier=safety_classifier,
        arena=arena_ctx,
    )

    executor = ToolExecutor(
        registry=TOOL_REGISTRY,
        ctx=tool_context,
    )

    agent = Agent(
        model_api=model_api or CONFIG.model_api,
        model_name=model_name or CONFIG.model_name,
        model_api_key=model_api_key or CONFIG.model_api_key,
        model_endpoint=model_endpoint or CONFIG.model_endpoint or None,
        tool_executor=executor,
    )

    scoring_config = ScoringConfig(
        alpha=CONFIG.arena_scoring_alpha,
        beta=CONFIG.arena_scoring_beta,
        gamma=CONFIG.arena_scoring_gamma,
        delta=CONFIG.arena_scoring_delta,
        payout_rate=CONFIG.arena_payout_rate,
    )

    scorer = Scorer(
        x402_client=x402_client,
        safety_classifier=safety_classifier,
        store=store,
        config=scoring_config,
    )

    return arena_ctx, executor, agent, scorer


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command(name="arena")
@shared_options
@arena_options
def arena_cmd(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
    model_api: Literal["anthropic", "openai", "openapi"] | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
    arena_host: str | None,
    arena_port: int | None,
    x402_wallet_key: str | None,
    x402_wallet_address: str | None,
    dev_mode: bool | None,
):
    """Start the Sandbox Arena HTTP server (red teaming marketplace)."""
    is_dev = dev_mode if dev_mode is not None else CONFIG.arena_dev_mode

    clickhouse = build_clickhouse(
        clickhouse_host, clickhouse_port, clickhouse_user,
        clickhouse_password, clickhouse_database,
    )
    x402_client = build_x402(x402_wallet_key, x402_wallet_address, dev_mode)
    classifier = build_safety_classifier()

    store = ArenaStore(clickhouse)

    arena_ctx, executor, agent, scorer = build_arena_services(
        clickhouse=clickhouse,
        x402_client=x402_client,
        safety_classifier=classifier,
        store=store,
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
        model_endpoint=model_endpoint,
    )

    server = ArenaServer(
        scorer=scorer,
        store=store,
        submission_fee_usdc=CONFIG.arena_submission_fee,
        arena_wallet=CONFIG.arena_wallet,
        facilitator_url=CONFIG.x402_facilitator_url,
        dev_mode=is_dev,
    )

    host = arena_host or CONFIG.arena_host
    port = arena_port or CONFIG.arena_port

    async def run():
        await clickhouse.initialize()
        await store.initialize()
        await executor.initialize()

        import uvicorn

        app = server.build_app()
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        srv = uvicorn.Server(config)

        logger.info("Sandbox Arena starting on %s:%d", host, port)
        logger.info("x402 wallet: %s (chain: %s)", x402_client.wallet_address, CONFIG.x402_chain)
        logger.info("Dev mode: %s", is_dev)
        await srv.serve()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Arena shutting down.")


@cli.command(name="chat")
@shared_options
@arena_options
def chat(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
    model_api: Literal["anthropic", "openai", "openapi"] | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
    arena_host: str | None,
    arena_port: int | None,
    x402_wallet_key: str | None,
    x402_wallet_address: str | None,
    dev_mode: bool | None,
):
    """Interactive chat mode with Phoebe (arena judge)."""
    clickhouse = build_clickhouse(
        clickhouse_host, clickhouse_port, clickhouse_user,
        clickhouse_password, clickhouse_database,
    )
    x402_client = build_x402(x402_wallet_key, x402_wallet_address, dev_mode)
    classifier = build_safety_classifier()

    store = ArenaStore(clickhouse)

    arena_ctx, executor, agent, scorer = build_arena_services(
        clickhouse=clickhouse,
        x402_client=x402_client,
        safety_classifier=classifier,
        store=store,
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
        model_endpoint=model_endpoint,
    )

    async def run():
        await clickhouse.initialize()
        await store.initialize()
        await executor.initialize()
        logger.info("Services initialized. Starting interactive chat.")
        print("\nPhoebe (Arena Judge) ready. Type your message (Ctrl+C to exit).\n")

        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break

            if not user_input.strip():
                continue

            logger.info("User: %s", user_input)
            response = await agent.chat(user_input)
            print(f"\nPhoebe: {response}\n")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    cli()
