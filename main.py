import logging
from collections.abc import Callable
from typing import Literal

import click
import httpx
from aiokafka.client import asyncio

from src.agent.agent import Agent
from src.clickhouse.clickhouse import Clickhouse
from src.config import CONFIG
from src.osprey.osprey import Osprey
from src.ozone.ozone import Ozone
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOL_REGISTRY, ToolContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


SHARED_OPTIONS: list[Callable[..., Callable[..., object]]] = [
    click.option("--clickhouse-host"),
    click.option("--clickhouse-port"),
    click.option("--clickhouse-user"),
    click.option("--clickhouse-password"),
    click.option("--clickhouse-database"),
    click.option("--osprey-base-url"),
    click.option("--model-api"),
    click.option("--model-name"),
    click.option("--model-api-key"),
]


def shared_options[F: Callable[..., object]](func: F) -> F:
    for option in reversed(SHARED_OPTIONS):
        func = option(func)  # type: ignore[assignment]
    return func


def build_services(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
    osprey_base_url: str | None,
    model_api: Literal["anthropic", "openai", "openapi"] | None,
    model_name: str | None,
    model_api_key: str | None,
) -> tuple[Clickhouse, ToolExecutor, Agent]:
    http_client = httpx.AsyncClient()

    clickhouse = Clickhouse(
        host=clickhouse_host or CONFIG.clickhouse_host,
        port=clickhouse_port or CONFIG.clickhouse_port,
        user=clickhouse_user or CONFIG.clickhouse_user,
        password=clickhouse_password or CONFIG.clickhouse_password,
        database=clickhouse_database or CONFIG.clickhouse_database,
    )

    osprey = Osprey(
        http_client=http_client,
        base_url=osprey_base_url or CONFIG.osprey_base_url,
    )

    ozone = Ozone()

    tool_context = ToolContext(
        clickhouse=clickhouse,
        osprey=osprey,
        ozone=ozone,
    )

    executor = ToolExecutor(
        registry=TOOL_REGISTRY,
        ctx=tool_context,
    )

    agent = Agent(
        model_api=model_api or CONFIG.model_api,
        model_name=model_name or CONFIG.model_name,
        model_api_key=model_api_key or CONFIG.model_api_key,
        tool_executor=executor,
    )

    return clickhouse, executor, agent


@click.group()
def cli():
    pass


@cli.command()
@shared_options
def main(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
    osprey_base_url: str | None,
    model_api: Literal["anthropic", "openai", "openapi"] | None,
    model_name: str | None,
    model_api_key: str | None,
):
    clickhouse, executor, agent = build_services(
        clickhouse_host=clickhouse_host,
        clickhouse_port=clickhouse_port,
        clickhouse_user=clickhouse_user,
        clickhouse_password=clickhouse_password,
        clickhouse_database=clickhouse_database,
        osprey_base_url=osprey_base_url,
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
    )

    async def run():
        await clickhouse.initialize()
        await executor.initialize()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(agent.run())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("received keyboard interrupt")


@cli.command(name="chat")
@shared_options
def chat(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
    osprey_base_url: str | None,
    model_api: Literal["anthropic", "openai", "openapi"] | None,
    model_name: str | None,
    model_api_key: str | None,
):
    clickhouse, executor, agent = build_services(
        clickhouse_host=clickhouse_host,
        clickhouse_port=clickhouse_port,
        clickhouse_user=clickhouse_user,
        clickhouse_password=clickhouse_password,
        clickhouse_database=clickhouse_database,
        osprey_base_url=osprey_base_url,
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
    )

    async def run():
        await clickhouse.initialize()
        await executor.initialize()
        logger.info("Services initialized. Starting interactive chat.")
        print("\nAgent ready. Type your message (Ctrl+C to exit).\n")

        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break

            if not user_input.strip():
                continue

            logger.info("User: %s", user_input)
            response = await agent.chat(user_input)
            print(f"\nAgent: {response}\n")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    cli()
