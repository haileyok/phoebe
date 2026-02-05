import logging
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
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--clickhouse-host")
@click.option("--clickhouse-port")
@click.option("--clickhouse-user")
@click.option("--clickhouse-password")
@click.option("--clickhouse-database")
@click.option("--bootstrap-server")
@click.option("--input-topic")
@click.option("--group-id")
@click.option("--osprey-base-url")
@click.option("--model-api")
@click.option("--model-name")
@click.option("--model-api-key")
def main(
    clickhouse_host: str | None,
    clickhouse_port: int | None,
    clickhouse_user: str | None,
    clickhouse_password: str | None,
    clickhouse_database: str | None,
    bootstrap_server: str | None,
    input_topic: str | None,
    group_id: str | None,
    osprey_base_url: str | None,
    model_api: Literal["anthropic", "openai", "openapi"],
    model_name: str | None,
    model_api_key: str | None,
):
    http_client = httpx.AsyncClient()

    clickhouse = Clickhouse(
        host=clickhouse_host or CONFIG.clickhouse_host,
        port=clickhouse_port or CONFIG.clickhouse_port,
        user=clickhouse_user or CONFIG.clickhouse_user,
        password=clickhouse_password or CONFIG.clickhouse_password,
        database=clickhouse_database or CONFIG.clickhouse_database,
    )

    # indexer = Indexer(
    #     bootstrap_servers=[bootstrap_server or CONFIG.bootstrap_server],
    #     input_topic=input_topic or CONFIG.input_topic,
    #     group_id=group_id or CONFIG.group_id,
    #     clickhouse=clickhouse,
    # )

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

    async def run():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(agent.run())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("received keyboard interrupt")


if __name__ == "__main__":
    main()
