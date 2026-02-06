from typing import Any

from clickhouse_connect import get_async_client  # type: ignore
from clickhouse_connect.driver.asyncclient import AsyncClient  # type: ignore


class Clickhouse:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database

        self._client: AsyncClient | None

    async def initialize(self):
        self._client = await get_async_client(
            host=self._host,
            port=self._port,
            username=self._user,
            password=self._password,
            database=self._database,
        )

    async def get_schema(self):
        schema: list[dict[str, str]] = []

        resp = await self._client.query(  # type: ignore
            "DESCRIBE TABLE default.osprey_execution_results"
        )  # type: ignore

        for row in resp.result_rows:  # type: ignore
            schema.append(
                {
                    "name": row[0],
                    "type": row[1],
                }
            )
        return schema

    async def query(self, sql: str) -> dict[str, Any]:
        return await self._client.query(sql)  # type: ignore
