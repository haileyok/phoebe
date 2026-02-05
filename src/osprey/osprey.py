import httpx

from src.osprey.config import OspreyConfig
from src.osprey.udfs import UdfCatalog


class Osprey:
    def __init__(self, http_client: httpx.AsyncClient, base_url: str) -> None:
        self._http_client = http_client
        self._base_url = base_url

    async def get_udfs(self) -> UdfCatalog:
        """gets the udf documentation from the given osprey instance"""

        url = f"{self._base_url}/docs/udfs"
        resp = await self._http_client.get(url)
        resp.raise_for_status()
        return UdfCatalog.model_validate(resp.json())

    async def get_config(self) -> OspreyConfig:
        """gets the config from the given osprey instance, for label names, features, etc."""

        url = f"{self._base_url}/config"
        resp = await self._http_client.get(url)
        resp.raise_for_status()
        return OspreyConfig.model_validate(resp.json())
