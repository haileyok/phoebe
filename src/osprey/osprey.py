import asyncio
import logging
from pathlib import Path
import httpx

from src.osprey.config import OspreyConfig
from src.osprey.udfs import UdfCatalog

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")

OSPREY_REPO_PATH = Path("./data/osprey")

OSPREY_RULESET_PATH = Path("./data/ruleset")


class Osprey:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        osprey_repo_url: str,
        osprey_ruleset_url: str,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url
        self._osprey_repo_url = osprey_repo_url
        self._osprey_ruleset_url = osprey_ruleset_url

    async def initialize(self):
        DATA_DIR.mkdir(exist_ok=True)

        if not OSPREY_REPO_PATH.exists():
            logging.info(
                f"Fetching Osprey repo from '{self._osprey_repo_url}' and saving to '{OSPREY_REPO_PATH}'"
            )
            await self._fetch_osprey_repo()
        else:
            logging.info("Osprey repo was already available, not fetching...")

        if not OSPREY_RULESET_PATH.exists():
            logging.info(
                f"Fetching Osprey ruleset from '{self._osprey_ruleset_url}' and saving to '{OSPREY_RULESET_PATH}'"
            )
            await self._fetch_osprey_ruleset()
        else:
            logging.info("Osprey ruleset was already available, not fetching...")

        logging.info("syncing python deps for osprey repo...")
        await self._repo_deps()

        logging.info("verifying current ruleset validates properly...")
        await self.validate_rules()

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

    async def _fetch_osprey_repo(self):
        """fetches the osprey repo from the input http git url"""
        process = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            self._osprey_repo_url,
            str(OSPREY_REPO_PATH),
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stderr is not None

        await process.wait()

        if process.returncode != 0:
            stderr_content = await process.stderr.read()
            stderr_str = stderr_content.decode().strip()
            raise RuntimeError(
                f"Failed to fetch Osprey repo from specified url: {stderr_str}"
            )

    async def _fetch_osprey_ruleset(self):
        """Fetches the osprey ruleset from the input http git url"""
        process = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            self._osprey_ruleset_url,
            str(OSPREY_RULESET_PATH),
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stderr is not None

        await process.wait()

        if process.returncode != 0:
            stderr_content = await process.stderr.read()
            stderr_str = stderr_content.decode().strip()
            raise RuntimeError(
                f"Failed to fetch Osprey ruleset from specified url: {stderr_str}"
            )

    async def _repo_deps(self):
        """syncs deps with uv for the osprey repo"""
        process = await asyncio.create_subprocess_exec(
            "uv",
            "sync",
            "--frozen",
            stderr=asyncio.subprocess.PIPE,
            cwd=OSPREY_REPO_PATH,
        )

        assert process.stderr is not None

        await process.wait()

        if process.returncode != 0:
            stderr_content = await process.stderr.read()
            stderr_str = stderr_content.decode().strip()
            raise RuntimeError(
                f"failed to sync python deps in osprey repo: {stderr_str}"
            )

    async def validate_rules(self):
        """validates the rules that are in the specified ruleset directory. returns error if speicifed, otherwise None"""
        # uv run osprey-cli push-rules ../atproto-ruleset --dry-run
        process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "osprey-cli",
            "push-rules",
            "../ruleset",
            "--dry-run",  # doesn't actually push rules, only validates
            stderr=asyncio.subprocess.PIPE,
            cwd=OSPREY_REPO_PATH,
        )

        assert process.stderr is not None

        await process.wait()

        if process.returncode != 0:
            stderr_content = await process.stderr.read()
            stderr_str = stderr_content.decode().strip()
            raise RuntimeError(f"WARNING! Rule validation failed! Error: {stderr_str}")
