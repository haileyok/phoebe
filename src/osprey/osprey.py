import asyncio
import logging
from pathlib import Path
from typing import Any
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
        success, result = await self.validate_rules()
        if not success:
            raise RuntimeError(f"Rule validation failed!\n{result}")

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

    def list_rule_files(self, directory: str | None = None) -> list[str]:
        """list .sml files under the given directory (or all of rules/) relative to ruleset root"""
        if directory:
            search_dir = OSPREY_RULESET_PATH / directory
        else:
            search_dir = OSPREY_RULESET_PATH / "rules"

        resolved = search_dir.resolve()
        if not resolved.is_relative_to(OSPREY_RULESET_PATH.resolve()):
            raise ValueError("Directory path must be within the ruleset")

        if not search_dir.exists():
            return []

        return sorted(
            str(p.relative_to(OSPREY_RULESET_PATH)) for p in search_dir.rglob("*.sml")
        )

    def read_rule_file(self, file_path: str) -> str:
        """Read and return the content of an .sml file within the ruleset."""
        target = (OSPREY_RULESET_PATH / file_path).resolve()
        if not target.is_relative_to(OSPREY_RULESET_PATH.resolve()):
            raise ValueError("File path must be within the ruleset")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return target.read_text()

    def save_rule(
        self, file_path: str, content: str, require_if: str | None = None
    ) -> dict[str, Any]:
        """Write an .sml file to the ruleset. Auto-registers new files in parent index.sml."""
        if not file_path.endswith(".sml"):
            raise ValueError("File path must have .sml extension")

        target = (OSPREY_RULESET_PATH / file_path).resolve()
        if not target.is_relative_to(OSPREY_RULESET_PATH.resolve()):
            raise ValueError("File path must be within the ruleset")

        is_update = target.exists()

        # ensure content ends with newline
        if not content.endswith("\n"):
            content += "\n"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

        registered_in_index = False

        # auto-register new non-index files in parent index.sml, auto-register to prevent additioanl context bloat
        # HACK: maybe we want to remove this later if it isn't very good
        if not is_update and target.name != "index.sml":
            index_path = target.parent / "index.sml"
            if index_path.exists():
                rule_path = str(target.relative_to(OSPREY_RULESET_PATH.resolve()))
                if require_if:
                    require_line = f"Require(rule='{rule_path}', require_if={require_if})\n"
                else:
                    require_line = f"Require(rule='{rule_path}')\n"

                with open(index_path, "a") as f:
                    f.write(require_line)
                registered_in_index = True

        return {
            "action": "updated" if is_update else "created",
            "file_path": file_path,
            "registered_in_index": registered_in_index,
        }

    async def validate_rules(self) -> tuple[bool, str]:
        """validates the rules in the ruleset directory. returns (success, output_string). doesn't return stderr since its noisy"""
        # uv run osprey-cli push-rules ../atproto-ruleset --dry-run
        process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "osprey-cli",
            "push-rules",
            "../ruleset",
            "--dry-run",  # doesn't actually push rules, only validates
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=OSPREY_REPO_PATH,
        )

        stdout, stderr = await process.communicate()
        stdout_str = stdout.decode().strip()

        if process.returncode != 0:
            # stdout has the formatted error and stderr has noisy warnings
            error_output = stdout_str if stdout_str else stderr.decode().strip()
            return (False, error_output)

        return (True, stdout_str if stdout_str else "Validation successful")
