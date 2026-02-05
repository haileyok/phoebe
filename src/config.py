from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    clickhouse_host: str = "localhost"
    """host for the clickhouse server"""
    clickhouse_port: int = 8123
    """port for the clickhouse server"""
    clickhouse_user: str = "default"
    """username for the clickhouse server"""
    clickhouse_password: str = "clickhouse"
    """password for the clickhouse server"""
    clickhouse_database: str = "default"
    """default database for the clickhouse server"""

    bootstrap_server: str = "localhost:9092"
    """bootstrap server for atkafka events"""
    input_topic: str = "atproto-events"
    """input topic for atkafka events"""
    group_id: str = "osprey-agent"
    """group id for atkafka events"""

    model_api: Literal["anthropic", "openai", "openapi"] = "anthropic"
    """the model api to use. must be one of `anthropic`, `openai`, or `openapi`"""
    model_name: str = "claude-sonnet-4-5-20250929"
    """the model to use with the given api"""
    model_api_key: str = ""
    """the model api key"""
    model_endpoint: str = ""
    """for openapi model apis, the endpoint to use"""

    allowed_labels: str = ""
    """comma separated list of labels that Phoebe is allowed to apply"""

    osprey_base_url: str = ""
    """the base url for your osprey instance"""

    model_config = SettingsConfigDict(env_file=".env")


CONFIG = Config()
