from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # clickhouse config
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

    # kafka config (currently unused but maybe later...)
    bootstrap_server: str = "localhost:9092"
    """bootstrap server for atkafka events"""
    input_topic: str = "atproto-events"
    """input topic for atkafka events"""
    group_id: str = "osprey-agent"
    """group id for atkafka events"""

    # model config. currently only supporting anthropic, but we can add the other models later.
    # really want to see performance on kimi2.5...
    model_api: Literal["anthropic", "openai", "openapi"] = "anthropic"
    """the model api to use. must be one of `anthropic`, `openai`, or `openapi`"""
    model_name: str = "claude-sonnet-4-5-20250929"
    """the model to use with the given api"""
    model_api_key: str = ""
    """the model api key"""
    model_endpoint: str = ""
    """for openapi model apis, the endpoint to use"""

    # ozone config
    ozone_moderator_pds_host: str = ""
    """the PDS host for the moderator account that has at least moderator-level permissions in Ozone"""
    ozone_moderator_identifier: str = ""
    """the moderator account's identifier (handle)"""
    ozone_moderator_password: str = ""
    """the moderator account's password"""
    ozone_labeler_account_did: str = ""
    """the DID of the labeler account. this variable is not the same as the moderator account, though for purely-agentified ozone instances, they may be the same. not recommended, since that means you're giving the agent _admin_ permissions..."""
    ozone_allowed_labels: str = ""
    """comma separated list of labels that Phoebe is allowed to apply. both specified to the agent via prompting and validated before applying labels directly"""

    # osprey config
    osprey_base_url: str = ""
    """the base url for your osprey instance"""
    osprey_repo_url: str = "https://github.com/roostorg/osprey"
    """the url to fetch the osprey codebase from. used for letting the agent validate written rules directly"""
    osprey_ruleset_url: str = "https://github.com/haileyok/atproto-ruleset"
    """the url to fetch the osprey ruleset you are running. used when validating written rules (i.e. for having the needed features available for validation)"""

    model_config = SettingsConfigDict(env_file=".env")


CONFIG = Config()
