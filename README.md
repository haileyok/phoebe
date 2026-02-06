# Phoebe - AI Trust & Safety Agent for ATProto

Phoebe is an AI-powered trust and safety agent for the [AT Protocol](https://atproto.com/) network. It automates safety operations by analyzing network threats and creating rules to detect and resolve emerging issues. Phoebe uses three different services to achieve this:

- **[Osprey](https://github.com/roostorg/osprey)** - Real-time rules engine for threat detection
- **[Ozone](https://github.com/bluesky-social/ozone)** - Moderation service for labeling and takedowns
- **[ClickHouse](https://clickhouse.com/)** - Event analytics database for pattern discovery, which is populated by Osprey

This allows it to:

- **Rule Management** - Writes, validate, and deploys rules for Osprey
- **Data Analysis** - Queries via Clickhouse to analyze what is happening on your network
- **Moderation** - Apply labels and take moderation actions via Ozone (not actually implemented yet...)

## How It Works

Phoebe uses a model API as its reasoning backer. The agent writes and executes Typescript code in a sandboxed Deno runtime to interact with its tools — querying event data, creating safety rules, and managing moderation actions.

```
┌──────────────────────────────────────┐
│              Model API               │
├──────────────────────────────────────┤
│      Tool Execution (Deno Sandbox)   │
├──────────┬───────────┬───────────────┤
│  Osprey  │ ClickHouse│    Ozone      │
│  (Rules) │ (Queries) │ (Moderation)  │
└──────────┴───────────┴───────────────┘
```

#### Why not traditional tool calling?

See [Cloudflare's blog post](https://blog.cloudflare.com/code-mode/) on this topic.

One of the largest benefits of letting the agent write and execute its own code is that it allows for tool chaining and grouping. Traditionally, each subsequent tool call results in a round trip for _each_ tool call. When the agent can write its own code, it can instead
chain these calls together. For example, if the agent knows it wants to grab the results of _three separate_ SQL queries, it can group all three of those calls in a single `execute_code` block and receive the context.

## Prerequisites

- [Deno](https://deno.com/) runtime
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/haileyok/osprey-agent.git
cd osprey-agent
uv sync --frozen
```

## Configuration

Create a `.env` file in the project root:

```env
# Required
MODEL_API_KEY="sk-ant-api03-..."
MODEL_NAME="claude-sonnet-4-5-20250929"

# Osprey
OSPREY_BASE_URL="http://localhost:5004"
OSPREY_REPO_URL="https://github.com/roostorg/osprey"
OSPREY_RULESET_URL="https://github.com/your-org/your-ruleset"

# ClickHouse
CLICKHOUSE_HOST="localhost"
CLICKHOUSE_PORT=8123
CLICKHOUSE_DATABASE="default"
CLICKHOUSE_USER="default"
CLICKHOUSE_PASSWORD="clickhouse"
```

All settings can also be passed as CLI flags (see `--help`).

## Usage

### Interactive Chat

Start a conversation with Phoebe to investigate threats and create rules:

```bash
uv run main.py chat
```

### CLI Options

Both commands accept overrides for any config value:

```bash
uv run main.py chat \
  --clickhouse-host localhost \
  --clickhouse-port 8123 \
  --osprey-base-url http://localhost:5004 \
  --model-api-key $ANTHROPIC_API_KEY
```
