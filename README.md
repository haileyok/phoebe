# Phoebe

Phoebe is an AI-powered trust and safety agent for the [AT Protocol](https://atproto.com/) network. It automates safety operations by analyzing network threats and creating rules to detect and resolve emerging issues. Phoebe uses three different services to achieve this:

- **[Osprey](https://github.com/roostorg/osprey)** - Real-time rules engine for threat detection
- **[Ozone](https://github.com/bluesky-social/ozone)** - Moderation service for labeling and takedowns
- **[ClickHouse](https://clickhouse.com/)** - Event analytics database for pattern discovery, which is populated by Osprey

This allows it to:

- **Rule Management** - Write, validate, and deploy rules for Osprey
- **Data Analysis** - Query ClickHouse to analyze what is happening on your network
- **Investigation** - Look up domains, IPs, URLs, and WHOIS records to investigate threats
- **Content Detection** - Find similar posts to detect coordinated spam and templated abuse
- **Moderation** - Apply labels and take moderation actions via Ozone (not actually implemented yet...)

## How It Works

Phoebe uses a model API as its reasoning backer. The agent writes and executes Typescript code in a sandboxed Deno runtime to interact with its tools — querying event data, creating safety rules, and managing moderation actions.

```
┌─────────────────────────────────────────────────────────┐
│                       Model API                         │
├─────────────────────────────────────────────────────────┤
│              Tool Execution (Deno Sandbox)              │
├──────────┬───────────┬──────────────┬───────────────────┤
│  Osprey  │ ClickHouse│    Ozone     │  Investigation    │
│  (Rules) │ (Queries) │ (Moderation) │ (Domain/IP/WHOIS) │
└──────────┴───────────┴──────────────┴───────────────────┘
```

#### Why not traditional tool calling?

See [Cloudflare's blog post](https://blog.cloudflare.com/code-mode/) on this topic.

One of the largest benefits of letting the agent write and execute its own code is that it allows for tool chaining and grouping. Traditionally, each subsequent tool call results in a round trip for _each_ tool call. When the agent can write its own code, it can instead
chain these calls together. For example, if the agent knows it wants to grab the results of _three separate_ SQL queries, it can group all three of those calls in a single `execute_code` block and receive the context.

When executing code inside of Deno, Deno is ran with the bare minimum of permissions. For example, it cannot access the file system, the network (local or remote), or use NPM packages. Both execution time and memory limits are applied. All network requests are done in Python,
in code that _you_ write, not the agent.

| Limit | Value |
|-------|-------|
| Max code size | 50,000 characters |
| Max tool calls per execution | 25 |
| Max output size | 1 MB |
| Execution timeout | 60 seconds |
| V8 heap memory | 256 MB |

## Tools

Phoebe has access to the following tools, organized by namespace:

| Namespace | Tool | Description |
|-----------|------|-------------|
| `clickhouse` | `query(sql)` | Execute SQL queries against Clickhouse |
| `clickhouse` | `getSchema()` | Get the table schema and column info |
| `osprey` | `getConfig()` | Get available features, labels, rules, and actions |
| `osprey` | `getUdfs()` | Get available UDFs for rule writing |
| `osprey` | `listRuleFiles(directory?)` | List existing `.sml` rule files |
| `osprey` | `readRuleFile(file_path)` | Read an existing rule file |
| `osprey` | `saveRule(file_path, content)` | Save or create a rule file |
| `osprey` | `validateRules()` | Validate the ruleset |
| `content` | `similarity(text, threshold?, limit?)` | Find similar posts using n-gram distance |
| `domain` | `checkDomain(domain)` | DNS lookups and HTTP status checks |
| `ip` | `lookup(ip)` | GeoIP and ASN lookups |
| `url` | `expand(url)` | Follow redirect chains and detect shorteners |
| `whois` | `lookup(domain)` | Domain registration and WHOIS info |
| `ozone` | `applyLabel(subject, label)` | Apply a moderation label (not yet implemented) |
| `ozone` | `removeLabel(subject, label)` | Remove a moderation label (not yet implemented) |

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

# Optional - Model API backend (default: anthropic)
# MODEL_API="anthropic"  # or "openai", "openapi"
# MODEL_ENDPOINT=""       # required for openapi, ie https://api.moonshot.ai/v1/completions

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
