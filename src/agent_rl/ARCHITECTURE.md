# Sara — Generalized Agent Framework & RL Architecture

> **Version**: 2.0  
> **Status**: Active development  
> **Last updated**: April 2026

---

## Overview

Sara is a provider-agnostic, config-driven, multi-persona agent framework built on top of Anthropic Claude. Originally housing Phoebe (the Sandbox Arena judge/red-teamer), it has been generalized so that new agent personas can be created by adding a config file — no framework code changes required.

The framework has four layers:

1. **Agent Runtime** — provider abstraction, session management, tool execution
2. **OpenClaw Integration** — multi-channel deployment adapter
3. **Reinforcement Learning Pipeline** — learn from experience across all deployments
4. **Fine-Tuning** — weight updates for open-source model backends

---

## Directory Structure

```
sara/
├── src/
│   ├── agent/
│   │   ├── agent.py          ← Generalized Agent runtime (provider-agnostic)
│   │   ├── config.py         ← AgentConfig + ModelConfig dataclasses
│   │   ├── session.py        ← Session/memory layer (in-memory, file, Redis, ClickHouse)
│   │   └── tools.py          ← Pluggable ToolExecutor interface + registry
│   ├── openclaw/
│   │   └── adapter.py        ← OpenClaw integration (channels, MCP, registry)
│   └── learning/
│       ├── layer1_logging.py  ← Interaction capture → ClickHouse
│       ├── layer2_reward.py   ← Reward signal computation
│       ├── layer3_retrieval.py ← In-context learning via experience retrieval
│       └── layer4_finetune.py  ← DPO/OpenAI fine-tuning pipeline
├── agents/
│   ├── phoebe/
│   │   └── config.py         ← Phoebe persona (judge/admin/redteam modes)
│   └── sara_base/
│       └── config.py         ← Minimal template for new agents
└── ARCHITECTURE.md           ← This file
```

---

## Part 1: Generalized Agent Framework

### Design Principle: Separate Persona from Runtime

The original agent had Phoebe's identity baked into the Agent class. The refactored design externalizes all persona-specific content into `AgentConfig` objects:

```
┌─────────────────────────────────┐
│  AgentConfig (persona layer)    │  ← lives in agents/<name>/config.py
│  - name, description            │
│  - modes: {mode → system_prompt}│
│  - default_mode                 │
│  - default_model (ModelConfig)  │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Agent (runtime layer)          │  ← lives in src/agent/agent.py
│  - client (Anthropic or OAI)    │
│  - tool_executor (pluggable)    │
│  - session_store (pluggable)    │
│  - chat(message, session_id)    │
└─────────────────────────────────┘
```

To create a new agent persona, you only need to create `agents/<name>/config.py` with a new `AgentConfig`. The runtime handles everything else.

### Supported Model Providers

All providers share the same `Agent.chat()` interface. Switch by changing `ModelConfig`:

| Provider | Class | Config shortcut |
|---|---|---|
| Anthropic Claude | `AnthropicClient` | `ModelConfig.anthropic("claude-sonnet-4-5")` |
| OpenAI | `OpenAICompatibleClient` | `ModelConfig.openai("gpt-4o")` |
| Kimi (Moonshot) | `OpenAICompatibleClient` | `ModelConfig.kimi("moonshot-v1-8k")` |
| Qwen (DashScope) | `OpenAICompatibleClient` | `ModelConfig.qwen("qwen-plus")` |
| Gemini | `OpenAICompatibleClient` | `ModelConfig.gemini("gemini-2.0-flash")` |
| DeepSeek | `OpenAICompatibleClient` | `ModelConfig.deepseek("deepseek-chat")` |
| Ollama (local) | `OpenAICompatibleClient` | `ModelConfig.ollama("llama3.2")` |
| Any OAI-compat | `OpenAICompatibleClient` | `ModelConfig.custom(model, endpoint, key_env)` |

**Key insight**: The `AnthropicClient` adds prompt caching (cache_control breakpoints on system prompt, tools, and last message). The `OpenAICompatibleClient` handles Anthropic→OAI message format conversion including tool calls, reasoning_content, and edge cases with empty assistant messages.

### Pluggable Tool Executor

```python
class ToolExecutor(ABC):
    def get_tool_definitions(self) -> list[dict]: ...   # Anthropic-format
    async def execute(self, tool_name, input) -> dict: ...
```

Built-in executors:
- `NullToolExecutor` — no tools (text-only agents)
- `CodeToolExecutor` — wraps original execute_code behaviour
- `CompositeToolExecutor` — fan-out across multiple executors (first-match routing)

Custom executors (e.g. `OpenClawMCPExecutor`) just implement the two methods.

### Session/Memory Layer

Four backends, same interface:

```python
class SessionStore(ABC):
    async def get_or_create(session_id, agent_name, mode) -> Session
    async def save(session: Session) -> None
    async def delete(session_id) -> None
    async def list_sessions(agent_name) -> list[str]
```

| Backend | Use case |
|---|---|
| `InMemorySessionStore` | Development, single-user, original behaviour |
| `FileSessionStore` | Single-node persistence without Redis |
| `RedisSessionStore` | Multi-process OpenClaw deployments |
| `ClickHouseSessionStore` | Append-only log; doubles as RL training data |

Session IDs are arbitrary strings. Convention: `{agent_name}:{channel_id}:{user_id}` for per-user-per-channel isolation.

---

## Part 2: OpenClaw Integration

### Architecture

```
OpenClaw (Discord/Telegram/Slack/WhatsApp)
│
├── AgentRegistry
│   ├── "phoebe" → OpenClawAgent(PHOEBE_CONFIG, redis, mcp)
│   └── "sara"   → OpenClawAgent(SARA_BASE_CONFIG, redis, mcp)
│
├── MCP Skill Plugins
│   └── OpenClawMCPExecutor bridges Sara tool calls → MCP dispatcher
│
├── Cron Scheduler
│   ├── run_reward_labelling_job()   (every hour)
│   └── index_new_examples()         (every hour)
│
└── Channel → Mode Mapping
    ├── #bounty-eval  → phoebe/judge
    ├── #red-team     → phoebe/redteam
    └── #admin        → phoebe/admin
```

### Wiring (entrypoint)

```python
from src.openclaw.adapter import register_all

agents = await register_all(
    openclaw_instance=oc,
    redis_url=os.getenv("REDIS_URL"),
    phoebe_channel_map={
        "CHANNEL_JUDGE_ID":   "judge",
        "CHANNEL_REDTEAM_ID": "redteam",
        "CHANNEL_ADMIN_ID":   "admin",
    },
)

@oc.on_message
async def handle(event):
    agent = agents.get("phoebe")
    reply = await agent.on_message(
        user_id=str(event.author.id),
        channel_id=str(event.channel.id),
        text=event.content,
    )
    await event.channel.send(reply)
```

### Tool Call Flow

```
Sara calls tool "target.generate"
        │
        ▼
CompositeToolExecutor
        │
        ├─ CodeToolExecutor  (no match)
        │
        └─ OpenClawMCPExecutor
                │
                ▼
        openclaw.mcp.call("target.generate", input)
                │
                ▼
        MCP Server (Sandbox Arena tools)
                │
                ▼
        result dict → back to Sara
```

---

## Part 3: Reinforcement Learning Pipeline

The RL pipeline has four layers, each building on the previous. Implement them in order.

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: Fine-Tuning (open models, weight updates)          │
│  Prerequisite: 5,000+ labelled interactions per mode         │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: In-Context Learning (experience retrieval)         │
│  Prerequisite: 500+ interactions, rewards computed           │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Reward Computation (automatic + human labels)      │
│  Prerequisite: interactions logged in ClickHouse             │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Interaction Logging (capture everything)           │
│  Prerequisite: none — implement this first                   │
└──────────────────────────────────────────────────────────────┘
```

### Layer 1 — Interaction Logging

**What**: Capture every agent interaction as a structured record in ClickHouse.  
**When**: Implement immediately. No RL is possible without this data.  
**Cost**: Near-zero. One INSERT per agent turn.

Every `Agent.chat()` call produces one `InteractionRecord`:

```python
@dataclass
class InteractionRecord:
    session_id: str
    agent_name: str
    mode: str              # judge | redteam | admin
    model_name: str
    user_message: str
    response_text: str
    tool_calls: list[ToolCallRecord]
    outcome: dict          # safety scores, novelty, coverage, etc.
    reward: float | None   # filled by Layer 2
    human_label: str | None
```

ClickHouse is already in the Sara stack (`clickhouse.query` is a tool). Interactions go into `sara_interactions` — Sara can query her own experience as part of task analysis.

**ClickHouse schema**:
```sql
CREATE TABLE sara_interactions (
    interaction_id   String,
    session_id       String,
    agent_name       String,
    mode             String,
    model_name       String,
    provider         String,
    user_message     String,
    response_text    String,
    tool_calls       String,   -- JSON
    outcome          String,   -- JSON
    reward           Nullable(Float32),
    human_label      Nullable(String),
    created_at       DateTime64(3) DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (agent_name, mode, created_at);
```

---

### Layer 2 — Reward Signal Computation

**What**: Assign a scalar [0, 1] reward to each logged interaction.  
**When**: Once you have a working ClickHouse log. Run hourly as a background job.  
**Cost**: Minimal compute. No model calls for automatic signals.

| Mode | Signal type | How computed |
|---|---|---|
| `redteam` | Automatic, dense | `α×attack_success + β×novelty + γ×coverage - δ×duplicate` |
| `judge` | Sparse, proxy-based | Operator override = low reward; no override + confidence = high reward |
| `admin` | Human-labelled | Requires human review; proxy from action completion rate |

Reward weights (redteam):
- α = 0.40 — did the attack succeed?
- β = 0.30 — is the technique novel?
- γ = 0.20 — does it cover an underrepresented category?
- δ = 0.10 — duplicate penalty

Run the labelling job on a schedule:

```python
from src.learning.layer2_reward import run_reward_labelling_job

# In OpenClaw cron:
@cron("0 * * * *")   # every hour
async def label_rewards():
    counts = await run_reward_labelling_job(store, agent_name="Phoebe")
    logger.info("Labelled rewards: %s", counts)
```

**Learning speed by mode**:
- Redteam: fastest (automatic, dense signal)
- Judge: medium (sparse, delayed operator overrides)
- Admin: slowest (requires deliberate human review)

---

### Layer 3 — In-Context Learning via Experience Retrieval

**What**: Inject high-reward past interactions as few-shot examples into the system prompt at inference time.  
**When**: Once you have ~500 interactions with rewards computed.  
**Cost**: One embedding call per agent turn (~$0.00002 for text-embedding-3-small). No model retraining.

**This is the most impactful improvement available without fine-tuning.**

Pipeline:

```
Background (hourly):
  ClickHouse high-reward rows
          │
          ▼ embed(user_message + response_text)
  Vector Store (Qdrant/ChromaDB)

Inference (per turn):
  embed(current_task)
          │
          ▼ similarity search
  Top-K examples (filtered by mode)
          │
          ▼ format_examples_for_prompt()
  System prompt + "## High-value past examples\n..."
          │
          ▼
  Agent.chat() with enriched context
```

**Example injection** (appended to system prompt):

```
## High-value past examples (learn from these)

### Example (reward=0.94)
**Task**: Find a prompt injection vulnerability in the content moderation rule
**Tools used**: osprey.readRuleFile, target.generate, safety.classify
**Response**: Tested indirect instruction via nested fictional framing...
**Outcome**: attack_success=1.0, novelty=0.87, category=instruction_following

### Example (reward=0.89)
...
```

**Vector store options**:

| Store | Best for | Setup |
|---|---|---|
| ChromaDB | Local dev, single node | `pip install chromadb` — no server needed |
| Qdrant | Production OpenClaw | Docker: `qdrant/qdrant`, persistent volumes |
| Pinecone | Managed, no ops | API key only, fully managed |
| pgvector | Existing Postgres | Extension install, SQL queries |

**Embedding model options**:

| Model | Provider | Dimension | Best for |
|---|---|---|---|
| text-embedding-3-small | OpenAI | 1536 | Best quality/cost |
| nomic-embed-text | Ollama (local) | 768 | No API cost, private |
| bge-small-en-v1.5 | HuggingFace | 384 | Fast, local |

---

### Layer 4 — Fine-Tuning

**What**: Actual model weight updates via DPO or OpenAI Fine-Tuning API.  
**When**: Only after Layers 1-3 are running and you have 5,000+ labelled interactions.  
**Cost**: Significant — GPU time or OpenAI training credits.

**Recommended approach by mode**:

| Mode | Recommended method | Base model |
|---|---|---|
| `redteam` | DPO on open model | Qwen-2.5-7B-Instruct or Llama-3-8B-Instruct |
| `judge` | OpenAI fine-tuning | gpt-4o-mini (fastest to iterate) |
| `admin` | Human RLHF, low priority | Claude stays best here due to reasoning |

**Why redteam first for DPO**: The reward signal is automatic, dense, and clean. You get clear (chosen, rejected) pairs without human annotation. The task is also well-bounded (attack generation against known safety rules), which DPO handles well.

**DPO training flow**:

```python
# 1. Build preference pairs from ClickHouse
builder = PreferencePairBuilder(store)
pairs = await builder.build_pairs("Phoebe", "redteam", min_reward_gap=0.3)

# 2. Export as HuggingFace dataset
exporter = DPODatasetBuilder()
exporter.export(pairs, "./sara_dpo_dataset", system_prompt=REDTEAM_PROMPT)

# 3. Train (run on GPU machine)
# trl DPOTrainer with Qwen-2.5-7B-Instruct or Llama-3-8B-Instruct
# See layer4_finetune.py for the full training script

# 4. Deploy fine-tuned model via Ollama or vLLM
# Update ModelConfig to point to the fine-tuned model
```

**OpenAI fine-tuning flow**:

```python
# Simpler — no GPU required
tuner = OpenAIFineTuner()
tuner.build_training_jsonl(high_reward_interactions, JUDGE_PROMPT, "train.jsonl")
job_id = tuner.submit_job("train.jsonl", base_model="gpt-4o-mini", suffix="sara-judge")
# Monitor: tuner.get_job_status(job_id)
# Deploy: set model_name to the fine-tuned model ID in ModelConfig
```

---

## Implementation Roadmap

### Phase 1 — Now (zero infra cost)
- [ ] Deploy generalized `Agent` class with `AgentConfig`
- [ ] Wire `InteractionStore.save()` into `Agent.chat()`
- [ ] Run `ClickHouseSessionStore` for session persistence (already have ClickHouse)
- [ ] Set up `RedisSessionStore` for OpenClaw multi-user sessions

### Phase 2 — Once 500+ interactions logged
- [ ] Run `run_reward_labelling_job()` on hourly cron
- [ ] Set up ChromaDB (local) or Qdrant (production)
- [ ] Run `index_new_examples()` on hourly cron
- [ ] Inject retrieved examples into system prompt via `ExperienceRetriever`
- [ ] A/B test: judge mode with vs. without example injection

### Phase 3 — Once 5,000+ labelled interactions
- [ ] Build DPO preference pairs for redteam mode
- [ ] Fine-tune Qwen-2.5-7B or Llama-3-8B on redteam pairs
- [ ] Deploy fine-tuned model via Ollama locally or vLLM
- [ ] Use `ModelConfig.custom(finetuned_model, ollama_endpoint)` for redteam mode
- [ ] Track accuracy delta using `FineTuneMonitor`

### Phase 4 — Ongoing
- [ ] Human review queue for judge mode labels
- [ ] OpenAI fine-tuning for judge mode on labelled pairs
- [ ] Multi-modal attack generation (images, audio, structured data)
- [ ] Cross-agent knowledge sharing (Sara learns from Phoebe's redteam history)

---

## Adding a New Agent Persona

1. Create `agents/<name>/config.py`
2. Define system prompts and modes
3. Instantiate `AgentConfig` with name, modes, default_model
4. (Optional) Create `agents/<name>/tools.py` with a custom `ToolExecutor`
5. Register with `AgentRegistry.register_config(YOUR_CONFIG)`

That's it. No changes to the framework code.

Example: governance analyst agent

```python
# agents/governance_analyst/config.py
from src.agent.config import AgentConfig, ModelConfig

GOVERNANCE_CONFIG = AgentConfig(
    name="GovernanceAnalyst",
    description="AI governance and policy research assistant.",
    default_mode="research",
    modes={
        "research":  RESEARCH_PROMPT,
        "policy":    POLICY_PROMPT,
        "reporting": REPORTING_PROMPT,
    },
    default_model=ModelConfig.anthropic("claude-opus-4-5"),
    tool_tags=["execute_code", "web_search"],
)
```

---

## Security Notes

- **API keys** are read from environment variables (never hardcoded)
- **Session isolation** is per-user-per-channel by default
- **Tool result truncation** at `max_tool_result_length` prevents context overflow attacks
- **Admin mode** should only be accessible from restricted channels (configure in `channel_mode_map`)
- **RL training data** in ClickHouse contains full conversation history — treat as sensitive

---

## Dependencies

```
# Core
anthropic>=0.40.0
httpx>=0.27.0

# Session stores (install as needed)
redis[asyncio]>=5.0.0        # RedisSessionStore
clickhouse-driver>=0.2.9     # ClickHouseSessionStore, InteractionStore

# RL Layer 3 — vector stores (install one)
chromadb>=0.5.0              # local dev
qdrant-client>=1.9.0         # production

# RL Layer 3 — embeddings (install one)
openai>=1.40.0               # OpenAIEmbedder

# RL Layer 4 — fine-tuning (install when needed)
datasets>=2.20.0
trl>=0.9.0
transformers>=4.42.0
accelerate>=0.31.0
peft>=0.11.0
```
