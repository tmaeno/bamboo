# Bamboo Documentation

Welcome to the Bamboo documentation.

## 📚 Documents

### 🚀 [Quick Start Guide](QUICKSTART.md)
Step-by-step guide to install Bamboo, configure databases and API keys, and run your first agent.
- Prerequisites and system requirements
- Installation and database configuration
- LLM / API key setup, verification procedures
- Running agents and the MCP server, troubleshooting

### 🔌 [Database Plugin System](database-plugins/INDEX.md)
Graph database and vector database plugin architecture, backend configuration, and how to add new backends.

### 🧩 [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md)
Pluggable knowledge extraction strategies (`panda`, `llm`, `rule_based`, …), how to select one, and how to register a custom extractor.

### 🐼 [PanDA Integration](PANDA_INTEGRATION.md)
Fetch task and job data live from PanDA — no local JSON file required.

### 🗄️ [Knowledge Population](KNOWLEDGE_POPULATION.md)
Two paths for populating the graph and vector databases: individual (`bamboo populate`)
for live incidents, and batch (`seed-drafts → review-drafts → batch-populate`) for
bulk commissioning.

### 🔎 [Task Analysis](ANALYZE.md)
`bamboo analyze` — diagnose a failing task against the accumulated knowledge base:
retrieve matching causes + similar past incidents, synthesize a root cause /
confidence / resolution, re-run stored procedures, and route novel incidents to
the draft pipeline. Includes pattern mode (common subgraph across tasks).

### 🔬 [Co-Investigation Mode](INVESTIGATE.md)
`bamboo investigate` — live human-in-the-loop dialog for an ongoing incident. Each tool
turn produces a sandboxed orchestration code block stored on the resulting Procedure
node, so future analyze runs re-execute the exact code that worked last time.

### 💬 [Mattermost Integration](MATTERMOST.md)
The ops-facing chat frontend: a bot (`bamboo serve-mattermost`) that runs
`investigate`, `capture`, and analysis posting from Mattermost. Covers deployment
topology, bot setup, the two tokens (Mattermost vs. CERN-IAM PanDA OIDC),
per-user `login`, authorization, and troubleshooting.

### 🤖 [Agent Reference](AGENTS.md)
All agents and sub-agents — responsibilities, inputs/outputs, configuration, and failure handling for `KnowledgeAccumulator`, `KnowledgeReviewer`, `ContextEnricher`, `ReasoningNavigator`, and their components.

### 👨‍💻 [Development Guide](DEVELOPMENT.md)
Development setup, code style, extension points (agents, node types, relationships, prompts), testing, and deployment.

---

## Reading Paths

### New users
1. [Quick Start Guide](QUICKSTART.md) — set up the environment
2. Run `bamboo verify`

### PanDA operators
1. [Quick Start Guide](QUICKSTART.md) — environment setup
2. [PanDA Integration](PANDA_INTEGRATION.md) — fetch tasks directly from PanDA
3. [Knowledge Population](KNOWLEDGE_POPULATION.md) — populate databases from live tasks or CSV batches
4. [Co-Investigation Mode](INVESTIGATE.md) — live human-in-the-loop investigation; the captured procedures auto-replay on similar future incidents
5. [Task Analysis](ANALYZE.md) — diagnose a failing task against the populated knowledge base

### Ops users (Mattermost)
1. [Mattermost Integration](MATTERMOST.md) — use bamboo from chat: `investigate`,
   `capture`, and per-user `login` (no local install required)

### Bot operators (deploying the Mattermost bot)
1. [Quick Start Guide](QUICKSTART.md) — base environment (LLM, Neo4j, Qdrant)
2. [PanDA Integration](PANDA_INTEGRATION.md) — PanDA/OIDC setup
3. [Mattermost Integration](MATTERMOST.md) — create the bot, configure, and run `bamboo serve-mattermost`

### Developers
1. [Quick Start Guide](QUICKSTART.md) — environment setup
2. [Agent Reference](AGENTS.md) — understand the pipeline and each agent's role
3. [Development Guide](DEVELOPMENT.md) — architecture and code conventions
4. [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md) — adding extraction strategies
5. [Database Plugin System](database-plugins/INDEX.md) — adding database backends
