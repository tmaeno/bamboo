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

### 🤖 [Agent Reference](AGENTS.md)
All agents and sub-agents — responsibilities, inputs/outputs, configuration, and failure handling for `KnowledgeAccumulator`, `KnowledgeReviewer`, `ExtraSourceExplorer`, `ReasoningNavigator`, and their components.

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

### Developers
1. [Quick Start Guide](QUICKSTART.md) — environment setup
2. [Agent Reference](AGENTS.md) — understand the pipeline and each agent's role
3. [Development Guide](DEVELOPMENT.md) — architecture and code conventions
4. [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md) — adding extraction strategies
5. [Database Plugin System](database-plugins/INDEX.md) — adding database backends
