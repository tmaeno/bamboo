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
Graph database (Neo4j) and vector database (Qdrant) plugin architecture, backend configuration, and how to add new backends.

### 🧩 [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md)
Pluggable knowledge extraction strategies (`panda`, `llm`, `rule_based`, …), how to select one, and how to register a custom extractor.

### 🐼 [PanDA Integration](PANDA_INTEGRATION.md)
Fetch task data live from a PanDA server by `jediTaskID` — no local JSON file required.
- PanDA configuration (`PANDA_URL` / `PANDA_URL_SSL`)
- CLI usage: `--task-id`, `bamboo fetch-task`
- Programmatic usage, interactive mode, error handling and testing

### 👨‍💻 [Development Guide](DEVELOPMENT.md)
Development setup, code style, extension points (agents, node types, relationships, prompts), testing, and deployment.

---

## Reading Paths

### New users
1. [Quick Start Guide](QUICKSTART.md) — set up the environment
2. Run `bamboo-verify`

### PanDA operators
1. [Quick Start Guide](QUICKSTART.md) — environment setup
2. [PanDA Integration](PANDA_INTEGRATION.md) — fetch tasks directly from PanDA

### Developers
1. [Quick Start Guide](QUICKSTART.md) — environment setup
2. [Development Guide](DEVELOPMENT.md) — architecture and code conventions
3. [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md) — adding extraction strategies
4. [Database Plugin System](database-plugins/INDEX.md) — adding database backends
