# Bamboo Documentation

Welcome to the Bamboo documentation. This is your hub for all project documentation.

## 📚 Documentation

### 🚀 [Quick Start Guide](QUICKSTART.md)
Step-by-step guide to install Bamboo, configure databases and API keys, and run your first agent.

### 🔌 [Database Plugin System](database-plugins/INDEX.md)
Graph database (Neo4j) and vector database (Qdrant) plugin architecture, backend configuration, and how to add new backends.

### 🧩 [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md)
Pluggable knowledge extraction strategies (`panda`, `llm`, `rule_based`, …), how to select one, and how to register a custom extractor.

### 🐼 [PanDA Integration](PANDA_INTEGRATION.md)
Fetch task data live from a PanDA server by `jediTaskID` — no local JSON file required.  Covers `bamboo fetch-task`, `--task-id`, server configuration, and error handling.

### 👨‍💻 [Development Guide](DEVELOPMENT.md)
Development setup, code style, extension points (agents, node types, relationships, prompts), testing, and deployment.

---

See [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) for a full index and recommended reading paths.
