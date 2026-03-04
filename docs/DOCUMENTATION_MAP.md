# Bamboo Documentation Map

All documentation lives directly in the `docs/` directory.  The `database-plugins/`
subdirectory is the only exception — it contains multiple related reference files for
the database plugin system.

---

## Documentation Files

### [Project Overview](README.md)
Main project documentation.
- Project overview and acronym explanation
- Agent descriptions
- Architecture overview, graph schema
- MCP tools overview
- Technology stack, installation, usage examples

**When to use**: Start here for a project overview.

---

### [Quick Start Guide](QUICKSTART.md)
Step-by-step setup guide.
- Prerequisites and system requirements
- Installation and database configuration
- LLM / API key setup
- Verification procedures
- Running agents and the MCP server
- Troubleshooting

**When to use**: Follow this to set up your environment.

---

### [Development Guide](DEVELOPMENT.md)
Developer reference.
- Development setup and code style (PEP 8, async/await, error handling)
- Extension points: node types, relationships, queries, prompts, sub-agents, workflows
- Testing guide (unit tests, integration tests)
- Debugging, performance optimisation, deployment
- Contributing guidelines

**When to use**: Reference this when developing features or extending Bamboo.

---

### [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md)
Extraction strategy plugin system.
- Supported strategies: `panda`, `llm`, `rule_based`, `jira`, `github`, `generic`
- How to register a custom strategy
- Strategy selection and factory API

**When to use**: Reference this when adding or choosing an extraction strategy.

---

### [PanDA Integration](PANDA_INTEGRATION.md)
PanDA integration reference.
- `bamboo/utils/panda_client.py` API (`fetch_task_data`)
- Server configuration (`PANDA_URL` / `PANDA_URL_SSL`)
- CLI usage: `--task-id`, `bamboo fetch-task`
- Programmatic usage and interactive mode
- Error handling and testing

**When to use**: Reference this when fetching task data directly from PanDA.

---

### [Database Plugin System](database-plugins/INDEX.md)
Graph database and vector database plugin system (multiple reference files).
- `DATABASE_PLUGINS.md` — overview
- `DATABASE_PLUGINS_IMPLEMENTATION.md` — implementation guide
- `DATABASE_PLUGINS_EXAMPLES.md` — code examples
- `DATABASE_PLUGINS_QUICK_REFERENCE.md` — quick reference
- `DATABASE_PLUGINS_FINAL_CHECKLIST.md` — checklist
- `DATABASE_PLUGINS_INDEX.md` / `INDEX.md` — index

**When to use**: Reference this when adding a new graph database or vector database backend.

---


## Reading Paths

### New users
1. [Project Overview](README.md) — understand what Bamboo is
2. [Quick Start Guide](QUICKSTART.md) — set up the environment
3. Run `bamboo-verify`

### PanDA operators
1. [Project Overview](README.md) — project overview
2. [Quick Start Guide](QUICKSTART.md) — environment setup
3. [PanDA Integration](PANDA_INTEGRATION.md) — fetch tasks directly from PanDA

### Developers
1. [Project Overview](README.md) — project overview
2. [Quick Start Guide](QUICKSTART.md) — environment setup
3. [Development Guide](DEVELOPMENT.md) — architecture and code conventions
4. [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md) — adding extraction strategies
5. [Database Plugin System](database-plugins/INDEX.md) — adding database backends

---

**Last Updated**: 2026-03-04
