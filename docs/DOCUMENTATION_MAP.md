# Bamboo Documentation Map

All documentation lives directly in the `docs/` directory.  The `database-plugins/`
subdirectory is the only exception — it contains multiple related reference files for
the database plugin system.

---

## Documentation Files

### [`README.md`](README.md)
Main project documentation.
- Project overview and acronym explanation
- Agent descriptions
- Architecture overview, graph schema
- MCP tools overview
- Technology stack, installation, usage examples

**When to use**: Start here for a project overview.

---

### [`QUICKSTART.md`](QUICKSTART.md)
Step-by-step setup guide.
- Prerequisites and system requirements
- Installation and database configuration
- LLM / API key setup
- Verification procedures
- Running agents and the MCP server
- Troubleshooting

**When to use**: Follow this to set up your environment.

---

### [`DEVELOPMENT.md`](DEVELOPMENT.md)
Developer reference.
- Development setup and code style (PEP 8, async/await, error handling)
- Extension points: node types, relationships, queries, prompts, sub-agents, workflows
- Testing guide (unit tests, integration tests)
- Debugging, performance optimisation, deployment
- Contributing guidelines

**When to use**: Reference this when developing features or extending Bamboo.

---

### [`EXTRACTION_PLUGIN_SYSTEM.md`](EXTRACTION_PLUGIN_SYSTEM.md)
Extraction strategy plugin system.
- Supported strategies: `panda`, `llm`, `rule_based`, `jira`, `github`, `generic`
- How to register a custom strategy
- Strategy selection and factory API

**When to use**: Reference this when adding or choosing an extraction strategy.

---

### [`PANDA_INTEGRATION.md`](PANDA_INTEGRATION.md)
PanDA integration reference.
- `bamboo/utils/panda_client.py` API (`fetch_task_data`)
- Server configuration (`PANDA_URL` / `PANDA_URL_SSL`)
- CLI usage: `--task-id`, `bamboo fetch-task`
- Programmatic usage and interactive mode
- Error handling and testing

**When to use**: Reference this when fetching task data directly from PanDA.

---

### [`database-plugins/`](database-plugins/)
Graph database and vector database plugin system (multiple reference files).
- `DATABASE_PLUGINS.md` — overview
- `DATABASE_PLUGINS_IMPLEMENTATION.md` — implementation guide
- `DATABASE_PLUGINS_EXAMPLES.md` — code examples
- `DATABASE_PLUGINS_QUICK_REFERENCE.md` — quick reference
- `DATABASE_PLUGINS_FINAL_CHECKLIST.md` — checklist
- `DATABASE_PLUGINS_INDEX.md` / `INDEX.md` — index

**When to use**: Reference this when adding a new graph database or vector database backend.

---

## Documentation at a Glance

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `QUICKSTART.md` | Full setup and installation guide |
| `DEVELOPMENT.md` | Developer guide and extension points |
| `EXTRACTION_PLUGIN_SYSTEM.md` | Extraction strategy plugin API |
| `PANDA_INTEGRATION.md` | PanDA live-fetch integration |
| `database-plugins/` | Graph / vector database plugin system |

---

## Reading Paths

### New users
1. `README.md` — understand what Bamboo is
2. `QUICKSTART.md` — set up the environment
3. Run `bamboo-verify`

### PanDA operators
1. `README.md` — project overview
2. `QUICKSTART.md` — environment setup
3. `PANDA_INTEGRATION.md` — fetch tasks directly from PanDA

### Developers
1. `README.md` — project overview
2. `QUICKSTART.md` — environment setup
3. `DEVELOPMENT.md` — architecture and code conventions
4. `EXTRACTION_PLUGIN_SYSTEM.md` — adding extraction strategies
5. `database-plugins/` — adding database backends

---

**Last Updated**: 2026-03-04
