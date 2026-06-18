# Development Guide

This guide covers development setup, code organization, and extension points for the Bamboo project.

## Project Structure

```
bamboo/
├── bamboo/                # Main package
│   ├── agents/            # AI agents and sub-agents
│   │   ├── extractors/                 # Knowledge extraction strategies (panda / llm / …)
│   │   ├── helpers/                    # Shared agent helpers (e.g. tool selection)
│   │   ├── knowledge_accumulator.py    # Orchestrates extraction + review loop
│   │   ├── knowledge_reviewer.py       # LLM quality gate
│   │   ├── context_enricher.py         # MCP tool orchestration / context enrichment
│   │   └── reasoning_navigator.py      # Diagnoses tasks against the knowledge base
│   ├── database/          # Graph + vector database clients
│   │   └── backends/                   # neo4j_backend.py, qdrant_backend.py
│   ├── llm/               # LLM client (llm_client.py) and prompts (prompts.py)
│   ├── mcp/               # MCP client layer (built-in + external servers)
│   ├── models/            # Data models (graph_element.py defines NodeType / RelationType)
│   ├── frontends/         # User-facing frontends (e.g. Mattermost)
│   ├── workflows/         # LangGraph workflows
│   ├── scripts/           # CLI scripts
│   ├── utils/             # Utilities
│   ├── cli.py             # CLI entry point
│   └── config.py          # Configuration
├── tests/                 # Test suite
├── examples/              # Example data
└── docker-compose.yml     # Dev database services (Neo4j + Qdrant)
```

## Development Setup

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Start Development Databases

```bash
# Start graph database and vector database
docker-compose up -d

# Verify services are running
docker ps
```

### 3. Configure Environment

```bash
# Get the installed .env.example path and copy it
cp "$(python -c "import importlib.resources; print(importlib.resources.files('bamboo.data').joinpath('.env.example'))")" .env

# Edit .env with your settings (see the Quick Start for the per-backend options)
```

> Each extraction attempt is evaluated by `KnowledgeReviewer` before being written to the
> databases.  If the reviewer rejects the graph, `ContextEnricher` fires once to fetch
> additional PanDA data sources (parent task, retry chain, job diagnostics, error-dialog logs)
> and the extraction is retried with the enriched context and reviewer feedback (up to 2
> retries total).  See [Agent Reference](AGENTS.md) for the full pipeline.

### 4. Verify Setup

```bash
pytest                 # run the test suite (see Testing below)
black --check bamboo/  # formatting
ruff check bamboo/     # linting
mypy bamboo/           # type checking
```

## Code Style

- **Formatting:** Black, 88-character line length — `black bamboo/`.
- **Linting:** Ruff — `ruff check bamboo/`.
- **Types:** use type hints; check with `mypy bamboo/`.
- **Async:** use `async`/`await` for all I/O (database, LLM, MCP calls).

## Extension Points

Bamboo is designed to be extended by editing a small number of well-defined modules. Rather than
duplicate code samples here (which drift from the source), the table below points at the real file
to edit for each kind of extension — read the current code there for the exact API.

| To add… | Edit | Notes |
|---|---|---|
| A node or relationship type | [`bamboo/models/graph_element.py`](https://github.com/tmaeno/bamboo/blob/master/bamboo/models/graph_element.py) | Defines the `NodeType` and `RelationType` enums and node models. See [Graph Schema](SCHEMA.md) for the existing catalogue. |
| Extraction behaviour for a node type | [`bamboo/agents/extractors/knowledge_graph_extractor.py`](https://github.com/tmaeno/bamboo/blob/master/bamboo/agents/extractors/knowledge_graph_extractor.py) | The dispatcher that selects the active strategy and assigns node UUIDs. |
| A new extraction **strategy** | see [Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md) | Register a custom extractor (`panda`, `llm`, `rule_based`, …) without touching the dispatcher. |
| Custom LLM prompts | [`bamboo/llm/prompts.py`](https://github.com/tmaeno/bamboo/blob/master/bamboo/llm/prompts.py) | Includes the extraction prompt with the canonical relationship list. |
| Custom database queries | [`bamboo/database/backends/neo4j_backend.py`](https://github.com/tmaeno/bamboo/blob/master/bamboo/database/backends/neo4j_backend.py) | Backend implementation; the higher-level client is `bamboo/database/graph_database_client.py`. See [Database Plugin System](database-plugins/INDEX.md) to add a whole new backend. |
| A new database backend | see [Database Plugin System](database-plugins/INDEX.md) | Plugin architecture for graph/vector backends. |
| A LangGraph workflow | [`bamboo/workflows/`](https://github.com/tmaeno/bamboo/tree/master/bamboo/workflows) | Existing workflows are the best template. |

## Testing

```bash
# Run all tests (must be run from the repo root)
pytest

# Run a specific test file
pytest tests/test_extractors.py -v

# Run with coverage
pytest tests/ --cov=bamboo --cov-report=html

# Integration tests only / everything except integration
pytest tests/ -v -m integration
pytest tests/ -v -m "not integration"
```

> **Run `pytest` from the repo root.** `pytest` with no arguments reads `pyproject.toml`, which sets
> `testpaths = ["tests"]` and `asyncio_mode = "auto"`. Running it from another directory fails with
> `file or directory not found: tests/`. The `tests/` directory is part of the source tree, not the
> installed package, so an editable install (`pip install -e ".[dev]"`) is required.

Note: some tests require actual API keys and database connections.

## Debugging

### Database Inspection

```bash
# Graph database browser
open http://localhost:7474

# Cypher query examples
MATCH (n) RETURN n LIMIT 25;
MATCH (c:Cause) RETURN c.name, c.frequency ORDER BY c.frequency DESC;
MATCH (s:Symptom)-[r:indicate]->(c:Cause) RETURN s, r, c;

# Vector database inspection
curl http://localhost:6333/collections
curl http://localhost:6333/collections/bamboo_knowledge
```

Set `LOG_LEVEL=DEBUG` in `.env` (or pass `--verbose` to CLI commands) for verbose logging.

## Deployment

### Container images

The repository ships one multi-stage `Dockerfile` with two targets:

- **`bamboo`** — the application image, configured entirely by environment variables
  (`NEO4J_URI`, `QDRANT_URL`, `LLM_PROVIDER` + key or `OLLAMA_BASE_URL`,
  `EMBEDDINGS_*`). It talks to *external* Neo4j / Qdrant / LLM services. This is the
  artifact for a normal Docker deployment where the services run elsewhere.

  ```bash
  docker build --target bamboo -t bamboo .
  docker run --rm --env-file .env bamboo analyze --task-data task.json --output out.json
  ```

- **`bamboo-batch-analyze`** — `FROM bamboo`, additionally bundles Neo4j + Qdrant +
  Ollama so a single container is self-sufficient on an air-gapped batch node. It is
  converted to an Apptainer `.sif` for offline batch execution.

CI builds and pushes both images to GHCR on tags / manual dispatch — see
`.github/workflows/build-images.yml`.

For the air-gapped Apptainer batch deployment (KB/model staging on the shared
filesystem, `bamboo batch-analyze`, CPU/GPU queues), see **[BATCH.md](BATCH.md)**.

> The legacy `docker-compose.yml` at the repo root is a dev-only convenience for
> standing up Neo4j + Qdrant locally; it is not used for deployment.

### Production considerations

- Never commit `.env` files; use a secrets manager for keys.
- Use persistent volumes for the databases.
- The LLM clients already retry; tune rate limits to your API plan.

## Contributing

1. Branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. `pytest` · `black bamboo/` · `ruff check bamboo/`
4. Commit using conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
5. Push and open a pull request

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Graph Schema

The knowledge graph node and relationship types have moved to a dedicated reference:
**[Graph Schema](SCHEMA.md)**.
