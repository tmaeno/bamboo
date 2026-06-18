---
title: "Quick Start Guide"
---

**BAMBOO**: **B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

This guide will help you get started with Bamboo quickly.

:::tip[New to Bamboo?]
Follow the steps in order. After setup, run `bamboo verify` to confirm your databases, API keys, and embedding model are all reachable before running an agent.
:::

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for local databases)
- OpenAI API key (or Anthropic API key)

## Setup Steps

### 1. Install the Package

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install the package
pip install .

# For development — editable install with dev tools (pytest, black, ruff, mypy).
# Use this (not the plain install above) if you intend to run the test suite;
# see the Development Guide.
pip install -e ".[dev]"
```

### 2. Verify the Installation

```bash
# Works from any directory after pip install .
bamboo verify
```

This checks that all modules import correctly, CLI entry points are registered, all dependencies are present, and that Python has a working CA trust store (auto-installing the bundled `certifi` roots into your `.env` if it is empty — this is what makes HTTPS, including `@bamboo login`, verify correctly).

`bamboo verify` prints the exact path to the installed `.env.example`. Copy it and fill in your keys:

```bash
cp <path-shown-by-bamboo-verify> .env
```

Then edit `.env`.  The minimum required settings depend on which LLM and embeddings backend you choose:


#### OpenAI Option — OpenAI for both LLM and embeddings (default, paid)

```env
LLM_API_KEY=sk-...          # Your OpenAI API key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
EMBEDDINGS_PROVIDER=openai  # default — reuses LLM_API_KEY automatically
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

#### Hybrid Option — Anthropic LLM + OpenAI embeddings (both paid)

> **Note:** `LLM_PROVIDER=anthropic` requires a **Claude API subscription**
> (api.anthropic.com), not a claude.ai consumer subscription — they are
> separate products with separate billing.

```env
LLM_API_KEY=sk-ant-...      # Your Anthropic API key
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
EMBEDDINGS_PROVIDER=openai  # Anthropic has no embeddings API
EMBEDDINGS_API_KEY=sk-...   # Separate OpenAI key for embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

#### Free Option — Fully free: Ollama (local LLM) + local embeddings

No API keys required at all.

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull a model (one-time download, ~2-8GB depending on model)
ollama pull llama3.2
# 3. Keep the server running in a separate terminal
ollama serve
# 4. Install the extra dependencies for local LLM and Ollama
pip install langchain-ollama sentence-transformers langchain-huggingface
# Install the extra dependencies for local embeddings
pip install sentence-transformers langchain-huggingface
```

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2          # or mistral, gemma3, etc.
EMBEDDINGS_PROVIDER=local   # runs sentence-transformers in-process, no API key
EMBEDDING_MODEL=all-MiniLM-L6-v2   # fast (384-dim); or all-mpnet-base-v2 (768-dim)
EMBEDDING_DIMENSION=384            # must match EMBEDDING_MODEL
# Optional but recommended: avoids HuggingFace Hub rate-limiting on model downloads.
# Get a free token at https://huggingface.co/settings/tokens
HF_TOKEN=
```

> **Many external MCP servers?** When configured MCP servers add up to hundreds of tools, the
> orchestration tool list is budgeted (a truncation backstop plus a relevance cap) and tunable via
> `LLM_CONTEXT_WINDOW` / `TOOL_MAX_FULL_SCHEMAS` and related settings. See
> [Agent Reference → Bounding the tool list for large catalogues](/bamboo/architecture/agents/).

> **Contributing?** If you are working inside the project source tree and haven't installed yet, you can also run `python verify_installation.py` from the project root as a fallback.

### 3. Start Database Services

```bash
# Start graph database and vector database using Docker Compose
docker-compose up -d

# Wait about 10 seconds for services to be ready
# Graph database will be available at http://localhost:7474
# Vector database will be available at http://localhost:6333
```

Or use the Makefile:

```bash
make docker-up
```

### 4. Test the Setup

```bash
bamboo interactive
```

## Usage Examples

### Example 1: Populate Knowledge Base

```bash
bamboo populate \
  --email-thread examples/sample_email.txt \
  --task-data examples/sample_task.json \
  --external-data examples/sample_external.json
```

### Example 2: Analyze a Task

```bash
bamboo analyze \
  --task-data examples/sample_task.json \
  --external-data examples/sample_external.json \
  --output results.json
```

### Example 3: Fetching tasks from PanDA

If your environment has access to a PanDA server, you can fetch task data directly instead
of loading a local JSON file.

**Inspect raw task data:**

```bash
bamboo fetch-task 12345
bamboo fetch-task 12345 --output task_12345.json
bamboo fetch-task 12345 --verbose   # sets bamboo logger to DEBUG
```

**Populate and analyze using a live task ID:**

```bash
bamboo populate --task-id 12345
bamboo populate --task-id 12345 --email-thread incident.txt
bamboo analyze --task-id 12345
bamboo analyze --task-id 12345 --output result.json
```

`--task-id` and `--task-data` are mutually exclusive on both commands.

See `docs/PANDA_INTEGRATION.md` for server configuration and authentication setup.

---

### Example 4: Interactive Mode

```bash
bamboo interactive
```

The interactive mode provides:
- Knowledge base population wizard
- Task analysis wizard
- Knowledge graph querying
- Human-in-the-loop review

## Troubleshooting

### Database Connection Issues

```bash
# Check if databases are running
docker ps

# Check graph database logs
docker logs bamboo-graph_db-1

# Check vector database logs
docker logs bamboo-vector_db-1
```

### API Key Issues

```bash
# Check which keys are set in your .env
grep API_KEY .env

# Check which provider and model are configured
python -c "
from bamboo.config import get_settings
s = get_settings()
print('llm_provider        :', s.llm_provider)
print('llm_model           :', s.llm_model)
print('llm_api_key         :', 'set' if s.llm_api_key else 'MISSING')
print('embeddings_provider :', s.embeddings_provider)
print('embeddings_api_key  :', 'set' if s.effective_embeddings_api_key else 'MISSING (not needed for local)')
"

# Test that the LLM client can be constructed (does not make a network call)
python -c "from bamboo.llm.llm_client import get_llm; print(get_llm())"
```

Common errors:

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` / `401` | Wrong or missing API key | Check `LLM_API_KEY` (and `EMBEDDINGS_API_KEY` when `EMBEDDINGS_PROVIDER=openai`) in `.env` |
| `ValueError: llm_provider must be...` | Invalid `LLM_PROVIDER` value | Set `LLM_PROVIDER=openai`, `anthropic`, or `ollama` |
| `llm_api_key` is empty | `.env` not loaded | Make sure `.env` is in the working directory where you run Bamboo |
| `RateLimitError` | API quota exceeded | Check your API plan or wait and retry |
| `ImportError: sentence-transformers` | Missing local-embeddings deps | Run `pip install sentence-transformers langchain-huggingface` |
| `ImportError: langchain-ollama` | Missing Ollama deps | Run `pip install langchain-ollama` |
| Ollama connection refused | Ollama server not running | Run `ollama serve` in a separate terminal |
| HuggingFace Hub rate-limit warning | `HF_TOKEN` not set | Add `HF_TOKEN=<token>` to `.env` — free token at https://huggingface.co/settings/tokens |

### Import Errors

```bash
# Reinstall the package
pip install .

# Or in editable/development mode
pip install -e .
```

## Learn more

- **Architecture, agents, and MCP tool budgeting** → [Agent Reference](/bamboo/architecture/agents/)
- **Graph schema** (node & relationship types) → [Graph Schema](/bamboo/architecture/schema/)
- **Extending Bamboo** (custom node types, queries, prompts), testing, and deployment → [Development Guide](/bamboo/development/)
- **Fetching tasks from PanDA** → [PanDA Integration](/bamboo/integrations/panda-integration/)
- **Live human-in-the-loop investigation** → [Co-Investigation Mode](/bamboo/guides/investigate/)

## Next Steps

1. **Populate your knowledge base** with real email threads and task data
2. **Analyze problematic tasks** and review the generated explanations
3. **Provide feedback** to improve the system's accuracy
4. **Customize prompts** for your specific domain
5. **Extend the graph schema** with domain-specific nodes
