# Quick Start Guide

**BAMBOO**: **B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

This guide will help you get started with Bamboo quickly.

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

# For development (includes pytest, black, ruff, mypy)
pip install ".[dev]"
```

### 2. Verify the Installation

```bash
# Works from any directory after pip install .
bamboo-verify
```

This checks that all modules import correctly, CLI entry points are registered, and all dependencies are present.

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

### 4. Configure Environment

`bamboo-verify` prints the exact path to the installed `.env.example`. Copy it and fill in your API keys:

```bash
# bamboo-verify shows the exact path — copy it here
cp <path-shown-by-bamboo-verify> .env

# Edit .env and add your API keys
# At minimum, set:
# - OPENAI_API_KEY (or ANTHROPIC_API_KEY)
# - NEO4J_PASSWORD (default: password)
```

Alternatively, retrieve the path directly:

```bash
python -c "import importlib.resources; print(importlib.resources.files('bamboo.data').joinpath('.env.example'))"
```

### 5. Test the Setup

```bash
# Start the interactive CLI
python -m bamboo.cli interactive
```

## Usage Examples

### Example 1: Populate Knowledge Base

Using sample data:

```bash
python -m bamboo.scripts.populate_knowledge \
  --email-thread examples/sample_email.txt \
  --task-data examples/sample_task.json \
  --external-data examples/sample_external.json
```

Or using the CLI shortcut:

```bash
bamboo-populate \
  --email-thread examples/sample_email.txt \
  --task-data examples/sample_task.json
```

### Example 2: Analyze a Task

```bash
python -m bamboo.scripts.analyze_task \
  --task-data examples/sample_task.json \
  --external-data examples/sample_external.json \
  --output results.json
```

Or using the CLI shortcut:

```bash
bamboo-analyze \
  --task-data examples/sample_task.json \
  --output results.json
```

### Example 3: Interactive Mode

```bash
# Start interactive mode
python -m bamboo.cli interactive

# Or use the installed command
bamboo interactive
```

The interactive mode provides:
- Knowledge base population wizard
- Task analysis wizard
- Knowledge graph querying
- Human-in-the-loop review

## Architecture Overview

### Knowledge Extraction Agent

The Knowledge Extraction Agent processes input data in several steps:

1. **Extraction**: Uses LLM to extract structured graph from unstructured data
2. **Job aggregation**: Aggregates raw job records into stable `Job_Feature` nodes (dominant site, failure rates, CPU ranges, dominant pilot version, etc.)
3. **Canonicalization**: Normalizes node names to avoid duplicates
4. **Graph Storage**: Stores nodes and relationships in the graph database
5. **Summarization**: Generates summary using LLM
6. **Vector Storage**: Stores summary and key insights in the vector database

### Reasoning Agent

The Reasoning Agent analyzes tasks in these steps:

1. **Feature Extraction**: Extracts key features from task data
2. **Graph Query**: Queries the graph database for matching causes and resolutions
3. **Vector Search**: Finds similar cases in the vector database
4. **Analysis**: Uses LLM to determine root cause
5. **Email Generation**: Creates human-readable explanation
6. **Human Review**: Allows operator to review and provide feedback

## LangGraph Workflows

Both agents are implemented as LangGraph workflows:

### Knowledge Workflow
```
[Start] → [Extract Knowledge] → [Validate] → [End]
```

### Reasoning Workflow
```
[Start] → [Analyze Task] → [Human Review] → [Send/Revise]
                              ↑                    |
                              └────────────────────┘
```

## Sub-Agents

The system includes specialized sub-agents:

### In Knowledge Accumulation Agent:
- **Knowledge Extraction** - Extracts knowledge graph from structured and unstructured data
- **Graph Summarization** - Summarizes graph data for quick insights
- **Node Canonicalization** - Transforms diverse node data into a canonical format
- **Feature Classification** - Classifies node features for better reasoning

### In Reasoning Navigation Agent:
- **Knowledge Extraction** - Extracts features from task data for querying
- **Output Synthesis** - Synthesizes outputs from various agents into coherent responses

## Graph Schema

The core knowledge graph schema used by the incident-analysis pipeline:

```
Symptom       -[indicate]->        Cause  -[solved_by]->  Resolution
Environment   -[associated_with]-> Cause
Task_Feature  -[contribute_to]->   Cause
Job_Feature   -[contribute_to]->   Cause
Task_Context  -[contribute_to]->   Cause
Component     -[originated_from]-> Cause
Symptom       -[has_job_pattern]-> Job_Feature
```

### Core Node Types
| Node | Description |
|------|-------------|
| `Symptom` | Observed failure class (e.g. error message category) |
| `Cause` | Root cause of the incident |
| `Resolution` | Solution or fix applied |
| `Environment` | External factor contributing to the cause (e.g. OS, runtime version) |
| `Task_Feature` | Discrete task *configuration* attribute stored as `attribute=value` (e.g. `coreCount=8`) |
| `Job_Feature` | Aggregated job *execution* pattern stored as `attribute=value` (e.g. `site_failure_rate=AGLT2:high(>50%)`, `cpuConsumptionTime=1-6h`) |
| `Component` | System component where the cause originated |
| `Task_Context` | Free-form prose context — stored in vector database only, not in graph |

### Core Relationships
| Relationship | From → To | Description |
|---|---|---|
| `indicate` | Symptom → Cause | Symptom points to a root cause |
| `solved_by` | Cause → Resolution | Cause is resolved by a resolution |
| `contribute_to` | Task_Feature / Job_Feature / Task_Context → Cause | Feature or context contributes to a cause |
| `originated_from` | Component → Cause | Cause originated in a component |
| `associated_with` | Environment → Cause | Cause associated with an external factor |
| `has_job_pattern` | Symptom → Job_Feature | Links a task-level symptom to the aggregated job-execution patterns observed across the failing jobs |

### Log-level distinction

Two separate log parameters flow through the pipeline:

| Parameter | Source | Node tag |
|---|---|---|
| `task_logs` | Orchestration services (JEDI, Harvester, …) | `log_level="task"` |
| `job_logs` | Execution workers (pilot, Athena/payload, …) | `log_level="job"` |

Both are processed by the same LLM log-extraction path but every extracted node is tagged with its provenance level, allowing graph queries to filter by origin.

> Extended node types (Metric, Anomaly, Issue, System, Pattern, Optimization, Event, Action, Dependency, User) and relationships are available for future extraction strategies.

## Customization

### Adding Custom Node Types

Edit `bamboo/models/graph.py` to add new node types:

```python
class CustomNode(BaseNode):
    node_type: NodeType = NodeType.CUSTOM
    custom_field: str
```

### Custom LLM Prompts

Edit `bamboo/llm/prompts.py` to customize prompts:

```python
CUSTOM_PROMPT = """
Your custom prompt here...
{variable_placeholder}
"""
```

### Custom Database Queries

Extend the graph database client in `bamboo/database/backends/neo4j_backend.py`:

```python
async def custom_query(self, params):
    async with self.driver.session() as session:
        query = "MATCH (n) WHERE ... RETURN n"
        result = await session.run(query, **params)
        return await result.values()
```

## Testing

The `tests/` directory is part of the **source tree**, not the installed package, so tests must always be run from the **project root** with an editable install:

```bash
# 1. Clone / cd into the source tree
cd /path/to/bamboo

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Run tests
pytest
```

`pytest` with no arguments reads `pyproject.toml` from the current directory, which sets `testpaths = ["tests"]` and `asyncio_mode = "auto"`. Running `pytest tests/ -v` from any other directory will fail with `file or directory not found: tests/`.

Note: Some tests require actual API keys and database connections.

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
print('provider :', s.llm_provider)
print('model    :', s.llm_model)
print('openai   :', 'set' if s.openai_api_key else 'MISSING')
print('anthropic:', 'set' if s.anthropic_api_key else 'MISSING')
"

# Verify the key for the configured provider is actually set
python -c "
from bamboo.config import get_settings
s = get_settings()
if s.llm_provider == 'openai' and not s.openai_api_key:
    print('ERROR: LLM_PROVIDER=openai but OPENAI_API_KEY is not set')
elif s.llm_provider == 'anthropic' and not s.anthropic_api_key:
    print('ERROR: LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set')
else:
    print('OK: API key for', s.llm_provider, 'is set')
"

# Test that the LLM client can be constructed (does not make a network call)
python -c "from bamboo.llm.llm_client import get_llm; print(get_llm())"

# Test a real API call (requires a valid key and network access)
python -c "
import asyncio
from bamboo.llm.llm_client import get_llm
from langchain_core.messages import HumanMessage
async def test():
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content='Say hello.')])
    print('LLM response:', response.content[:80])
asyncio.run(test())
"
```

Common errors:

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` / `401` | Wrong or missing API key | Check `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in `.env` |
| `ValueError: llm_provider must be...` | Invalid `LLM_PROVIDER` value | Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` |
| `openai_api_key` is empty | `.env` not loaded | Make sure `.env` is in the working directory where you run Bamboo |
| `RateLimitError` | API quota exceeded | Check your API plan or wait and retry |

### Import Errors

```bash
# Reinstall the package
pip install .

# Or in editable/development mode
pip install -e .
```

## Next Steps

1. **Populate your knowledge base** with real email threads and task data
2. **Analyze problematic tasks** and review the generated explanations
3. **Provide feedback** to improve the system's accuracy
4. **Customize prompts** for your specific domain
5. **Extend the graph schema** with domain-specific nodes
