# Quick Start Guide

**BAMBOO**: **B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

This guide will help you get started with Bamboo quickly.

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for local databases)
- OpenAI API key (or Anthropic API key)

## Setup Steps

### 1. Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install the package in development mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Start Database Services

```bash
# Start Neo4j and Qdrant using Docker Compose
docker-compose up -d

# Wait about 10 seconds for services to be ready
# Neo4j will be available at http://localhost:7474
# Qdrant will be available at http://localhost:6333
```

Or use the Makefile:

```bash
make docker-up
```

### 3. Configure Environment

```bash
# Copy the example environment file
cp examples/.env.example .env

# Edit .env and add your API keys
# At minimum, set:
# - OPENAI_API_KEY (or ANTHROPIC_API_KEY)
# - NEO4J_PASSWORD (default: password)
```

### 4. Test the Setup

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
2. **Canonicalization**: Normalizes node names to avoid duplicates
3. **Graph Storage**: Stores nodes and relationships in Neo4j
4. **Summarization**: Generates summary using LLM
5. **Vector Storage**: Stores summary and key insights in Qdrant

### Reasoning Agent

The Reasoning Agent analyzes tasks in these steps:

1. **Feature Extraction**: Extracts key features from task data
2. **Graph Query**: Queries Neo4j for matching causes and resolutions
3. **Vector Search**: Finds similar cases in Qdrant
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

### In Knowledge Extraction Agent:
- **GraphExtractor**: Extracts knowledge graphs from text
- **FeatureExtractor**: Extracts features from structured data
- **Canonicalizer**: (via LLM) Normalizes node names
- **Summarizer**: (via LLM) Creates summaries

### In Reasoning Agent:
- **FeatureExtractor**: Extracts features from task data
- **CauseIdentifier**: (via LLM) Identifies root causes
- **EmailGenerator**: (via LLM) Generates explanatory emails

## Graph Schema

The knowledge graph follows this structure:

```
┌───────────┐
│   Error   │
└─────┬─────┘
      │ indicate
      ↓
┌───────────┐     ┌────────────────┐
│   Cause   │────→│  Resolution    │
└─────┬─────┘     └────────────────┘
      ↑              solved_by
      │
      ├── originated_from ── Component
      ├── associated_with ── Environment
      └── contribute_to ─── Feature
```

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

Extend `Neo4jClient` in `bamboo/database/neo4j_client.py`:

```python
async def custom_query(self, params):
    async with self.driver.session() as session:
        query = "MATCH (n) WHERE ... RETURN n"
        result = await session.run(query, **params)
        return await result.values()
```

## Testing

Run tests:

```bash
pytest tests/ -v
```

Note: Some tests require actual API keys and database connections.

## Troubleshooting

### Database Connection Issues

```bash
# Check if databases are running
docker ps

# Check Neo4j logs
docker logs bamboo-graph_db-1

# Check Qdrant logs
docker logs bamboo-vector_db-1
```

### API Key Issues

```bash
# Verify your .env file
cat .env | grep API_KEY

# Test LLM connection
python -c "from bamboo.llm import get_llm; llm = get_llm(); print(llm)"
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Or install all dependencies
pip install -r requirements.txt
```

## Next Steps

1. **Populate your knowledge base** with real email threads and task data
2. **Analyze exhausted tasks** and review the generated explanations
3. **Provide feedback** to improve the system's accuracy
4. **Customize prompts** for your specific domain
5. **Extend the graph schema** with domain-specific nodes
