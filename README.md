# Bamboo

**B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

An intelligent multi-agent AI system for system automation and operations, leveraging graph and vector databases for knowledge management.

## Quick Start

```bash
# 1. Verify installation
python verify_installation.py

# 2. Setup
cp examples/.env.example .env          # Configure your API keys
docker-compose up -d          # Start databases (Neo4j, Qdrant)
pip install -r requirements.txt

# 3. Try it
python -m bamboo.cli interactive
```

For detailed setup: see [QUICKSTART.md](QUICKSTART.md)

## What is Bamboo?

Bamboo provides **6 specialized AI agents** working together:

## Key Features

✅ **Multi-Agent Architecture**  
✅ **Dual Database** - Graph + Vector search  
✅ **LLM-Powered** - Intelligent extraction, analysis, reasoning, decision-making  
✅ **Knowledge Graph** - Structured operational knowledge  
✅ **MCP Tools** - Domain-specific Model Context Protocol tools  
✅ **Human-in-Loop** - Safety through human oversight

## Main Agents
- **Knowledge Accumulation** - Learns from operational data, builds knowledge database
- **Reasoning Navigation** - Analyzes issues, finds root causes, suggests resolutions

### Sub-agents
- **Graph Extraction** - Extracts structured knowledge from unstructured data
- 
### Agents to come
- **Automation Agent** - Plans and executes operational workflows
- **Anomaly Detection Agent** - Monitors metrics, detects anomalies
- **Proactive Mitigation Agent** - Predicts failures, applies preventive measures
- **System Enhancement Agent** - Identifies optimization opportunities


## Project Structure

```
bamboo/
├── agents/            # AI agents
├── database/          # Database clients
├── mcp_tools/         # MCP tools
├── llm/               # LLM integration
├── models/            # Data models (Pydantic)
├── workflows/         # LangGraph workflows
├── scripts/           # CLI tools
└── utils/             # Utilities
```


## Installation

1. **Prerequisites**: Python 3.10+, Docker, API keys (OpenAI/Anthropic)
2. **Install**: `pip install -r requirements.txt`
3. **Databases**: `docker-compose up -d`
4. **Config**: `cp examples/.env.example .env` and add API keys
5. **Verify**: `python verify_installation.py`

For detailed instructions: see [QUICKSTART.md](QUICKSTART.md)


## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Development setup and code style
- Extension points (agents, tools, node types)
- Testing and deployment guides


## Next Steps

1. Setup: `python verify_installation.py`
2. Learn: [QUICKSTART.md](QUICKSTART.md)
3. Develop: [DEVELOPMENT.md](DEVELOPMENT.md)

## License

MIT
