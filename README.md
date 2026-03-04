# Bamboo

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

**B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

An intelligent multi-agent AI framework for system automation and operations, leveraging graph and vector databases for knowledge management.

## Key Features

✅ **Multi-Agent Architecture**  
✅ **Dual Database** - Graph database (Neo4j) + Vector database (Qdrant)  
✅ **LLM-Powered** - Intelligent extraction, analysis, reasoning, decision-making  
✅ **Knowledge Graph** - Modeling of operational dynamics in scientific computing workforce  
✅ **Hierarchical Workload Insights** - Full visibility across tasks, jobs, and their dependencies  
✅ **PanDA Integration** - Fetch task data live from a PanDA server by task ID  
✅ **MCP Tools** - Domain-specific Model Context Protocol tools  
✅ **Human-in-Loop** - Safety through human oversight

## Main Agents
- **Knowledge Accumulation** - Learns from operational data, builds knowledge database
- **Reasoning Navigation** - Analyzes issues, finds root causes, suggests resolutions

### Sub-Agents

#### In Knowledge Accumulation Agent
- **Knowledge Extraction** - Extracts knowledge graph from structured and unstructured data
- **Node Canonicalization** - Normalises diverse node names into a stable canonical form
- **Graph Summarization** - Summarises graph data for quick insights
- **Feature Classification** - Classifies node features for better reasoning

#### In Reasoning Navigation Agent
- **Knowledge Extraction** - Extracts features from task data for querying
- **Output Synthesis** - Synthesises outputs from various agents into coherent responses

### Agents to come
- **Automation Agent** - Plans and executes operational workflows
- **Anomaly Detection Agent** - Monitors metrics, detects anomalies
- **Proactive Mitigation Agent** - Predicts failures, applies preventive measures
- **System Enhancement Agent** - Identifies optimization opportunities


## Quick Start

```bash
# 1. Install
pip install .

# 2. Verify the installation
bamboo-verify

# 3. Copy .env.example (path shown by bamboo-verify) and add your API keys
cp <path-from-bamboo-verify> .env

# 4. Start databases
docker-compose up -d

# 5. Try it
bamboo interactive
```

### Fetch a PanDA task directly

```bash
# Inspect raw task data from a live PanDA server
bamboo fetch-task 12345

# Populate the knowledge base using a live task (no local JSON file needed)
bamboo-populate --task-id 12345

# Analyze a live task
bamboo-analyze --task-id 12345
```

For detailed setup: see [docs/QUICKSTART.md](docs/QUICKSTART.md)


## Documentation

| File | Purpose |
|------|---------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Full setup and installation guide |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Developer guide and extension points |
| [docs/EXTRACTION_PLUGIN_SYSTEM.md](docs/EXTRACTION_PLUGIN_SYSTEM.md) | Extraction strategy plugin API |
| [docs/PANDA_INTEGRATION.md](docs/PANDA_INTEGRATION.md) | PanDA live-fetch integration |
| [docs/database-plugins/](docs/database-plugins/) | Graph / vector database plugin system |


## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Development setup and code style
- Extension points (agents, tools, node types)
- Testing and deployment guides

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
