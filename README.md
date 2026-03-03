# Bamboo

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

**B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

An intelligent multi-agent AI framework for system automation and operations, leveraging graph and vector databases for knowledge management.

## Key Features

✅ **Multi-Agent Architecture**  
✅ **Dual Database** - Graph + Vector search  
✅ **LLM-Powered** - Intelligent extraction, analysis, reasoning, decision-making  
✅ **Knowledge Graph** - Modeling of operational dynamics in scientific computing workforce  
✅ **Task & Job awareness** - Task-level configuration (`Task_Feature`) and aggregated job-execution patterns (`Job_Feature`) are both first-class graph entities; logs are distinguished as task-level (orchestration) or job-level (execution)  
✅ **MCP Tools** - Domain-specific Model Context Protocol tools  
✅ **Human-in-Loop** - Safety through human oversight

## Main Agents
- **Knowledge Accumulation** - Learns from operational data, builds knowledge database
- **Reasoning Navigation** - Analyzes issues, finds root causes, suggests resolutions

### Sub-agents
- **Knowledge Extraction** - Extracts knowledge graph from structured and unstructured data
- **Graph Summarization** - Summarizes graph data for quick insights
- **Node Canonicalization** - Transforms diverse node data into a canonical format
- **Feature Classification** - Classifies node features for better reasoning
- **Output Synthesis** - Synthesizes outputs from various agents into coherent responses

### Agents to come
- **Automation Agent** - Plans and executes operational workflows
- **Anomaly Detection Agent** - Monitors metrics, detects anomalies
- **Proactive Mitigation Agent** - Predicts failures, applies preventive measures
- **System Enhancement Agent** - Identifies optimization opportunities


## Quick Start

```bash
# 1. Install
pip install .

# 2. Verify — prints the exact cp command for your .env setup
bamboo-verify

# 3. Copy .env.example (path shown by bamboo-verify) and add your API keys
cp <path-from-bamboo-verify> .env

# 4. Start databases
docker-compose up -d

# 5. Try it
bamboo interactive
```

For detailed setup: see [QUICKSTART.md](docs/getting-started/QUICKSTART.md)


## Development

See [DEVELOPMENT.md](docs/development/DEVELOPMENT.md) for:
- Development setup and code style
- Extension points (agents, tools, node types)
- Testing and deployment guides

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
