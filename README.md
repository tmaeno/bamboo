# Bamboo

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

**B**olstered **A**ssistance for **M**anaging and **B**uilding **O**perations and **O**versight

An intelligent multi-agent AI framework for system automation and operations, leveraging graph and vector databases for knowledge management.

## Key Features

✅ **Multi-Agent Architecture**  
✅ **Dual Database** - Graph database + Vector database  
✅ **LLM-Powered** - Intelligent extraction, analysis, reasoning, decision-making  
✅ **Knowledge Graph** - Modeling of operational dynamics in scientific computing workforce  
✅ **Hierarchical Workload Insights** - Full visibility across tasks, jobs, and their dependencies  
✅ **PanDA Integration** - Fetch task data live from a PanDA server by task ID  
✅ **MCP Tools** - Domain-specific Model Context Protocol tools for both reasoning and source exploration  
✅ **Human-in-Loop** - Safety through human oversight

## Main Agents
- **Knowledge Accumulation** - Learns from operational data, builds knowledge database
- **Reasoning Navigation** - Analyzes issues, finds root causes, suggests resolutions

## LLM-Driven Pipelines and Sub-Agents

### For Knowledge Accumulation
- **Knowledge Extractor** - Extracts knowledge graph from structured and unstructured data
- **Node Canonicalizer** - Normalises diverse node names into a stable canonical form
- **Graph Summarizer** - Summarises graph data for quick insights
- **Feature Classifier** - Classifies node features for better reasoning
- **Knowledge Reviewer** - Quality gate; evaluates the extracted graph against source data before DB writes; retries extraction with feedback up to 2 times if rejected (opt-in via `ENABLE_KNOWLEDGE_REVIEW=true`)
- **Extra Source Explorer** - Fires once on the first reviewer rejection; selects and fetches additional data sources via the MCP tool layer, then feeds the results into the next extraction attempt

### For Reasoning Navigation
- **Knowledge Extractor** - Extracts features from task data for querying
- **Output Synthesiser** - Synthesises outputs from various agents into coherent responses

## Agents to Come
- **Automation Agent** - Plans and executes operational workflows
- **Anomaly Detection Agent** - Monitors metrics, detects anomalies
- **Proactive Mitigation Agent** - Predicts failures, applies preventive measures
- **System Enhancement Agent** - Identifies optimization opportunities


## Quick Start

```bash
# 1. Install
pip install .

# 2. Verify the installation
bamboo verify

# 3. Start databases
docker-compose up -d

# 4. Try it
bamboo interactive
```

For detailed setup: see [Quick Start](docs/QUICKSTART.md).


## Documentation Map

See [Documentation Map](docs/README.md) for a full index of all documentation and recommended reading paths.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
