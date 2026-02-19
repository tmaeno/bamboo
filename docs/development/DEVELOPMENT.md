# Development Guide

This guide covers development practices, code organization, and extension points for the Bamboo project.

## Project Structure

```
bamboo/
├── bamboo/                # Main package
│   ├── agents/            # AI agent implementations
│   ├── database/          # Database clients
│   ├── extractors/        # Knowledge extractors
│   ├── llm/               # LLM integration
│   ├── models/            # Data models
│   ├── mcp_tools/         # MCP tools
│   ├── workflows/         # LangGraph workflows
│   ├── scripts/           # CLI scripts
│   ├── utils/             # Utilities
│   ├── cli.py             # Interactive CLI
│   └── config.py          # Configuration
├── tests/                 # Test suite
├── examples/              # Example data
└── docker-compose.yml     # Database services
```

## Development Setup

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Start Development Databases

```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Verify services are running
docker ps
```

### 3. Configure Environment

```bash
# Copy example config
cp examples/.env.example .env

# Edit .env with your settings
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 4. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check code style
black --check bamboo/
ruff check bamboo/

# Type checking
mypy bamboo/
```

## Code Style and Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Use Black for formatting (88 char line length)
black bamboo/

# Use Ruff for linting
ruff check bamboo/

# Use type hints where possible
from typing import Optional, Any

async def process_data(
    data: dict[str, Any],
    timeout: Optional[int] = None,
) -> list[str]:
    """Process data and return results.
    
    Args:
        data: Input data dictionary
        timeout: Optional timeout in seconds
        
    Returns:
        List of processed strings
    """
    pass
```

### Async/Await

Use async/await for I/O operations:

```python
# Good: Async for database operations
async def query_database(query: str) -> list[dict]:
    async with self.driver.session() as session:
        result = await session.run(query)
        return await result.values()

# Good: Async for LLM calls
async def call_llm(prompt: str) -> str:
    response = await self.llm.ainvoke(messages)
    return response.content
```

### Error Handling

```python
import logging

logger = logging.getLogger(__name__)

async def process_task(task_id: str):
    try:
        result = await perform_operation(task_id)
        logger.info(f"Task {task_id} processed successfully")
        return result
    except SpecificError as e:
        logger.error(f"Failed to process task {task_id}: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing task {task_id}")
        raise
```

## Extension Points

### 1. Adding New Node Types

**File: `bamboo/models/graph.py`**

```python
from enum import Enum
from pydantic import BaseModel, Field

# Add to NodeType enum
class NodeType(str, Enum):
    # ...existing types...
    CUSTOM_TYPE = "CustomType"

# Create node class
class CustomTypeNode(BaseNode):
    """Represents a custom type."""
    
    node_type: NodeType = NodeType.CUSTOM_TYPE
    custom_field: str
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
```

**Update extractor in `bamboo/extractors/graph_extractor.py`:**

```python
def _create_node(self, node_data: dict[str, Any]):
    """Create appropriate node type from data."""
    node_type = NodeType(node_data["node_type"])
    
    # ...existing code...
    
    elif node_type == NodeType.CUSTOM_TYPE:
        return CustomTypeNode(
            **base_fields,
            custom_field=node_data.get("custom_field"),
            custom_metadata=node_data.get("custom_metadata", {}),
        )
```

### 2. Adding New Relationship Types

**File: `bamboo/models/graph.py`**

```python
class RelationType(str, Enum):
    # ...existing types...
    CUSTOM_RELATION = "custom_relation"
```

**Update prompts in `bamboo/llm/prompts.py`:**

```python
EXTRACTION_PROMPT = """
...
Relationship Types:
- indicate
- associated_with
- contribute_to
- originated_from
- solved_by
- signals
- leads_to
- has_component
- depends_on
- suggests
- improves
- triggers
- affects
- performed_by (User performs Action)
- reported_by (User reports Issue)
- assigned_to (Task assigned to User)
- approved_by (User approves Action)
- custom_relation  # Add your new relation
...
"""
```

### 3. Adding Custom Database Queries

**File: `bamboo/database/neo4j_client.py`**

```python
class Neo4jClient:
    # ...existing methods...
    
    async def custom_query(
        self,
        param1: str,
        param2: int,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Custom query description.
        
        Args:
            param1: Description
            param2: Description
            limit: Maximum results
            
        Returns:
            List of query results
        """
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            query = """
            MATCH (n:NodeType)
            WHERE n.property = $param1 AND n.count > $param2
            RETURN n
            LIMIT $limit
            """
            result = await session.run(
                query,
                param1=param1,
                param2=param2,
                limit=limit
            )
            records = await result.values()
            return [{"data": record[0]} for record in records]
```

### 4. Custom LLM Prompts

**File: `bamboo/llm/prompts.py`**

```python
CUSTOM_ANALYSIS_PROMPT = """You are an expert in {domain}.

Context:
{context}

Task:
{task_description}

Analyze and provide:
1. Key findings
2. Recommendations
3. Risk assessment

Response format:
{{
  "findings": [...],
  "recommendations": [...],
  "risks": [...]
}}
"""
```

**Use in agent:**

```python
from bamboo.llm import get_llm, CUSTOM_ANALYSIS_PROMPT

async def custom_analysis(context: str, task: str):
    llm = get_llm()
    
    prompt = CUSTOM_ANALYSIS_PROMPT.format(
        domain="your_domain",
        context=context,
        task_description=task
    )
    
    messages = [
        SystemMessage(content="You are an expert analyst."),
        HumanMessage(content=prompt),
    ]
    
    response = await llm.ainvoke(messages)
    return json.loads(response.content)
```

### 5. Adding Sub-Agents

Create a new sub-agent for specific tasks:

**File: `bamboo/agents/custom_subagent.py`**

```python
import logging
from typing import Any

from bamboo.llm import get_llm

logger = logging.getLogger(__name__)


class CustomSubAgent:
    """Sub-agent for specific task."""
    
    def __init__(self):
        """Initialize sub-agent."""
        self.llm = get_llm()
    
    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process input and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed results
        """
        logger.info("CustomSubAgent processing data")
        
        # Your processing logic here
        result = await self._analyze(input_data)
        
        return result
    
    async def _analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Private method for analysis."""
        # Implementation
        pass
```

**Integrate in main agent:**

```python
from bamboo.agents.custom_subagent import CustomSubAgent

class MainAgent:
    def __init__(self):
        self.custom_subagent = CustomSubAgent()
    
    async def process(self, data):
        # Use sub-agent
        result = await self.custom_subagent.process(data)
        # Continue with main agent logic
        pass
```

### 6. Extending LangGraph Workflows

**File: `bamboo/workflows/custom_workflow.py`**

```python
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END


class CustomState(TypedDict):
    """State for custom workflow."""
    input_data: dict[str, Any]
    processed_data: dict[str, Any]
    result: dict[str, Any]
    status: str
    error: Optional[str]


async def step_1_node(state: CustomState) -> CustomState:
    """First step in workflow."""
    try:
        # Process data
        processed = process_step_1(state["input_data"])
        
        return {
            **state,
            "processed_data": processed,
            "status": "step_1_complete",
        }
    except Exception as e:
        return {
            **state,
            "status": "error",
            "error": str(e),
        }


async def step_2_node(state: CustomState) -> CustomState:
    """Second step in workflow."""
    # Implementation
    pass


def should_continue(state: CustomState) -> str:
    """Determine next step based on state."""
    if state["status"] == "error":
        return "error"
    elif state["status"] == "step_1_complete":
        return "step_2"
    else:
        return "end"


def create_custom_workflow() -> StateGraph:
    """Create custom workflow."""
    workflow = StateGraph(CustomState)
    
    # Add nodes
    workflow.add_node("step_1", step_1_node)
    workflow.add_node("step_2", step_2_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Add edges
    workflow.set_entry_point("step_1")
    
    # Conditional edges
    workflow.add_conditional_edges(
        "step_1",
        should_continue,
        {
            "step_2": "step_2",
            "error": "error_handler",
            "end": END,
        },
    )
    
    workflow.add_edge("step_2", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()
```

## Testing

### Unit Tests

```python
# tests/test_custom_agent.py
import pytest
from bamboo.agents.custom_agent import CustomAgent


@pytest.mark.asyncio
async def test_custom_agent_process():
    """Test custom agent processing."""
    agent = CustomAgent()
    
    input_data = {"key": "value"}
    result = await agent.process(input_data)
    
    assert result is not None
    assert "expected_key" in result


@pytest.fixture
def mock_llm(mocker):
    """Mock LLM for testing."""
    mock = mocker.patch("bamboo.llm.get_llm")
    mock.return_value.ainvoke.return_value.content = '{"result": "test"}'
    return mock


@pytest.mark.asyncio
async def test_agent_with_mock_llm(mock_llm):
    """Test agent with mocked LLM."""
    agent = CustomAgent()
    result = await agent.process({})
    
    assert result["result"] == "test"
    mock_llm.return_value.ainvoke.assert_called_once()
```

### Integration Tests

```python
# tests/integration/test_knowledge_extraction.py
import pytest
from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
from bamboo.database.neo4j_client import Neo4jClient
from bamboo.database.qdrant_client import QdrantClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_knowledge_extraction():
    """Test full knowledge extraction pipeline."""
    neo4j = Neo4jClient()
    qdrant = QdrantClient()

    await neo4j.connect()
    await qdrant.connect()

    try:
        agent = KnowledgeAccumulator(neo4j, qdrant)

        result = await agent.process_knowledge(
            email_text="Sample email",
            task_data={"task_id": "TEST-1"},
        )

        assert result.graph.nodes
        assert result.summary

    finally:
        await neo4j.close()
        await qdrant.close()
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_extractors.py -v

# Run with coverage
pytest tests/ --cov=bamboo --cov-report=html

# Run integration tests only
pytest tests/ -v -m integration

# Run excluding integration tests
pytest tests/ -v -m "not integration"
```

## Debugging

### Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or in .env
LOG_LEVEL=DEBUG
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()
```

### Database Inspection

```bash
# Neo4j Browser
open http://localhost:7474

# Cypher query examples
MATCH (n) RETURN n LIMIT 25;
MATCH (c:Cause) RETURN c.name, c.frequency ORDER BY c.frequency DESC;
MATCH (e:Error)-[r:indicate]->(c:Cause) RETURN e, r, c;

# Qdrant inspection
curl http://localhost:6333/collections
curl http://localhost:6333/collections/bamboo_knowledge
```

## Performance Optimization

### Database Indexing

Neo4j indexes are created automatically. To add custom indexes:

```python
async def create_custom_indexes(self):
    """Create custom indexes."""
    async with self.driver.session() as session:
        await session.run(
            "CREATE INDEX custom_index IF NOT EXISTS "
            "FOR (n:NodeType) ON (n.property)"
        )
```

### Batch Operations

```python
async def batch_create_nodes(self, nodes: list[BaseNode]):
    """Create multiple nodes in batch."""
    async with self.driver.session() as session:
        async with session.begin_transaction() as tx:
            for node in nodes:
                await tx.run(
                    "CREATE (n:NodeType $properties)",
                    properties=node.model_dump()
                )
            await tx.commit()
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_config(key: str) -> Any:
    """Cache configuration lookups."""
    return load_config(key)
```

## Deployment

### Production Considerations

1. **Environment Variables**: Never commit `.env` files
2. **Database Persistence**: Use volumes in production
3. **API Rate Limits**: Implement retry logic and rate limiting
4. **Monitoring**: Add logging and metrics
5. **Security**: Secure database connections, use secrets management

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bamboo/ bamboo/
COPY setup.py .

RUN pip install -e .

CMD ["python", "-m", "bamboo.cli", "interactive"]
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  bamboo:
    build: .
    env_file: .env
    depends_on:
      - neo4j
      - qdrant
    volumes:
      - ./data:/app/data
  
  neo4j:
    image: neo4j:5.16.0
    environment:
      - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
    restart: unless-stopped
  
  qdrant:
    image: qdrant/qdrant:v1.7.4
    volumes:
      - qdrant_data:/vector_db/storage
    restart: unless-stopped

volumes:
  neo4j_data:
  qdrant_data:
```

## Contributing

### Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run tests: `pytest tests/ -v`
4. Format code: `black bamboo/`
5. Check linting: `ruff check bamboo/`
6. Commit changes: `git commit -m "Add feature"`
7. Push branch: `git push origin feature/your-feature`
8. Create pull request

### Commit Messages

Follow conventional commits:

```
feat: Add new node type for performance metrics
fix: Correct canonicalization logic
docs: Update architecture documentation
test: Add tests for graph extractor
refactor: Simplify database query methods
```

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Graph Schema Reference

The Bamboo knowledge graph consists of **16 node types** and **18 relationship types**:

### Node Types (16)

```
- Symptom: Symptom messages and failures
- Cause: Root causes of issues
- Resolution: Solutions and fixes
- Environment: External factors
- Task_Feature: Task features
- Task_Context: Task context and reproduction steps
- Component: System origin of causes
- Metric: System metrics and KPIs
- Anomaly: Detected anomalies
- Issue: System issues
- System: Systems and services
- Pattern: Operational patterns
- Optimization: Optimization opportunities
- Event: System events
- Action: Actions (automated/manual)
- Dependency: System dependencies
- User: Users (operators, engineers, admins)
```

### Relationship Types (18)

```
Core Relationships:
- indicate: Symptom indicates Cause
- associated_with: Environment associated with Cause
- contribute_to: Task_Feature/Task_Context contributes to Cause
- originated_from: Cause originated from Component
- solved_by: Cause solved by Resolution

System Relationships:
- signals: Metric signals Anomaly
- leads_to: Anomaly leads to Issue
- has_component: System has Component
- depends_on: Component depends on Dependency
- suggests: Pattern suggests Optimization
- improves: Optimization improves Performance
- triggers: Event triggers Action
- affects: Action affects System

User Relationships:
- performed_by: Action performed by User
- reported_by: Issue reported by User
- assigned_to: Task assigned to User
- approved_by: Action approved by User
```

### Graph Example

```
User (operator) ──[performed_by]──> Action ──[affects]──> System
                                        ↑
                                   ──[approved_by]──> User (admin)

Metric ──[signals]──> Anomaly ──[leads_to]──> Issue ──[reported_by]──> User
   ↓
Symptom ──[indicate]──> Cause ──[solved_by]──> Resolution
         ↓
     Component ──[originated_from]──> System
```
