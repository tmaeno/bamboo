# Database Plugin Architecture

The Bamboo project now supports a plugin-based architecture for database backends, allowing you to easily swap between different database implementations or add new ones.

## Current Supported Backends

### Graph Database
- **Neo4j** (default) - Property graph database

### Vector Database
- **Qdrant** (default) - Vector similarity search database

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Application Code (CLI, Workflows, Agents)              │
│  Uses GraphDatabaseClient & VectorDatabaseClient         │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│  Client Wrappers (Factory Pattern)                       │
│  - GraphDatabaseClient                                   │
│  - VectorDatabaseClient                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│  Factory & Registry                                      │
│  - get_graph_backend()                                   │
│  - get_vector_backend()                                  │
│  - Backend registration                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼──────┐    ┌──────▼────────┐
│ Neo4j Backend│    │ Qdrant Backend │
│ (Graph DB)   │    │ (Vector DB)    │
└──────────────┘    └────────────────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼──────────────┐
        │ Database Implementations│
        │ - Neo4j Driver         │
        │ - Qdrant SDK           │
        └────────────────────────┘
```

## Configuration

### Environment Variables

Set these in your `.env` file to select backends:

```env
# Database Backend Selection
GRAPH_DATABASE_BACKEND=neo4j
VECTOR_DATABASE_BACKEND=qdrant

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=graph_db
NEO4J_PASSWORD=password
NEO4J_DATABASE=graph_db

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=bamboo_knowledge
```

## Usage

### Default Usage (Auto-configured)

```python
from bamboo.database import GraphDatabaseClient, VectorDatabaseClient

# Creates clients with configured backends
graph_db = GraphDatabaseClient()
vector_db = VectorDatabaseClient()

# Connect and use normally
await graph_db.connect()
await graph_db.create_node(node)
await graph_db.close()
```

### Manual Backend Selection

```python
from bamboo.database.factory import get_graph_backend, get_vector_backend

# Get the configured backend instances
graph_backend = get_graph_backend()
vector_backend = get_vector_backend()

# Use backends directly (advanced use case)
await graph_backend.connect()
```

### List Available Backends

```python
from bamboo.database import list_graph_backends, list_vector_backends

graph_backends = list_graph_backends()
vector_backends = list_vector_backends()

print(f"Available graph backends: {graph_backends}")
print(f"Available vector backends: {vector_backends}")
```

## Adding a New Backend

### 1. Create Backend Implementation

Create a new file implementing the abstract base class:

```python
# bamboo/database/backends/custom_backend.py

from bamboo.database.base import GraphDatabaseBackend
from typing import Any

class CustomBackend(GraphDatabaseBackend):
    """Custom graph database implementation."""

    async def connect(self):
        """Establish connection."""
        # Implementation here
        pass

    async def close(self):
        """Close connection."""
        # Implementation here
        pass

    async def create_node(self, node) -> str:
        """Create a node."""
        # Implementation here
        pass

    # ... implement other required methods ...
```

### 2. Register the Backend

Option A: Register programmatically:

```python
from bamboo.database.factory import register_graph_backend
from bamboo.database.backends.custom_backend import CustomBackend

register_graph_backend("custom", CustomBackend)
```

Option B: Add to `factory.py` for auto-registration:

```python
# In bamboo/database/factory.py

def _register_builtin_backends():
    """Register built-in backend implementations."""
    # ... existing registrations ...
    
    try:
        from bamboo.database.backends.custom_backend import CustomBackend
        register_graph_backend("custom", CustomBackend)
    except ImportError:
        logger.warning("Custom backend dependencies not installed")
```

### 3. Use the New Backend

Set environment variable:

```env
GRAPH_DATABASE_BACKEND=custom
```

Or configure in `.env` and restart the application.

## Backend Implementation Guide

### GraphDatabaseBackend Abstract Methods

All methods must be implemented:

```python
class GraphDatabaseBackend(ABC):
    async def connect(self) -> None:
        """Establish connection to database."""

    async def close(self) -> None:
        """Close connection."""

    async def create_node(self, node: BaseNode) -> str:
        """Create a node and return its ID."""

    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Get existing node or create with canonical name."""

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create relationship between nodes."""

    async def find_causes_by_error(
        self, error_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find causes matching an error."""

    async def find_causes_by_features(
        self, features: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find causes matching features."""

    async def increment_cause_frequency(self, cause_id: str) -> None:
        """Increment cause frequency counter."""

    async def update_resolution_success_rate(
        self, resolution_id: str, success: bool
    ) -> None:
        """Update resolution success metrics."""
```

### VectorDatabaseBackend Abstract Methods

All methods must be implemented:

```python
class VectorDatabaseBackend(ABC):
    async def connect(self) -> None:
        """Establish connection to database."""

    async def close(self) -> None:
        """Close connection."""

    async def upsert_section_vector(
        self,
        vector_id: str,
        embedding: list[float],
        content: str,
        section: str,
        metadata: dict[str, Any],
    ) -> str:
        """Insert or update document."""

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID."""

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve document by ID."""
```

## Examples

### Example: Adding PostgreSQL with pgvector

1. Create backend:

```python
# bamboo/database/backends/postgres_backend.py
import asyncpg
from bamboo.database.base import VectorDatabaseBackend

class PostgresVectorBackend(VectorDatabaseBackend):
    def __init__(self):
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(...)
    
    async def close(self):
        await self.pool.close()
    
    # ... implement other methods ...
```

2. Register in `factory.py`:

```python
try:
    from bamboo.database.backends.postgres_backend import PostgresVectorBackend
    register_vector_backend("postgres", PostgresVectorBackend)
except ImportError:
    logger.warning("PostgreSQL backend dependencies not installed")
```

3. Configure in `.env`:

```env
VECTOR_DATABASE_BACKEND=postgres
POSTGRES_URL=postgresql://user:pass@localhost/bamboo
```

## File Structure

```
bamboo/database/
├── __init__.py                 # Exports clients and factory functions
├── base.py                     # Abstract base classes
├── factory.py                  # Backend factory and registry
├── graph_database_client.py    # Graph DB client wrapper
├── vector_database_client.py   # Vector DB client wrapper
└── backends/
    ├── __init__.py
    ├── neo4j_backend.py        # Neo4j implementation
    └── qdrant_backend.py       # Qdrant implementation
```

## Testing with Different Backends

To test with different backends:

```bash
# Test with Neo4j
GRAPH_DATABASE_BACKEND=neo4j pytest tests/

# Test with different vector backend (if added)
VECTOR_DATABASE_BACKEND=postgres pytest tests/
```

## Troubleshooting

### Backend not found error

```
ValueError: Graph database backend 'custom' not found
```

**Solution**: Ensure the backend is registered before creating the client:

```python
from bamboo.database.factory import register_graph_backend
from bamboo.database.backends.custom import CustomBackend

register_graph_backend("custom", CustomBackend)
```

### Import errors

If you see import errors for a backend:

```
ImportError: No module named 'neo4j'
```

**Solution**: Install required dependencies:

```bash
pip install neo4j
# or
pip install qdrant-client
```

## Migration Guide

### From Direct Neo4j Usage

**Before:**
```python
from neo4j import AsyncGraphDatabase

driver = AsyncGraphDatabase.driver(uri, auth=auth)
```

**After:**
```python
from bamboo.database import GraphDatabaseClient

graph_db = GraphDatabaseClient()
await graph_db.connect()
```

Benefits:
- Abstracted backend selection
- Easy to switch implementations
- Consistent interface across the codebase
- Environment-based configuration

