# Database Backend Plugin System - Quick Reference

## 🎯 Overview

The Bamboo database layer now supports a **plugin-based architecture** that allows you to:
- ✅ Switch between database implementations without code changes
- ✅ Add new database backends easily
- ✅ Configure backends via environment variables
- ✅ Test with different databases

## 🚀 Quick Start

### Using Default Backends

```python
from bamboo.database import GraphDatabaseClient, VectorDatabaseClient

# Automatically uses configured backends (Neo4j + Qdrant)
graph_db = GraphDatabaseClient()
vector_db = VectorDatabaseClient()

await graph_db.connect()
# Use normally...
await graph_db.close()
```

### Switching Backends

Edit `.env`:
```env
GRAPH_DATABASE_BACKEND=neo4j
VECTOR_DATABASE_BACKEND=qdrant
```

## 📋 Available Backends

### Graph Databases
- `neo4j` - Neo4j property graph database (default)
- `in_memory` - In-memory backend (testing/development)

### Vector Databases
- `qdrant` - Qdrant vector search (default)

## 🔧 File Structure

```
bamboo/database/
├── base.py                      # Abstract base classes
├── factory.py                   # Backend registry & factory
├── graph_database_client.py     # Wrapper (delegates to backend)
├── vector_database_client.py    # Wrapper (delegates to backend)
└── backends/
    ├── neo4j_backend.py         # Neo4j implementation
    ├── qdrant_backend.py        # Qdrant implementation
    └── examples/
        └── in_memory_backend.py # Example: in-memory backend
```

## 🛠️ Creating a Custom Backend

### Step 1: Implement Backend Class

```python
# bamboo/database/backends/my_backend.py

from bamboo.database.base import GraphDatabaseBackend
from bamboo.models.graph import BaseNode, GraphRelationship
from typing import Any

class MyGraphBackend(GraphDatabaseBackend):
    async def connect(self):
        # Initialize connection
        pass
    
    async def close(self):
        # Close connection
        pass
    
    async def create_node(self, node: BaseNode) -> str:
        # Implement node creation
        pass
    
    # ... implement all other required methods ...
```

### Step 2: Register Backend

**Option A - Manual Registration:**
```python
from bamboo.database.factory import register_graph_backend
from bamboo.database.backends.my_backend import MyGraphBackend

register_graph_backend("my_backend", MyGraphBackend)
```

**Option B - Auto-registration in `factory.py`:**
```python
def _register_builtin_backends():
    # ... existing code ...
    try:
        from bamboo.database.backends.my_backend import MyGraphBackend
        register_graph_backend("my_backend", MyGraphBackend)
    except ImportError:
        logger.warning("My backend dependencies not installed")
```

### Step 3: Configure

```env
GRAPH_DATABASE_BACKEND=my_backend
```

## 🧪 Using In-Memory Backend (Testing)

```env
GRAPH_DATABASE_BACKEND=in_memory
```

Benefits:
- No external database needed
- Fast testing
- Automatic cleanup
- Perfect for unit tests

Example test:
```python
import pytest
from bamboo.database import GraphDatabaseClient

@pytest.mark.asyncio
async def test_with_in_memory_backend():
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    
    # Data persists in memory during test
    node_id = await graph_db.create_node(my_node)
    
    await graph_db.close()
    # Data is cleared after close
```

## 📚 Required Interface

### GraphDatabaseBackend

```python
class GraphDatabaseBackend:
    async def connect() -> None
    async def close() -> None
    async def create_node(node: BaseNode) -> str
    async def get_or_create_canonical_node(node: BaseNode, name: str) -> str
    async def create_relationship(rel: GraphRelationship) -> bool
    async def find_causes_by_error(error: str, limit: int) -> list
    async def find_causes_by_features(features: list, limit: int) -> list
    async def increment_cause_frequency(cause_id: str) -> None
    async def update_resolution_success_rate(res_id: str, success: bool) -> None
```

### VectorDatabaseBackend

```python
class VectorDatabaseBackend:
    async def connect() -> None
    async def close() -> None
    async def upsert_section_vector(id, embedding, content, section, metadata) -> str
    async def search_similar(embedding, limit, threshold, filters) -> list
    async def delete_document(doc_id: str) -> bool
    async def get_document(doc_id: str) -> dict | None
```

## 🔍 Debugging

### List Available Backends

```python
from bamboo.database import list_graph_backends, list_vector_backends

print(list_graph_backends())   # ['neo4j', 'in_memory']
print(list_vector_backends())  # ['qdrant']
```

### Check Current Configuration

```python
from bamboo.config import get_settings

settings = get_settings()
print(f"Graph backend: {settings.graph_database_backend}")
print(f"Vector backend: {settings.vector_database_backend}")
```

## ⚙️ Environment Variables

```env
# Backend Selection
GRAPH_DATABASE_BACKEND=neo4j
VECTOR_DATABASE_BACKEND=qdrant

# Neo4j Config
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=graph_db
NEO4J_PASSWORD=password
NEO4J_DATABASE=graph_db

# Qdrant Config
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=bamboo_knowledge

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

## 🎓 Examples

### Example 1: Testing with In-Memory Backend

```python
# Test file
import os
os.environ['GRAPH_DATABASE_BACKEND'] = 'in_memory'

from bamboo.database import GraphDatabaseClient
from bamboo.models.graph import ErrorNode

async def test_create_and_query():
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    
    # Create test nodes
    error_node = ErrorNode(
        name="Connection Timeout",
        description="Database connection timeout"
    )
    node_id = await graph_db.create_node(error_node)
    
    assert node_id is not None
    
    await graph_db.close()
```

### Example 2: Multiple Backends for Different Environments

**.env.development:**
```env
GRAPH_DATABASE_BACKEND=in_memory
VECTOR_DATABASE_BACKEND=in_memory  # future
```

**.env.production:**
```env
GRAPH_DATABASE_BACKEND=neo4j
VECTOR_DATABASE_BACKEND=qdrant
```

```bash
# Development
source .env.development && python app.py

# Production
source .env.production && python app.py
```

## 📖 Full Documentation

See `DATABASE_PLUGINS.md` for comprehensive documentation including:
- Detailed architecture explanation
- Complete backend implementation guide
- Migration guide from direct backend usage
- Troubleshooting guide

## ❓ Common Issues

**Q: Backend not found error?**
A: Ensure it's registered before creating clients:
```python
from bamboo.database.factory import register_graph_backend
register_graph_backend("my_backend", MyBackend)
```

**Q: ImportError for database library?**
A: Install the required dependencies:
```bash
pip install neo4j qdrant-client
```

**Q: How to test without Neo4j?**
A: Use the in-memory backend:
```env
GRAPH_DATABASE_BACKEND=in_memory
```

**Q: Can I use multiple backend types?**
A: Yes! You can use Neo4j for graphs and Qdrant for vectors (or any combination).

## 🚦 Next Steps

1. Read `DATABASE_PLUGINS.md` for full details
2. Try the in-memory backend for testing
3. Create your own backend following the guide
4. Share your backends with the community!

