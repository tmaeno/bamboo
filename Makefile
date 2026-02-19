.PHONY: help install dev-install test lint format clean docker-up docker-down

help:
	@echo "Bamboo - Bolstered Assistance for Managing and Building Operations and Oversight"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean up generated files"
	@echo "  make docker-up    - Start Docker services (Neo4j, Qdrant)"
	@echo "  make docker-down  - Stop Docker services"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check bamboo/
	mypy bamboo/

format:
	black bamboo/ tests/
	#ruff check --fix bamboo/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Neo4j: http://localhost:7474 (neo4j/password)"
	@echo "Qdrant: http://localhost:6333"

docker-down:
	docker-compose down

