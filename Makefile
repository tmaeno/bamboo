.PHONY: help install dev-install test lint format clean clean-all docker-up docker-down docs-install docs-serve docs-build

help:
	@echo "Bamboo - Bolstered Assistance for Managing and Building Operations and Oversight"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean up generated files and caches"
	@echo "  make clean-all    - Clean, plus remove website/node_modules"
	@echo "  make docker-up    - Start Docker services (Neo4j, Qdrant)"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make docs-install - Install documentation dependencies (npm)"
	@echo "  make docs-serve   - Serve the docs locally (Astro dev server)"
	@echo "  make docs-build   - Build the docs site (validates internal links)"

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
	find . -type f -name ".DS_Store" -delete
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	rm -rf website/dist/ website/.astro/ website/public/api/

clean-all: clean
	rm -rf website/node_modules/

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Neo4j: http://localhost:7474 (neo4j/password)"
	@echo "Qdrant: http://localhost:6333"

docker-down:
	docker-compose down

docs-install:
	cd website && npm install

docs-serve:
	cd website && npm run dev

docs-build:
	cd website && npm run build

