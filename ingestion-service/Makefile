.PHONY: setup lint test run bench clean help install dev-install

# Default target
help:
	@echo "Available targets:"
	@echo "  setup       - Install dependencies and set up development environment"
	@echo "  dev-install - Install package in development mode"
	@echo "  lint        - Run code linting with ruff and mypy"
	@echo "  test        - Run test suite with pytest"
	@echo "  run         - Start the service with docker compose"
	@echo "  bench       - Run benchmark suite"
	@echo "  clean       - Clean up generated files and containers"

setup: dev-install
	@echo "Setting up development environment..."
	cp .env.example .env
	docker compose pull
	@echo "Setup complete. Edit .env file with your configuration."

dev-install:
	@echo "Installing dependencies..."
	pip install -e ".[dev]"
	pre-commit install

lint:
	@echo "Running linters..."
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	@echo "Formatting code..."
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v -m "unit" --cov=src --cov-report=term-missing

test-integration:
	@echo "Running integration tests..."
	pytest tests/ -v -m "integration" --cov=src --cov-report=term-missing

run:
	@echo "Starting services with docker compose..."
	docker compose up --build

run-dev:
	@echo "Starting services in development mode..."
	docker compose up --build ingestion-service redis neo4j qdrant

stop:
	@echo "Stopping services..."
	docker compose down

bench:
	@echo "Running benchmark suite..."
	python -m src.benchmarks.run --engine all --report json

clean:
	@echo "Cleaning up..."
	docker compose down -v
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

install:
	@echo "Installing package..."
	pip install -e .

# CI targets
ci-setup:
	pip install -e ".[dev]"

ci-lint: lint

ci-test: test

ci-build:
	docker compose build

# Database management
db-reset:
	@echo "Resetting databases..."
	docker compose down -v neo4j qdrant redis
	docker compose up -d neo4j qdrant redis

# Monitoring
logs:
	docker compose logs -f ingestion-service

logs-all:
	docker compose logs -f