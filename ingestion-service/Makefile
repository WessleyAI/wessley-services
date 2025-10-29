# Wessley.ai Ingestion Service - Development Makefile
.PHONY: help setup start stop restart logs health test lint format typecheck clean reset

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Wessley.ai Ingestion Service - Development Commands$(NC)"
	@echo "=================================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make setup     # Complete system setup"
	@echo "  make start     # Start all services"
	@echo "  make test      # Run all tests"
	@echo "  make health    # Check service health"

setup: ## Install dependencies and setup system
	@echo "$(BLUE)Setting up system...$(NC)"
	@chmod +x scripts/*.sh
	@./scripts/setup-system.sh

start: ## Start local infrastructure and application
	@echo "$(BLUE)Starting services...$(NC)"
	@./scripts/start-local.sh

start-app: ## Start only the application (assumes infrastructure is running)
	@echo "$(BLUE)Starting application...$(NC)"
	@./scripts/run-dev.sh

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