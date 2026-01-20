.PHONY: setup start stop test lint format run clean

setup:
	pip install -e ".[dev]"

start:
	docker-compose up -d

stop:
	docker-compose down

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/ -v -m unit

test-integration:
	pytest tests/ -v -m integration

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

run:
	uvicorn src.main:app --reload --port 8000

run-scrape-reddit:
	python -m src.scrapers.reddit_scraper

run-scrape-forums:
	python -m src.scrapers.forum_scraper

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
