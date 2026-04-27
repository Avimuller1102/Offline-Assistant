.PHONY: help install test lint run-api

help:
	@echo "Available commands:"
	@echo "  make install    - Install project and all dependencies (dev, api, cli)"
	@echo "  make test       - Run tests using pytest"
	@echo "  make lint       - Run ruff linter and formatter"
	@echo "  make run-api    - Start the FastAPI server locally"
	@echo "  make cli        - Run the CLI (e.g., make cli ARGS='--help')"

install:
	pip install -e ".[dev,api,cli]"

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format .

run-api:
	uvicorn src.echoshield.api:app --reload

cli:
	echoshield $(ARGS)
