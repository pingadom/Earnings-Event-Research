.PHONY: help install install-dev lint fmt typecheck test test-cov demo clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:          ## Install the package (core dependencies only)
	pip install -e .

install-dev:      ## Install with dev + data extras
	pip install -e ".[dev,data]"

lint:             ## Run ruff
	ruff check src tests

fmt:              ## Auto-fix and format
	ruff check --fix src tests && ruff format src tests

typecheck:        ## Run mypy
	mypy src

test:             ## Run the test suite (offline only)
	pytest -m "not network"

test-cov:         ## Run tests with coverage
	pytest -m "not network" --cov=earnings_engine --cov-report=term-missing

demo:             ## End-to-end run on synthetic data; writes to reports/
	python -m earnings_engine.cli demo --out reports/demo

clean:            ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
