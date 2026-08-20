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

guide:            ## Render docs/how-it-works.md to PDF (needs pandoc + wkhtmltopdf)
	python scripts/make_guide.py

demo:             ## End-to-end run on synthetic data; writes to reports/
	python -m earnings_engine.cli demo --out reports/demo

holdout:          ## Rolling annual holdouts + null control + dashboard
	python -m earnings_engine.cli holdout --out reports/holdout
	python -m earnings_engine.cli holdout --drift 0 --out reports/holdout_null

reproduce:        ## Regenerate every published number and fingerprint it
	python scripts/reproduce.py

note:             ## Rebuild the two-page research note (docs/research-note.pdf)
	python scripts/make_research_note.py

verify:           ## Re-run and check the fingerprints against docs/manifest.json
	python scripts/reproduce.py --check

clean:            ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
