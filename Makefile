.PHONY: help install test lint fmt bench demo serve docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package plus dev extras
	python -m pip install -e ".[dev]"

test:  ## Run the test suite
	pytest -q

lint:  ## Lint and type-check
	ruff check punar tests
	mypy punar

fmt:  ## Auto-fix formatting and lint
	ruff check --fix punar tests
	ruff format punar tests

bench:  ## Canonical benchmark (reproduces the documented numbers)
	python scripts/benchmark.py --n-cases 250 --seed 42 --seeds 20

demo:  ## Regenerate the committed demo artifacts in outputs/
	python -m punar.main report --n-cases 250 --seed 42 --seeds 20 --out outputs/demo_run

serve:  ## Run the API locally
	uvicorn punar.api.server:app --reload

docker:  ## Build the container image
	docker build -t punar:latest .

clean:  ## Remove build and test artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
