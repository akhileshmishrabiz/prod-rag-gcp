.PHONY: install format format-check lint typecheck test test-security validate-contracts check

install:
	uv sync --frozen --all-groups

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy services packages tests

test:
	uv run pytest -q

test-security:
	uv run bandit -c pyproject.toml -r services packages

validate-contracts:
	uv run openapi-spec-validator contracts/openapi/openapi.v1.yaml

check: format-check lint typecheck test test-security validate-contracts
