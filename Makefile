.PHONY: install format format-check lint typecheck test test-security validate-contracts check local-up local-run local-down

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
	uv run mypy services packages local tests

test:
	uv run pytest -q

test-security:
	uv run bandit -c pyproject.toml -r services packages local

validate-contracts:
	uv run openapi-spec-validator contracts/openapi/openapi.v1.yaml

check: format-check lint typecheck test test-security validate-contracts

local-up:
	docker compose -f local/compose.yaml up --build --wait

local-run:
	sh local/scripts/run.sh

local-down:
	sh local/scripts/reset.sh
