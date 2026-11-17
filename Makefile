PY := uv run python
RUFF := uv run ruff
PYTEST := uv run pytest

.PHONY: setup test test-fast lint format

setup:
	uv sync --extra dev

test:
	$(PYTEST) -q

test-fast:
	$(PYTEST) -q -m "not slow and not shap"

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
