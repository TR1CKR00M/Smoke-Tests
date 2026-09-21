PY := uv run python
RUFF := uv run ruff
PYTEST := uv run pytest

.PHONY: setup house-prices jarvis-volume reproduce test test-fast lint format

setup:
	uv sync --extra dev

house-prices:
	$(PY) -m house_prices_shap.pipeline --config projects/house_prices_shap/configs/experiment.yaml

jarvis-volume:
	$(PY) -m jarvis_volume.pipeline --config projects/jarvis_volume/configs/experiment.yaml

reproduce: house-prices jarvis-volume

test:
	$(PYTEST) -q

test-fast:
	$(PYTEST) -q -m "not slow and not shap"

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
