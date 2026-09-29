# OfflineForge — developer tasks
PY := python
SRC := offlineforge tests

.PHONY: install test lint fmt benchmark demo clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check $(SRC)
	$(PY) -m ruff format --check $(SRC)

fmt:
	$(PY) -m ruff format $(SRC)
	$(PY) -m ruff check --fix $(SRC)

benchmark:
	$(PY) -m offlineforge.cli --out benchmark.json

demo:
	$(PY) -m offlineforge.examples.run_demo

clean:
	$(PY) -m ruff clean 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f benchmark.json .ruff_cache -rf
