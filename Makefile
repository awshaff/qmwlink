VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: install test lint figs paper-figs bench clean

install:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest --cov --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy --strict src

figs paper-figs bench:
	@echo "target '$@' is not implemented yet (milestone M6/M7)"; exit 1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
