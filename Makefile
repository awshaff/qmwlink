VENV ?= .venv
PY   := $(VENV)/bin/python
QMW  := $(PY) -m qmwlink.cli

.PHONY: install test lint figs paper-figs exp bench clean

install:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest --cov --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy --strict src

# make exp ID=E3 [FORCE=1]   (FR-EXP-2; cached, see `qmw run`)
exp:
	@test -n "$(ID)" || { echo "usage: make exp ID=E<n> [FORCE=1]"; exit 2; }
	$(QMW) run configs/$(shell echo $(ID) | tr A-Z a-z).yaml $(if $(FORCE),--force)

figs:
	$(QMW) figs

paper-figs:
	$(QMW) paper-figs

bench:
	@echo "target 'bench' is not implemented yet (milestone M7)"; exit 1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
