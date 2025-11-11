.PHONY: help docs install format lint test

SUCCESS = \033[0;32m
RESET   = \033[0m

SHELL=/bin/bash
UV        ?= uv
UV_RUN    ?= uv run
VENV_DIR  ?= .venv
UV_PYTHON ?= 3.11

help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

docs:
	(cd docs ; make clean; make doctest; make html)
	@echo -e "$(SUCCESS)============== Docs are available at docs/_build/html/index.html ==============$(RESET)"

venv: ## Create a virtualenv (UV_PYTHON=3.10 to force)
	@$(UV) venv --seed $(if $(strip $(UV_PYTHON)),--python $(UV_PYTHON),) $(VENV_DIR)
	@echo -e "$(SUCCESS)Virtualenv ready in $(VENV_DIR)$(RESET)"

sync: ## Resolve + install project and dev deps for development
	@$(UV) sync --dev
	@echo -e "$(SUCCESS)Environment synced from pyproject.toml$(RESET)"

install: sync  ## Backward-compat


format: sync ## Formats code with `autoflake`, `black` and `isort`
	@$(UV_RUN) autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place geometric_kernels tests --exclude=__init__.py
	@$(UV_RUN) black geometric_kernels tests
	@$(UV_RUN) isort geometric_kernels tests
	@echo -e "$(SUCCESS)Format done$(RESET)"

lint: sync
	@$(UV_RUN) flake8 geometric_kernels tests
	@$(UV_RUN) black geometric_kernels tests --check --diff
	@$(UV_RUN) isort geometric_kernels tests --check-only --diff
	@$(UV_RUN) mypy --namespace-packages geometric_kernels
	@echo -e "$(SUCCESS)Lint done$(RESET)"

test: sync ## Run the tests, start with the failing ones and break on first fail.
	@$(UV_RUN) pytest -v -x --ff -rN -Wignore -s --tb=short --durations=0 --cov --cov-report=xml --cov-report=html:coverage_html tests
	@$(UV_RUN) pytest --nbmake --nbmake-kernel=python3 --durations=0 --nbmake-timeout=1000 --ignore=notebooks/frontends/GPJax.ipynb notebooks/
	@if [ "$(UV_PYTHON)" = "python3.9" ]; then \
		echo "Skipping GPJax notebook on python3.9"; \
	else \
		$(UV_RUN) pytest --nbmake --nbmake-kernel=python3 --durations=0 --nbmake-timeout=1000 notebooks/frontends/GPJax.ipynb; \
	fi;
	@echo -e "$(SUCCESS)Tests done$(RESET)"
