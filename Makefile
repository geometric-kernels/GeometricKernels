.PHONY: help docs install format lint test

SUCCESS='\033[0;32m'

SHELL=/bin/bash
PYVERSION:=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
GK_REQUIREMENTS?=test_requirements-$(PYVERSION).txt

help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

docs:
	(cd docs ; make clean; make doctest; make html)
	@echo "${SUCCESS}============== Docs are available at docs/_build/html/index.html ============== ${SUCCESS}"


install:  ## Install repo for developement (Only for Linux)
	@echo "=== pip install package with dev requirements (using $(GK_REQUIREMENTS)) =============="
	pip install --upgrade pip
	pip install --upgrade --upgrade-strategy eager --no-cache-dir -r $(GK_REQUIREMENTS) | cat
	pip install -e .

format:  ## Formats code with `autoflake`, `black` and `isort`
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place geometric_kernels tests --exclude=__init__.py
	black geometric_kernels tests
	isort geometric_kernels tests

lint:
	flake8 geometric_kernels tests
	black geometric_kernels tests --check --diff
	isort geometric_kernels tests --check-only --diff
	mypy --namespace-packages geometric_kernels


test:  ## Run the tests, start with the failing ones and break on first fail.
	pytest -v -x --ff -rN -Wignore -s --tb=short --durations=0 --cov --cov-report=xml tests
	pytest --nbmake --nbmake-kernel=python3 --durations=0 --nbmake-timeout=1000 --ignore=notebooks/frontends/GPJax.ipynb notebooks/
