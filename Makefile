.PHONY: help install docs format check test check-and-test

SUCCESS='\033[0;32m'


help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

docs:
	(cd docs ; make doctest; make html)
	@echo "\n${SUCCESS}============== Docs are available at docs/_build/html/index.html ============== ${SUCCESS}"


install:  ## Install repo for developement (Only for Linux)
	@echo "\n=== pip install package with dev requirements (Only for Linux!) =============="
	sudo apt-get install gfortran
	# We need to pin `pip`. See https://github.com/pypa/pip/issues/10373.
	pip install --upgrade pip==20.2.2 setuptools numpy Cython
	pip install --upgrade --upgrade-strategy eager --no-cache-dir -U -r requirements.txt -r dev_requirements.txt | cat
	pip install --upgrade numpy
	pip install -e .

format:  ## Formats code with `autoflake`, `black` and `isort`
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place geometric_kernels tests --exclude=__init__.py
	black geometric_kernels tests
	isort geometric_kernels tests

lint:
	flake8 geometric_kernels tests
	black geometric_kernels tests --check --diff
	isort geometric_kernels tests --check-only --diff
	# Turn off mypy for now
	# mypy geometric_kernels


test:  ## Run the tests, start with the failing ones and break on first fail.
	pytest -v -x --ff -rN -Wignore -s --tb=short --durations=10 tests
