.PHONY: help install docs format check test check-and-test

help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with dev requirements =============="
ifeq ("$(UNAME_S)", "Linux")
	sudo apt-get install gfortran
	pip install --upgrade pip setuptools numpy Cython
	pip install --no-cache-dir -U -r requirements.txt | cat
	pip install --upgrade numpy
endif
ifeq ($(UNAME_S),Darwin)
	brew install gcc
	pip install --upgrade pip setuptools numpy Cython
	pip install --upgrade numpy
endif	
	pip install --upgrade --upgrade-strategy eager \
		-r requirements.txt \
		-r dev_requirements.txt \
		-e .

format:  ## Formats code with `autoflake`, `black` and `isort`
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place geometric_kernels tests --exclude=__init__.py
	black geometric_kernels tests
	isort geometric_kernels tests

lint:
	flake8 geometric_kernels tests
	black geometric_kernels tests --check
	isort geometric_kernels tests --check-only
	mypy geometric_kernels


test:  ## Run the tests, start with the failing ones and break on first fail.
	pytest -rN -Wignore --tb=short --durations=10