COMMIT_HASH=$(shell git rev-parse --short HEAD)
DEFAULT_QUILC_URL=tcp://localhost:5555
DEFAULT_QVM_URL=http://localhost:5000
DOCKER_TAG=rigetti/forest:$(COMMIT_HASH)

.DEFAULT := help

.PHONY: help
help: ## Print this help message
	@awk 'BEGIN { FS=":.*##"; print "Supported Makefile commands:\n" } \
          /^[a-zA-Z0-9_-]+:.*##/ { cmd=$$1; desc=$$2; printf "    \033[36m%-20s\033[0m %s\n", cmd, desc } \
          END { print "" }' $(MAKEFILE_LIST)
		  
.PHONY: all
all: dist ## Build the distribution artifacts

.PHONY: bench
bench: ## Run the benchmarks
	poetry run pytest -v test/benchmarks

.PHONY: check-all
check-all: check-format check-types check-style ## Run all format, type, and style checks

.PHONY: check-format
check-format: ## Check code formatting without modifying files
	ruff format --check --diff pyquil

.PHONY: check-types
check-types: ## Run static type checking with mypy
	mypy pyquil

.PHONY: check-style
check-style: ## Lint the code with ruff
	ruff check pyquil

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf dist
	rm -rf pyquil.egg-info
	rm -rf .pytest_cache/

.PHONY: config
config: ## Write a default Forest config to ~/.forest_config
	echo "[Rigetti Forest]" > ~/.forest_config
	echo "qvm_address = ${DEFAULT_QVM_URL}" >> ~/.forest_config
	echo "quilc_address = ${DEFAULT_QUILC_URL}" >> ~/.forest_config
	cat ~/.forest_config

.PHONY: coverage
coverage: ## Upload test coverage to Coveralls
	coveralls

.PHONY: docs
docs: ## Build the HTML documentation
	poetry install --extras docs --extras latex
	make -C docs clean html

.PHONY: doctest
doctest: ## Run doctests in modules and documentation
	poetry install --extras docs --extras latex
	pytest -v --cov=pyquil --doctest-modules pyquil
	make -C docs clean doctest

.PHONY: docker
docker: Dockerfile ## Build the Forest Docker image
	docker build -t $(DOCKER_TAG) .

.PHONY: format
format: ## Auto-format the code with ruff
	ruff format $(if $(format_file), $(format_file), pyquil)

.PHONY: info
info: ## Print Python version and installed packages
	python -V
	pip freeze

.PHONY: install
install: ## Install the project and its dependencies
	pip install --upgrade pip
	poetry install

.PHONY: test
test: ## Run the full unit test suite with coverage
	poetry install --extras latex
	pytest -v --runslow --cov=pyquil --cov-report xml:coverage.xml test/unit

.PHONY: test-fast
test-fast: ## Run unit tests quickly, stopping at first failure
	poetry install --extras latex
	pytest -vx --cov=pyquil test/unit

.PHONY: e2e
e2e: ## Run the end-to-end tests
	pytest -n 1 -v --cov=pyquil test/e2e

.PHONY: test-all
test-all: doctest test e2e ## Run doctests, unit tests, and e2e tests

docs/quil/grammars/Quil.g4:
	git submodule init
	git submodule update

.PHONY: generate-parser
generate-parser: docs/quil/grammars/Quil.g4 ## Regenerate the Quil ANTLR parser
	cd docs/quil/grammars && antlr -Dlanguage=Python3 -o ../../../pyquil/_parser/gen3 Quil.g4
