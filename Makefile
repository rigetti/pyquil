COMMIT_HASH=$(shell git rev-parse --short HEAD)
DEFAULT_QUILC_URL=tcp://localhost:5555
DEFAULT_QVM_URL=http://localhost:5000
DOCKER_TAG=rigetti/forest:$(COMMIT_HASH)

.PHONY: all
all: dist

.PHONY: check-all
check-all: check-format check-types check-style

.PHONY: check-format
check-format:
	black --check --diff pyquil

.PHONY: check-types
check-types:
	mypy pyquil

.PHONY: check-style
check-style:
	flake8 pyquil

.PHONY: clean
clean:
	rm -rf dist
	rm -rf pyquil.egg-info
	rm -rf .pytest_cache/

.PHONY: config
config:
	echo "[Rigetti Forest]" > ~/.forest_config
	echo "qvm_address = ${DEFAULT_QVM_URL}" >> ~/.forest_config
	echo "quilc_address = ${DEFAULT_QUILC_URL}" >> ~/.forest_config
	cat ~/.forest_config

.PHONY: coverage
coverage:
	coveralls

.PHONY: docs
docs:
	poetry install --extras docs --extras latex
	make -C docs clean html

.PHONY: doctest
doctest:
	poetry install --extras docs --extras latex
	pytest -v --cov=pyquil --doctest-modules pyquil
	make -C docs clean doctest

.PHONY: docker
docker: Dockerfile
	docker build -t $(DOCKER_TAG) .

.PHONY: format
format:
	black $(if $(format_file), $(format_file), pyquil)

.PHONY: info
info:
	python -V
	pip freeze

.PHONY: install
install:
	poetry install

.PHONY: test
test:
	poetry install --extras latex
	pytest -v --runslow --cov=pyquil --cov-report xml:coverage.xml test/unit

.PHONY: test-fast
test-fast:
	poetry install --extras latex
	pytest -vx --cov=pyquil test/unit

.PHONY: e2e
e2e:
	pytest -n 1 -v --cov=pyquil test/e2e

.PHONY: test-all
test-all: doctest test e2e

docs/quil/grammars/Quil.g4:
	git submodule init
	git submodule update

.PHONY: generate-parser
generate-parser: docs/quil/grammars/Quil.g4
	cd docs/quil/grammars && antlr -Dlanguage=Python3 -o ../../../pyquil/_parser/gen3 Quil.g4
