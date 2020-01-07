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

# The dream is to one day run mypy on the whole tree. For now, limit checks to known-good files.
.PHONY: check-types
check-types:
	mypy pyquil/gate_matrices.py pyquil/gates.py pyquil/noise.py pyquil/numpy_simulator.py \
		pyquil/operator_estimation.py pyquil/parser.py pyquil/paulis.py pyquil/pyqvm.py \
		pyquil/quil.py pyquil/quilatom.py pyquil/quilbase.py pyquil/reference_simulator.py \
		pyquil/unitary_tools.py pyquil/version.py pyquil/wavefunction.py \
		pyquil/device pyquil/experiment pyquil/latex pyquil/simulation

.PHONY: check-style
check-style:
	flake8

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

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: docs
docs: CHANGELOG.md
	pandoc --from=markdown --to=rst --output=docs/source/changes.rst CHANGELOG.md
	make -C docs html

.PHONY: docker
docker: Dockerfile
	docker build -t $(DOCKER_TAG) .

.PHONY: format
format:
	black pyquil

.PHONY: info
info:
	python -V
	pip freeze

.PHONY: install
install:
	pip install -e .

.PHONY: requirements
requirements: requirements.txt
	pip install -r requirements.txt

.PHONY: test
test:
	pytest -v --runslow --cov=pyquil

.PHONY: upload
upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: version
version:
	@git describe --tags | sed 's/v//' | sed 's/\(.*\)-.*/\1/'| sed 's/-/./'
