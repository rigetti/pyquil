QUILC_URL=tcp://localhost:5555
QVM_URL=http://localhost:5000

.PHONY: all
all: dist

.PHONY: clean
clean:
	rm -rf dist
	rm -rf pyquil.egg-info
	rm -rf .pytest_cache/

.PHONY: config
config:
	echo "[Rigetti Forest]" > ~/.forest_config
	echo "qvm_address = ${QVM_URL}" >> ~/.forest_config
	echo "quilc_address = ${QUILC_URL}" >> ~/.forest_config
	cat ~/.forest_config

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: docs
docs:
	pandoc --from=markdown --to=rst --output=docs/source/changes.rst CHANGELOG.md
	make -C docs html

.PHONY: info
info:
	python -V
	pip freeze

.PHONY: install
install:
	pip install -e .

.PHONY: requirements
requirements:
	pip install -r requirements.txt

.PHONY: style
style:
	flake8 pyquil

.PHONY: test
test:
	pytest

.PHONY: upload
upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
