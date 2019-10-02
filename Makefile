PYPI_URL=https://pypi.org
TEST_PYPI_URL=https://test.pypi.org
QUILC_URL=tcp://quilc:5555
QVM_URL=http://qvm:5000

.PHONY: all
all:
    dist

# 
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
    pip install --index-url ${TEST_PYPI_URL}/simple/ --extra-index-url ${PYPI_URL}/simple pyquil

.PHONY: requirements
requirements:
    pip install -r requirements.txt

.PHONY: setup
setup:
    python setup.py install -e .

.PHONY: style
style:
    flake8 pyquil

.PHONY: test
test:
    pytest

.PHONY: upload
upload:
    twine upload --repository-url ${TEST_PYPI_URL}/legacy/ dist/*
