"""Shared options for the benchmark suite."""

import pytest


def pytest_addoption(parser):
    # ``test/unit/conftest.py`` registers the same flag; guard against both being collected.
    try:
        parser.addoption("--runslow", action="store_true", default=False, help="run tests marked as being 'slow'")
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords and not any(mark.name == "skip" for mark in item.iter_markers()):
            item.add_marker(skip_slow)
