import pytest

from test import override_qcs_config


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as being 'slow'",
    )

    parser.addoption(
        "--qpu-id",
        action="store",
        metavar="QPU_ID",
        help="specify a quantum processor ID to test against (overrides TEST_QUANTUM_PROCESSOR env var)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow(reason=None): mark test as slow to run")
    config.addinivalue_line(
        "markers",
        "qpu(id=None): mark that a test requires a QPU ID is set "
        "(via --qpu-id or TEST_QUANTUM_PROCESSOR env var) "
        "and (if marked with a specific id) matches the user-given ID",
    )


def pytest_runtest_setup(item):
    if item.config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    for marker in item.iter_markers(name="slow"):
        print(item.name, marker)

    if marker := item.get_closest_marker("slow"):
        reason = marker.kwargs.get("reason", next(iter(marker.args), "this test is slow"))
        pytest.skip(f"{reason}; use --runslow to run")


@pytest.fixture
def qcs_config_env():
    override_qcs_config()
