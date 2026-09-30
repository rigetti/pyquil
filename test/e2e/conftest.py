import os

import pytest
from pytest import StashKey
from qcs_sdk import QCSClient

from pyquil import get_qc
from pyquil.api import QuantumComputer
from test import override_qcs_config

_QPU_ID_KEY = StashKey[str]()


def pytest_runtest_setup(item):
    user_qpu_id = item.config.getoption("--qpu-id") or os.environ.get("TEST_QUANTUM_PROCESSOR")
    item.config.stash[_QPU_ID_KEY] = user_qpu_id

    if not (marker := item.get_closest_marker("qpu")):
        return

    test_qpu_id = marker.kwargs.get("id", next(iter(marker.args), None))
    if not user_qpu_id:
        qid = f"QPU ID={test_qpu_id}" if test_qpu_id else "a QPU ID"
        pytest.skip(f"test requires {qid} (specify one with --qpu-id or TEST_QUANTUM_PROCESSOR env var)")
    elif test_qpu_id and user_qpu_id != test_qpu_id:
        pytest.skip(f"test requires QPU ID={test_qpu_id}, but currently testing {user_qpu_id}")


def pytest_addoption(parser: pytest.Parser):
    """Add command line option to skip tests marked integration"""
    parser.addoption(
        "--live-qpu-access",
        action="store_true",
        default=False,
        help="run tests that require access to a live QPU",
    )


def pytest_configure(config: pytest.Config):
    """Register custom marker 'live_qpu_access'"""
    config.addinivalue_line(
        "markers",
        "live_qpu_access: mark test as requiring live access to a QPU for execution",
    )


@pytest.fixture(scope="session")
def quantum_processor_id() -> str:
    return os.environ.get("TEST_QUANTUM_PROCESSOR", "Cepheus-1-108Q")


@pytest.fixture()
def qc(client_configuration: QCSClient, pytestconfig) -> QuantumComputer:
    quantum_processor_id = pytestconfig.stash[_QPU_ID_KEY]

    return get_qc(
        quantum_processor_id,
        client_configuration=client_configuration,
    )


@pytest.fixture()
def client_configuration(live_qpu_access: bool) -> QCSClient:
    if not live_qpu_access:
        override_qcs_config()
    return QCSClient.load()


@pytest.fixture(scope="session")
def live_qpu_access(request: pytest.FixtureRequest) -> bool:
    return (
        request.config.getoption("--live-qpu-access") is not None
        and request.config.getoption("--live-qpu-access") is not False
    )
