import os

import pytest
from qcs_sdk import QCSClient

from pyquil import get_qc
from pyquil.api import QuantumComputer

from .. import override_qcs_config


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
def qc(client_configuration: QCSClient, quantum_processor_id: str) -> QuantumComputer:
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
