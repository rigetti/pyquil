import os

import pytest
from qcs_sdk import QCSClient

from pyquil import get_qc
from pyquil.api import QuantumComputer

from .. import override_qcs_config


@pytest.fixture()
def qc(client_configuration: QCSClient) -> QuantumComputer:
    quantum_processor_id = os.environ.get("TEST_QUANTUM_PROCESSOR")

    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    return get_qc(
        quantum_processor_id,
        client_configuration=client_configuration,
    )


@pytest.fixture()
def client_configuration() -> QCSClient:
    override_qcs_config()
    return QCSClient.load()
