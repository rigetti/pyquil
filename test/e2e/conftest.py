import os

import pytest
from qcs_api_client.client import QCSClientConfiguration

from pyquil import get_qc
from pyquil.api import EngagementManager, QuantumComputer


@pytest.fixture()
def qc(client_configuration: QCSClientConfiguration, engagement_manager: EngagementManager) -> QuantumComputer:
    quantum_processor_id = os.environ.get("TEST_QUANTUM_PROCESSOR")

    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    return get_qc(
        quantum_processor_id,
        client_configuration=client_configuration,
        engagement_manager=engagement_manager,
    )


@pytest.fixture()
def client_configuration() -> QCSClientConfiguration:
    return QCSClientConfiguration.load()


@pytest.fixture()
def engagement_manager(client_configuration: QCSClientConfiguration) -> EngagementManager:
    return EngagementManager(client_configuration=client_configuration)
