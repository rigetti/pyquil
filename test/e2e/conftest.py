import os

import pytest
from qcs_api_client.client import QCSClientConfiguration

from pyquil.api import EngagementManager


@pytest.fixture()
def quantum_processor_id() -> str:
    quantum_processor_id = os.environ.get("TEST_QUANTUM_PROCESSOR")

    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    return quantum_processor_id


@pytest.fixture()
def client_configuration() -> QCSClientConfiguration:
    return QCSClientConfiguration.load()


@pytest.fixture()
def engagement_manager(client_configuration: QCSClientConfiguration) -> EngagementManager:
    return EngagementManager(client_configuration=client_configuration)
