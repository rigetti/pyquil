import os

import pytest
from pytest import StashKey
from qcs_sdk import QCSClient

from pyquil import get_qc
from pyquil.api import QuantumComputer

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


@pytest.fixture()
def qc(client_configuration: QCSClient, pytestconfig) -> QuantumComputer:
    quantum_processor_id = pytestconfig.stash[_QPU_ID_KEY]

    return get_qc(
        quantum_processor_id,
        client_configuration=client_configuration,
    )


@pytest.fixture()
def client_configuration(qcs_config_env) -> QCSClient:
    _ = qcs_config_env # we just need to ensure the fixture loads
    return QCSClient.load()

