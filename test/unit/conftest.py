import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import rpcq
from qcs_api_client.client._configuration import QCSClientConfiguration
from qcs_api_client.models import InstructionSetArchitecture, EngagementCredentials

from pyquil.api import (
    QVMConnection,
    QVMCompiler,
    BenchmarkConnection,
)
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I
from pyquil.paulis import sX
from pyquil.quantum_processor import QCSQuantumProcessor, CompilerQuantumProcessor
from pyquil.quantum_processor.transformers.graph_to_compiler_isa import (
    DEFAULT_1Q_GATES,
    DEFAULT_2Q_GATES,
    _transform_edge_operation_to_gates,
    _transform_qubit_operation_to_gates,
)
from pyquil.quil import Program
from test.unit.utils import DummyCompiler

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


@pytest.fixture
def compiler_isa() -> CompilerISA:
    """
    Configures an arbitrary ``CompilerISA`` that may be used to initialize
    a ``CompilerQuantumProcessor``. Developers should create specific test cases of
    ``CompilerISA`` as separate fixtures in conftest.py or in the test file.
    """
    gates_1q = []
    for gate in DEFAULT_1Q_GATES:
        gates_1q.extend(_transform_qubit_operation_to_gates(gate))
    gates_2q = []
    for gate in DEFAULT_2Q_GATES:
        gates_2q.extend(_transform_edge_operation_to_gates(gate))
    return CompilerISA.parse_obj(
        {
            "1Q": {
                "0": {"id": 0, "gates": gates_1q},
                "1": {"id": 1, "gates": gates_1q},
                "2": {"id": 2, "gates": gates_1q},
                "3": {"id": 3, "dead": True},
            },
            "2Q": {
                "0-1": {"ids": [0, 1], "gates": gates_2q},
                "1-2": {
                    "ids": [1, 2],
                    "gates": [
                        {
                            "operator_type": "gate",
                            "operator": "ISWAP",
                            "parameters": [],
                            "arguments": ["_", "_"],
                        }
                    ],
                },
                "0-2": {
                    "ids": [0, 2],
                    "gates": [
                        {
                            "operator_type": "gate",
                            "operator": "CPHASE",
                            "parameters": ["theta"],
                            "arguments": ["_", "_"],
                        },
                    ],
                },
                "0-3": {"ids": [0, 3], "dead": True},
            },
        }
    )


@pytest.fixture()
def compiler_quantum_processor(compiler_isa: CompilerISA) -> CompilerQuantumProcessor:
    return CompilerQuantumProcessor(isa=compiler_isa)


@pytest.fixture
def aspen8_compiler_isa() -> CompilerISA:
    """
    Read the Aspen-8 QCS ``CompilerISA`` from file. This should be an exact conversion of
    qcs_aspen8_isa to a ``CompilerISA``.
    """
    return CompilerISA.parse_file(os.path.join(TEST_DATA_DIR, "compiler-isa-Aspen-8.json"))


@pytest.fixture
def qcs_aspen8_isa() -> InstructionSetArchitecture:
    """
    Read the Aspen-8 QCS InstructionSetArchitecture from file and load it into
    the ``InstructionSetArchitecture`` QCS API client model.
    """
    with open(os.path.join(TEST_DATA_DIR, "qcs-isa-Aspen-8.json")) as f:
        return InstructionSetArchitecture.from_dict(json.load(f))


@pytest.fixture
def qcs_aspen8_quantum_processor(qcs_aspen8_isa: InstructionSetArchitecture) -> QCSQuantumProcessor:
    return QCSQuantumProcessor(quantum_processor_id="Aspen-8", isa=qcs_aspen8_isa)


@pytest.fixture
def noise_model_dict() -> Dict[str, Any]:
    return {
        "gates": [
            {
                "gate": "I",
                "params": (5.0,),
                "targets": (0, 1),
                "kraus_ops": [[[[1.0]], [[1.0]]]],
                "fidelity": 1.0,
            },
            {
                "gate": "RX",
                "params": (np.pi / 2.0,),
                "targets": (0,),
                "kraus_ops": [[[[1.0]], [[1.0]]]],
                "fidelity": 1.0,
            },
        ],
        "assignment_probs": {"1": [[1.0, 0.0], [0.0, 1.0]], "0": [[1.0, 0.0], [0.0, 1.0]]},
    }


@pytest.fixture(scope="session")
def qvm_connection(client_configuration: QCSClientConfiguration):
    return QVMConnection(
        client_configuration=client_configuration,
        random_seed=52,
    )


@pytest.fixture()
def compiler(compiler_quantum_processor: CompilerQuantumProcessor, client_configuration: QCSClientConfiguration):
    compiler = QVMCompiler(
        quantum_processor=compiler_quantum_processor, timeout=1, client_configuration=client_configuration
    )
    program = Program(I(0))
    compiler.quil_to_native_quil(program)
    return compiler


@pytest.fixture()
def dummy_compiler(qcs_aspen8_quantum_processor: QCSQuantumProcessor, client_configuration: QCSClientConfiguration):
    return DummyCompiler(qcs_aspen8_quantum_processor, client_configuration)


@pytest.fixture(scope="session")
def client_configuration() -> QCSClientConfiguration:
    return QCSClientConfiguration.load(
        secrets_file_path=Path(os.path.join(TEST_DATA_DIR, "qcs_secrets.toml")),
        settings_file_path=Path(os.path.join(TEST_DATA_DIR, "qcs_settings.toml")),
    )


# Valid, sample Z85-encoded keys specified by zmq curve for testing:
#   http://api.zeromq.org/master:zmq-curve#toc4
CLIENT_PUBLIC_KEY = "Yne@$w-vo<fVvi]a<NY6T1ed:M$fCG*[IaLV{hID"
CLIENT_SECRET_KEY = "D:)Q[IlAW!ahhC2ac:9*A}h:p?([4%wOTJ%JR%cs"
SERVER_PUBLIC_KEY = "rq:rM>}U?@Lns47E1%kR.o@n%FcmmsL/@{H8]yf7"
SERVER_SECRET_KEY = "JTKVSB%%)wK0E.X)V>+}o?pNmC{O&4W4b!Ni{Lh6"


@pytest.fixture(scope="session")
def engagement_credentials() -> EngagementCredentials:
    return EngagementCredentials(
        client_public=CLIENT_PUBLIC_KEY,
        client_secret=CLIENT_SECRET_KEY,
        server_public=SERVER_PUBLIC_KEY,
    )


@pytest.fixture(scope="function")
def rpcq_server_with_auth() -> rpcq.Server:
    auth_config = rpcq.ServerAuthConfig(
        server_secret_key=SERVER_SECRET_KEY.encode(),
        server_public_key=SERVER_PUBLIC_KEY.encode(),
        client_keys_directory=os.path.join(TEST_DATA_DIR, "rpcq_client_key"),
    )
    return rpcq.Server(auth_config=auth_config)


@pytest.fixture(scope="function")
def rpcq_server() -> rpcq.Server:
    return rpcq.Server()


@pytest.fixture(scope="session")
def benchmarker(client_configuration: QCSClientConfiguration):
    bm = BenchmarkConnection(timeout=2, client_configuration=client_configuration)
    bm.apply_clifford_to_pauli(Program(I(0)), sX(0))
    return bm


def _str_to_bool(s: str):
    """Convert either of the strings 'True' or 'False' to their Boolean equivalent"""
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError("Please specify either True or False")


def pytest_addoption(parser):
    parser.addoption(
        "--use-seed",
        action="store",
        type=_str_to_bool,
        default=True,
        help="run operator estimation tests faster by using a fixed random seed",
    )
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked as being 'slow'")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture()
def use_seed(pytestconfig):
    return pytestconfig.getoption("use_seed")
