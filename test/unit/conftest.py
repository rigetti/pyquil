import os
from typing import Dict, Any

import numpy as np
import pytest
from qcs_sdk import QCSClient
from qcs_sdk.qpu.isa import InstructionSetArchitecture
from qcs_sdk.qvm import QVMClient

from pyquil.api import (
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
from test.unit.utils import DummyCompiler, CLIENT_PUBLIC_KEY, CLIENT_SECRET_KEY, SERVER_PUBLIC_KEY
from .. import override_qcs_config

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
override_qcs_config()


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
    gates_1q = [g.dict() for g in gates_1q]
    gates_2q = [g.dict() for g in gates_2q]
    compiler_isa_dict = {
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
    # from pprint import pprint
    # pprint(compiler_isa_dict)
    return CompilerISA.parse_obj(compiler_isa_dict)


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
        return InstructionSetArchitecture.from_raw(f.read())


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


@pytest.fixture()
def compiler(compiler_quantum_processor: CompilerQuantumProcessor, client_configuration: QCSClient):
    compiler = QVMCompiler(
        quantum_processor=compiler_quantum_processor, timeout=1, client_configuration=client_configuration
    )
    program = Program(I(0))
    compiler.quil_to_native_quil(program)
    return compiler


@pytest.fixture()
def dummy_compiler(qcs_aspen8_quantum_processor: QCSQuantumProcessor, client_configuration: QCSClient):
    return DummyCompiler(qcs_aspen8_quantum_processor, client_configuration)


@pytest.fixture(scope="session")
def client_configuration() -> QCSClient:
    return QCSClient.load()


@pytest.fixture(scope="session")
def qvm_client(client_configuration: QCSClient) -> QVMClient:
    return QVMClient.new_http(client_configuration.qvm_url)


@pytest.fixture(scope="session")
def benchmarker(client_configuration: QCSClient):
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
