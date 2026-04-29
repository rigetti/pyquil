import itertools
from typing import cast

import networkx as nx
import numpy as np
import pytest
import respx
from qcs_sdk import QCSClient
from qcs_sdk.qpu.isa import InstructionSetArchitecture
from syrupy.assertion import SnapshotAssertion

from pyquil import Program, list_quantum_computers
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import (
    QuantumComputer,
    _get_qvm_with_topology,
    _parse_name,
    get_qc,
)
from pyquil.api._qvm import QVM
from pyquil.gates import CNOT, MEASURE, RX, H, I, X
from pyquil.quantum_processor import NxQuantumProcessor
from pyquil.quilbase import Declare, MemoryReference
from test.unit.utils import DummyCompiler


def test_list_qc():
    qc_names = list_quantum_computers(qpus=False)
    assert qc_names == ["9q-square-qvm", "9q-square-noisy-qvm"]


def test_parse_qc_name():
    name, qvm_type, noisy = _parse_name("9q-generic", None, None)
    assert name == "9q-generic"
    assert qvm_type is None
    assert not noisy

    name, qvm_type, noisy = _parse_name("9q-generic-qvm", None, None)
    assert name == "9q-generic"
    assert qvm_type == "qvm"
    assert not noisy

    name, qvm_type, noisy = _parse_name("9q-generic-noisy-qvm", None, None)
    assert name == "9q-generic"
    assert qvm_type == "qvm"
    assert noisy


def test_parse_qc_flags():
    name, qvm_type, noisy = _parse_name("9q-generic", False, False)
    assert name == "9q-generic"
    assert qvm_type is None
    assert not noisy

    name, qvm_type, noisy = _parse_name("9q-generic", True, None)
    assert name == "9q-generic"
    assert qvm_type == "qvm"
    assert not noisy

    name, qvm_type, noisy = _parse_name("9q-generic", True, True)
    assert name == "9q-generic"
    assert qvm_type == "qvm"
    assert noisy


def test_parse_qc_redundant():
    name, qvm_type, noisy = _parse_name("9q-generic", False, False)
    assert name == "9q-generic"
    assert qvm_type is None
    assert not noisy

    name, qvm_type, noisy = _parse_name("9q-generic-qvm", True, False)
    assert name == "9q-generic"
    assert qvm_type == "qvm"
    assert not noisy

    name, qvm_type, noisy = _parse_name("9q-generic-noisy-qvm", True, True)
    assert name == "9q-generic"
    assert qvm_type == "qvm"
    assert noisy


def test_parse_qc_conflicting():
    with pytest.raises(ValueError) as e:
        name, qvm_type, noisy = _parse_name("9q-generic-qvm", False, False)

    assert e.match(r".*but you have specified `as_qvm=False`")

    with pytest.raises(ValueError) as e:
        name, qvm_type, noisy = _parse_name("9q-generic-noisy-qvm", True, False)
    assert e.match(r".*but you have specified `noisy=False`")


def test_parse_qc_strip():
    # Originally used `str.strip` to remove the suffixes. This is not correct!
    name, _, _ = _parse_name("mvq-qvm", None, None)
    assert name == "mvq"

    name, _, _ = _parse_name("mvq-noisy-qvm", None, None)
    assert name == "mvq"


def test_parse_qc_no_prefix():
    prefix, qvm_type, noisy = _parse_name("qvm", None, None)
    assert qvm_type == "qvm"
    assert not noisy
    assert prefix == ""

    prefix, qvm_type, noisy = _parse_name("", True, None)
    assert qvm_type == "qvm"
    assert not noisy
    assert prefix == ""


def test_parse_qc_no_prefix_2():
    prefix, qvm_type, noisy = _parse_name("noisy-qvm", None, None)
    assert qvm_type == "qvm"
    assert noisy
    assert prefix == ""

    prefix, qvm_type, noisy = _parse_name("", True, True)
    assert qvm_type == "qvm"
    assert noisy
    assert prefix == ""


def test_qc(client_configuration: QCSClient):
    qc = get_qc("9q-square-noisy-qvm", client_configuration=client_configuration)
    assert isinstance(qc, QuantumComputer)
    assert qc.qubit_topology().number_of_nodes() == 9
    assert qc.qubit_topology().degree[0] == 2
    assert qc.qubit_topology().degree[4] == 4
    assert str(qc) == "9q-square-noisy-qvm"


def test_qc_run(client_configuration: QCSClient):
    qc = get_qc("9q-square-noisy-qvm", client_configuration=client_configuration)
    program = Program(
        Declare("ro", "BIT", 1),
        X(0),
        MEASURE(0, ("ro", 0)),
    ).wrap_in_numshots_loop(3)
    compiled_program = qc.compile(program)
    bs = qc.run(compiled_program).readout_data.get("ro")
    assert bs.shape == (3, 1)


def test_nq_qvm_qc(client_configuration: QCSClient):
    for n_qubits in [2, 4, 7, 19]:
        qc = get_qc(f"{n_qubits}q-qvm", client_configuration=client_configuration)
        for q1, q2 in itertools.permutations(range(n_qubits), r=2):
            assert (q1, q2) in qc.qubit_topology().edges
        assert qc.name == f"{n_qubits}q-qvm"


def test_qc_noisy(client_configuration: QCSClient):
    qc = get_qc("5q", as_qvm=True, noisy=True, client_configuration=client_configuration)
    assert isinstance(qc, QuantumComputer)


def test_qc_compile(dummy_compiler: DummyCompiler, client_configuration: QCSClient):
    qc = get_qc("5q", as_qvm=True, noisy=True, client_configuration=client_configuration)
    qc.compiler = dummy_compiler
    prog = Program()
    prog += H(0)
    assert qc.compile(prog) == prog


def test_qc_error(client_configuration: QCSClient):
    # QVM is not a QPU
    with pytest.raises(ValueError):
        get_qc("9q-square-noisy-qvm", as_qvm=False, client_configuration=client_configuration)

    with pytest.raises(ValueError):
        get_qc("5q", as_qvm=False, client_configuration=client_configuration)


@pytest.mark.parametrize("params", [[np.pi], np.array([np.pi])])
def test_run_with_parameters(client_configuration: QCSClient, params):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    executable = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(1000)

    bitstrings = qc.run(executable, {"theta": params}).readout_data.get("ro")

    assert bitstrings.shape == (1000, 1)
    assert all([bit == 1 for bit in bitstrings])


@pytest.mark.parametrize("param", ["not_a_number", ["not_a_number"]])
def test_run_with_bad_parameters(client_configuration: QCSClient, param):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    executable = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(1000)

    with pytest.raises((TypeError, ValueError)):
        qc.run(executable, {"theta": [param]})


def test_reset(client_configuration: QCSClient):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    p = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(10)
    result = qc.qam.run(p, {"theta": [np.pi]})

    assert result.readout_data["ro"].shape == (10, 1)
    assert all([bit == 1 for bit in result.readout_data["ro"]])


def test_get_qvm_with_topology(client_configuration: QCSClient):
    topo = nx.from_edgelist([(5, 6), (6, 7), (10, 11)])
    qc = _get_qvm_with_topology(
        name="test-qvm",
        topology=topo,
        noisy=False,
        qvm_type="qvm",
        compiler_timeout=5.0,
        client_configuration=client_configuration,
    )
    assert len(qc.qubits()) == 5
    assert min(qc.qubits()) == 5


def test_get_qvm_with_topology_2(client_configuration: QCSClient):
    topo = nx.from_edgelist([(5, 6), (6, 7)])
    qc = _get_qvm_with_topology(
        name="test-qvm",
        topology=topo,
        noisy=False,
        qvm_type="qvm",
        compiler_timeout=5.0,
        client_configuration=client_configuration,
    )
    results = qc.run(
        qc.compile(
            Program(
                Declare("ro", "BIT", 3),
                X(5),
                MEASURE(5, ("ro", 0)),
                MEASURE(6, ("ro", 1)),
                MEASURE(7, ("ro", 2)),
            ).wrap_in_numshots_loop(5)
        )
    ).readout_data.get("ro")
    assert results.shape == (5, 3)
    assert all(r[0] == 1 for r in results)


def test_parse_mix_qvm_and_noisy_flag():
    # https://github.com/rigetti/pyquil/issues/764
    name, qvm_type, noisy = _parse_name("1q-qvm", as_qvm=None, noisy=True)
    assert noisy


def test_undeclared_memory_region(client_configuration: QCSClient, dummy_compiler: DummyCompiler):
    """Test for https://github.com/rigetti/pyquil/issues/1596"""
    program = Program(
        """
DECLARE beta REAL[1]
RZ(0.5) 0
CPHASE(pi) 0 1
DECLARE ro BIT[2]
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""
    )
    program = program.copy_everything_except_instructions()
    assert len(program.instructions) == 0
    assert len(program.declarations) == 0
    qc = QuantumComputer(name="testy!", qam=QVM(), compiler=dummy_compiler)
    executable = qc.compiler.native_quil_to_executable(program)
    qc.run(executable)


# We sometimes narrowly miss the np.mean(parity) < 0.15 assertion, below.
@pytest.mark.flaky(reruns=1)
def test_run_noisy(client_configuration: QCSClient):
    from pyquil.noise import Channel, NoiseModel
    from pyquil.quilbase import Gate as QuilGate
    from pyquil.quilatom import Qubit as QuilQubit

    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    # Build noise model with explicit gate fidelities for the gates we use
    channels = set()
    for q in range(3):
        h_inst = QuilGate("H", [], [QuilQubit(q)])
        channels.add(Channel.from_gate_fidelity(inst=h_inst, fidelity=0.95))
    for q0, q1 in [(0, 1), (1, 2)]:
        cnot_inst = QuilGate("CNOT", [], [QuilQubit(q0), QuilQubit(q1)])
        channels.add(Channel.from_gate_fidelity(inst=cnot_inst, fidelity=0.95))
    noise_model = NoiseModel(channels=frozenset(channels))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(noise_model=noise_model),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    p = Program(
        Declare("ro", "BIT", 3),
        H(0),
        CNOT(0, 1),
        CNOT(1, 2),
        MEASURE(0, MemoryReference("ro", 0)),
        MEASURE(1, MemoryReference("ro", 1)),
        MEASURE(2, MemoryReference("ro", 2)),
    ).wrap_in_numshots_loop(1000)
    result = qc.run(p)
    bitstrings = result.readout_data.get("ro")

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


@pytest.mark.skip  # qcs_sdk client profiles do not support group accounts
@respx.mock
def test_get_qc_with_group_account(
    client_configuration: QCSClient,
    qcs_aspen8_isa: InstructionSetArchitecture,
):
    """Assert that a client may specify a ``QCSClientSettingsProfile`` representing a QCS group
    account.
    """
    respx.get(
        url=f"{client_configuration.api_url}/v1/quantumProcessors/test/instructionSetArchitecture",
    ).respond(json=qcs_aspen8_isa.json())

    group_profile = client_configuration.profile.copy()
    group_profile.account_id = "group0"
    group_profile.account_type = "group"
    client_configuration.settings.profiles["my-group-profile"] = group_profile
    client_configuration.profile_name = "my-group-profile"
    qc = get_qc("test", endpoint_id="test-endpoint", client_configuration=client_configuration)

    assert isinstance(qc, QuantumComputer)
    quantum_computer = cast(QuantumComputer, qc)
    assert isinstance(quantum_computer.qam, QPU)
    qpu = cast(QPU, quantum_computer.qam)
    engagement_manager = qpu._qpu_client._engagement_manager

    respx.post(
        url=f"{client_configuration.profile.api_url}/v1/engagements",
        headers__contains={
            "X-QCS-ACCOUNT-ID": "group0",
            "X-QCS-ACCOUNT-TYPE": QCSAccountType.group.value,
        },
    ).respond(
        json={
            "address": "address",
            "endpointId": "endpointId",
            "quantumProcessorId": "quantumProcessorId",
            "userId": "userId",
            "expiresAt": "01-01-2200T00:00:00Z",
            "credentials": {
                "clientPublic": "faux",
                "clientSecret": "faux",
                "serverPublic": "faux",
            },
        }
    )

    engagement = engagement_manager.get_engagement(quantum_processor_id="test")
    assert "faux" == engagement.credentials.client_public
