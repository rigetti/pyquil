import itertools
import random
from test.unit.utils import DummyCompiler

import networkx as nx
import numpy as np
import pytest
from pyquil import Program, list_quantum_computers
from pyquil.api import QCSClientConfiguration
from pyquil.api._quantum_computer import (
    QuantumComputer,
    _check_min_num_trials_for_symmetrized_readout,
    _consolidate_symmetrization_outputs,
    _construct_orthogonal_array,
    _construct_strength_three_orthogonal_array,
    _construct_strength_two_orthogonal_array,
    _flip_array_to_prog,
    _get_qvm_with_topology,
    _measure_bitstrings,
    _parse_name,
    _symmetrization,
    get_qc,
)
from pyquil.api._qvm import QVM
from pyquil.experiment import Experiment, ExperimentSetting
from pyquil.experiment._main import _pauli_to_product_state
from pyquil.gates import CNOT, MEASURE, RESET, RX, RY, H, I, X
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import sX, sY, sZ
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import NxQuantumProcessor
from pyquil.quilbase import Declare, MemoryReference
from rpcq.messages import ParameterAref


def test_flip_array_to_prog():
    # no flips
    flip_prog = _flip_array_to_prog((0, 0, 0, 0, 0, 0), [0, 1, 2, 3, 4, 5])
    assert flip_prog.out().splitlines() == []
    # mixed flips
    flip_prog = _flip_array_to_prog((1, 0, 1, 0, 1, 1), [0, 1, 2, 3, 4, 5])
    assert flip_prog.out().splitlines() == ["RX(pi) 0", "RX(pi) 2", "RX(pi) 4", "RX(pi) 5"]
    # flip all
    flip_prog = _flip_array_to_prog((1, 1, 1, 1, 1, 1), [0, 1, 2, 3, 4, 5])
    assert flip_prog.out().splitlines() == [
        "RX(pi) 0",
        "RX(pi) 1",
        "RX(pi) 2",
        "RX(pi) 3",
        "RX(pi) 4",
        "RX(pi) 5",
    ]


def test_symmetrization():
    prog = Program(I(0), I(1))
    meas_qubits = [0, 1]
    # invalid input if symm_type < -1 or > 3
    with pytest.raises(ValueError):
        _, _ = _symmetrization(prog, meas_qubits, symm_type=-2)
    with pytest.raises(ValueError):
        _, _ = _symmetrization(prog, meas_qubits, symm_type=4)
    # exhaustive symm
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=-1)
    assert sym_progs[0].out().splitlines() == ["I 0", "I 1"]
    assert sym_progs[1].out().splitlines() == ["I 0", "I 1", "RX(pi) 1"]
    assert sym_progs[2].out().splitlines() == ["I 0", "I 1", "RX(pi) 0"]
    assert sym_progs[3].out().splitlines() == ["I 0", "I 1", "RX(pi) 0", "RX(pi) 1"]
    right = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 0 i.e. no symm
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=-1)
    assert sym_progs[0].out().splitlines() == ["I 0", "I 1"]
    right = [np.array([0, 0])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 1
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=1)
    assert sym_progs[0].out().splitlines() == ["I 0", "I 1"]
    assert sym_progs[1].out().splitlines() == ["I 0", "I 1", "RX(pi) 0", "RX(pi) 1"]
    right = [np.array([0, 0]), np.array([1, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 2
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=2)
    assert sym_progs[0].out().splitlines() == ["I 0", "I 1"]
    assert sym_progs[1].out().splitlines() == ["I 0", "I 1", "RX(pi) 0"]
    assert sym_progs[2].out().splitlines() == ["I 0", "I 1", "RX(pi) 1"]
    assert sym_progs[3].out().splitlines() == ["I 0", "I 1", "RX(pi) 0", "RX(pi) 1"]
    right = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 3
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=3)
    assert sym_progs[0].out().splitlines() == ["I 0", "I 1", "RX(pi) 0", "RX(pi) 1"]
    assert sym_progs[1].out().splitlines() == ["I 0", "I 1", "RX(pi) 0"]
    assert sym_progs[2].out().splitlines() == ["I 0", "I 1"]
    assert sym_progs[3].out().splitlines() == ["I 0", "I 1", "RX(pi) 1"]
    right = [np.array([1, 1]), np.array([1, 0]), np.array([0, 0]), np.array([0, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])


def test_construct_orthogonal_array():
    # check for valid inputs
    with pytest.raises(ValueError):
        _construct_orthogonal_array(3, strength=-1)
    with pytest.raises(ValueError):
        _construct_orthogonal_array(3, strength=4)
    with pytest.raises(ValueError):
        _construct_orthogonal_array(3, strength=100)


def test_construct_strength_three_orthogonal_array():
    # This is test is table 1.3 in [OATA]. Next to the np.array below the "line" number refers to
    # the row in table 1.3. It is not important that the rows are switched! Specifically
    #  "A permutation of the runs or factors in an orthogonal array results in an orthogonal
    #  array with the same parameters." page 27 of [OATA].
    #
    # [OATA] Orthogonal Arrays Theory and Applications
    #        Hedayat, Sloane, Stufken
    #        Springer, 1999
    answer = np.array(
        [
            [1, 1, 1, 1],  # line 8
            [1, 0, 1, 0],  # line 6
            [1, 1, 0, 0],  # line 7
            [1, 0, 0, 1],  # line 5
            [0, 0, 0, 0],  # line 1
            [0, 1, 0, 1],  # line 3
            [0, 0, 1, 1],  # line 2
            [0, 1, 1, 0],
        ]
    )  # line 4
    assert np.allclose(_construct_strength_three_orthogonal_array(4), answer)


def test_construct_strength_two_orthogonal_array():
    # This is example 1.5 in [OATA]. Next to the np.array below the "line" number refers to
    # the row in example 1.5.
    answer = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])  # line 1  # line 3  # line 2  # line 4
    assert np.allclose(_construct_strength_two_orthogonal_array(3), answer)


def test_measure_bitstrings(client_configuration: QCSClientConfiguration):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(2))
    dummy_compiler = DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration)
    qc_pyqvm = QuantumComputer(name="testy!", qam=PyQVM(n_qubits=2), compiler=dummy_compiler)
    qc_forest = QuantumComputer(
        name="testy!",
        qam=QVM(client_configuration=client_configuration, gate_noise=(0.00, 0.00, 0.00)),
        compiler=dummy_compiler,
    )
    prog = Program(I(0), I(1))
    meas_qubits = [0, 1]
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=-1)
    results = _measure_bitstrings(qc_pyqvm, sym_progs, meas_qubits, num_shots=1)
    # test with pyQVM
    answer = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
    assert all([np.allclose(x, y) for x, y in zip(results, answer)])
    # test with regular QVM
    results = _measure_bitstrings(qc_forest, sym_progs, meas_qubits, num_shots=1)
    assert all([np.allclose(x, y) for x, y in zip(results, answer)])


def test_consolidate_symmetrization_outputs():
    flip_arrays = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
    # if results = flip_arrays should be a matrix of zeros
    ans1 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    assert np.allclose(_consolidate_symmetrization_outputs(flip_arrays, flip_arrays), ans1)
    results = [np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]])]
    # results are all zero then output should be
    ans2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.allclose(_consolidate_symmetrization_outputs(results, flip_arrays), ans2)


def test_check_min_num_trials_for_symmetrized_readout():
    # trials = -2 should get bumped up to 4 trials
    with pytest.warns(Warning):
        assert _check_min_num_trials_for_symmetrized_readout(num_qubits=2, trials=-2, symm_type=-1) == 4
    # can't have symm_type < -2 or > 3
    with pytest.raises(ValueError):
        _check_min_num_trials_for_symmetrized_readout(num_qubits=2, trials=-2, symm_type=-2)
    with pytest.raises(ValueError):
        _check_min_num_trials_for_symmetrized_readout(num_qubits=2, trials=-2, symm_type=4)


# We sometimes narrowly miss the np.mean(parity) < 0.15 assertion, below. Alternatively, that upper
# bound could be relaxed.
@pytest.mark.flaky(reruns=1)
def test_run(client_configuration: QCSClientConfiguration):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client_configuration=client_configuration, gate_noise=(0.01, 0.01, 0.01)),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    result = qc.run(
        Program(
            Declare("ro", "BIT", 3),
            H(0),
            CNOT(0, 1),
            CNOT(1, 2),
            MEASURE(0, MemoryReference("ro", 0)),
            MEASURE(1, MemoryReference("ro", 1)),
            MEASURE(2, MemoryReference("ro", 2)),
        ).wrap_in_numshots_loop(1000)
    )
    bitstrings = result.readout_data.get('ro')

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


def test_run_pyqvm_noiseless(client_configuration: QCSClientConfiguration):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=PyQVM(n_qubits=3),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    prog = Program(H(0), CNOT(0, 1), CNOT(1, 2))
    ro = prog.declare("ro", "BIT", 3)
    for q in range(3):
        prog += MEASURE(q, ro[q])
    result = qc.run(prog.wrap_in_numshots_loop(1000))
    bitstrings = result.readout_data.get('ro')

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert np.mean(parity) == 0


def test_run_pyqvm_noisy(client_configuration: QCSClientConfiguration):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=PyQVM(n_qubits=3, post_gate_noise_probabilities={"relaxation": 0.01}),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    prog = Program(H(0), CNOT(0, 1), CNOT(1, 2))
    ro = prog.declare("ro", "BIT", 3)
    for q in range(3):
        prog += MEASURE(q, ro[q])
    result = qc.run(prog.wrap_in_numshots_loop(1000))
    bitstrings = result.readout_data.get('ro')

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


def test_readout_symmetrization(client_configuration: QCSClientConfiguration):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    noise_model = decoherence_noise_with_asymmetric_ro(quantum_processor.to_compiler_isa())
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client_configuration=client_configuration, noise_model=noise_model),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )

    prog = Program(
        Declare("ro", "BIT", 2),
        I(0),
        X(1),
        MEASURE(0, MemoryReference("ro", 0)),
        MEASURE(1, MemoryReference("ro", 1)),
    )
    prog.wrap_in_numshots_loop(1000)

    result_1 = qc.run(prog)
    bitstrings_1 = result_1.readout_data.get('ro')
    avg0_us = np.mean(bitstrings_1[:, 0])
    avg1_us = 1 - np.mean(bitstrings_1[:, 1])
    diff_us = avg1_us - avg0_us
    assert diff_us > 0.03

    prog = Program(
        I(0),
        X(1),
    )
    bitstrings_2 = qc.run_symmetrized_readout(prog, 1000)
    avg0_s = np.mean(bitstrings_2[:, 0])
    avg1_s = 1 - np.mean(bitstrings_2[:, 1])
    diff_s = avg1_s - avg0_s
    assert diff_s < 0.05


@pytest.mark.slow
def test_run_symmetrized_readout_error(client_configuration: QCSClientConfiguration):
    # This test checks if the function runs for any possible input on a small number of qubits.
    # Locally this test was run on all 8 qubits, but it was slow.
    qc = get_qc("8q-qvm", client_configuration=client_configuration)
    sym_type_vec = [-1, 0, 1, 2, 3]
    prog_vec = [Program(I(x) for x in range(0, 3))[0:n] for n in range(0, 4)]
    trials_vec = list(range(0, 5))
    for prog, trials, sym_type in itertools.product(prog_vec, trials_vec, sym_type_vec):
        print(qc.run_symmetrized_readout(prog, trials, sym_type))


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


def test_parse_qc_pyqvm():
    prefix, qvm_type, noisy = _parse_name("9q-generic-pyqvm", None, None)
    assert prefix == "9q-generic"
    assert qvm_type == "pyqvm"
    assert not noisy


def test_qc(client_configuration: QCSClientConfiguration):
    qc = get_qc("9q-square-noisy-qvm", client_configuration=client_configuration)
    assert isinstance(qc, QuantumComputer)
    assert qc.qam.noise_model is not None
    assert qc.qubit_topology().number_of_nodes() == 9
    assert qc.qubit_topology().degree[0] == 2
    assert qc.qubit_topology().degree[4] == 4
    assert str(qc) == "9q-square-noisy-qvm"


def test_qc_run(client_configuration: QCSClientConfiguration):
    qc = get_qc("9q-square-noisy-qvm", client_configuration=client_configuration)
    bs = qc.run(
        qc.compile(
            Program(
                Declare("ro", "BIT", 1),
                X(0),
                MEASURE(0, ("ro", 0)),
            ).wrap_in_numshots_loop(3)
        )
    ).readout_data.get('ro')
    assert bs.shape == (3, 1)


def test_nq_qvm_qc(client_configuration: QCSClientConfiguration):
    for n_qubits in [2, 4, 7, 19]:
        qc = get_qc(f"{n_qubits}q-qvm", client_configuration=client_configuration)
        for q1, q2 in itertools.permutations(range(n_qubits), r=2):
            assert (q1, q2) in qc.qubit_topology().edges
        assert qc.name == f"{n_qubits}q-qvm"


def test_qc_noisy(client_configuration: QCSClientConfiguration):
    qc = get_qc("5q", as_qvm=True, noisy=True, client_configuration=client_configuration)
    assert isinstance(qc, QuantumComputer)


def test_qc_compile(dummy_compiler: DummyCompiler, client_configuration: QCSClientConfiguration):
    qc = get_qc("5q", as_qvm=True, noisy=True, client_configuration=client_configuration)
    qc.compiler = dummy_compiler
    prog = Program()
    prog += H(0)
    assert qc.compile(prog) == prog


def test_qc_error(client_configuration: QCSClientConfiguration):
    # QVM is not a QPU
    with pytest.raises(ValueError):
        get_qc("9q-square-noisy-qvm", as_qvm=False, client_configuration=client_configuration)

    with pytest.raises(ValueError):
        get_qc("5q", as_qvm=False, client_configuration=client_configuration)


@pytest.mark.parametrize("param", [1, np.pi, [np.pi], np.array([np.pi])])
def test_run_with_parameters(client_configuration: QCSClientConfiguration, param):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client_configuration=client_configuration),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    executable = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(1000)

    executable.write_memory(region_name="theta", value=param)
    bitstrings = qc.run(executable).readout_data.get('ro')

    assert bitstrings.shape == (1000, 1)
    assert all([bit == 1 for bit in bitstrings])


@pytest.mark.parametrize("param", [1j, "not_a_number", ["not_a_number"]])
def test_run_with_bad_parameters(client_configuration: QCSClientConfiguration, param):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client_configuration=client_configuration),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    executable = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(1000)

    with pytest.raises(TypeError, match="Parameter must be numeric or an iterable of numeric values"):
        executable.write_memory(region_name="theta", value=param)


def test_reset(client_configuration: QCSClientConfiguration):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client_configuration=client_configuration),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client_configuration=client_configuration),
    )
    p = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(10)
    p.write_memory(region_name="theta", value=np.pi)
    result = qc.qam.run(p)

    aref = ParameterAref(name="theta", index=0)
    assert p._memory.values[aref] == np.pi
    assert result.readout_data["ro"].shape == (10, 1)
    assert all([bit == 1 for bit in result.readout_data["ro"]])


def test_get_qvm_with_topology(client_configuration: QCSClientConfiguration):
    topo = nx.from_edgelist([(5, 6), (6, 7), (10, 11)])
    # Note to developers: perhaps make `get_qvm_with_topology` public in the future
    qc = _get_qvm_with_topology(
        name="test-qvm",
        topology=topo,
        noisy=False,
        qvm_type="qvm",
        compiler_timeout=5.0,
        execution_timeout=5.0,
        client_configuration=client_configuration,
    )
    assert len(qc.qubits()) == 5
    assert min(qc.qubits()) == 5


def test_get_qvm_with_topology_2(client_configuration: QCSClientConfiguration):
    topo = nx.from_edgelist([(5, 6), (6, 7)])
    qc = _get_qvm_with_topology(
        name="test-qvm",
        topology=topo,
        noisy=False,
        qvm_type="qvm",
        compiler_timeout=5.0,
        execution_timeout=5.0,
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
    ).readout_data.get('ro')
    assert results.shape == (5, 3)
    assert all(r[0] == 1 for r in results)


def test_parse_mix_qvm_and_noisy_flag():
    # https://github.com/rigetti/pyquil/issues/764
    name, qvm_type, noisy = _parse_name("1q-qvm", as_qvm=None, noisy=True)
    assert noisy


def test_noisy(client_configuration: QCSClientConfiguration):
    # https://github.com/rigetti/pyquil/issues/764
    p = Program(
        Declare("ro", "BIT", 1),
        X(0),
        MEASURE(0, ("ro", 0)),
    ).wrap_in_numshots_loop(10000)
    qc = get_qc("1q-qvm", noisy=True, client_configuration=client_configuration)
    result = qc.run(qc.compile(p)).readout_data.get('ro')
    assert result.mean() < 1.0


def test_orthogonal_array():
    def bit_array_to_int(bit_array):
        output = 0
        for bit in bit_array:
            output = (output << 1) | bit
        return output

    def check_random_columns(oa, strength):
        num_q = oa.shape[1]
        num_cols = min(num_q, strength)
        column_idxs = random.sample(range(num_q), num_cols)
        occurences = {entry: 0 for entry in range(2 ** num_cols)}
        for row in oa[:, column_idxs]:
            occurences[bit_array_to_int(row)] += 1
        assert all([count == occurences[0] for count in occurences.values()])

    for strength in [0, 1, 2, 3]:
        for num_q in range(1, 64):
            oa = _construct_orthogonal_array(num_q, strength=strength)
            for _ in range(10):
                check_random_columns(oa, strength)


def test_qc_expectation(client_configuration: QCSClientConfiguration, dummy_compiler: DummyCompiler):
    qc = QuantumComputer(name="testy!", qam=QVM(client_configuration=client_configuration), compiler=dummy_compiler)

    # bell state program
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10)

    # XX, YY, ZZ experiment
    sx = ExperimentSetting(in_state=_pauli_to_product_state(sZ(0) * sZ(1)), out_operator=sX(0) * sX(1))
    sy = ExperimentSetting(in_state=_pauli_to_product_state(sZ(0) * sZ(1)), out_operator=sY(0) * sY(1))
    sz = ExperimentSetting(in_state=_pauli_to_product_state(sZ(0) * sZ(1)), out_operator=sZ(0) * sZ(1))

    e = Experiment(settings=[sx, sy, sz], program=p)

    results = qc.run_experiment(e)

    # XX expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[0].expectation, 1)
    assert np.isclose(results[0].std_err, 0)
    assert results[0].total_counts == 40

    # YY expectation value for bell state |00> + |11> is -1
    assert np.isclose(results[1].expectation, -1)
    assert np.isclose(results[1].std_err, 0)
    assert results[1].total_counts == 40

    # ZZ expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[2].expectation, 1)
    assert np.isclose(results[2].std_err, 0)
    assert results[2].total_counts == 40


def test_qc_expectation_larger_lattice(client_configuration: QCSClientConfiguration, dummy_compiler: DummyCompiler):
    qc = QuantumComputer(name="testy!", qam=QVM(client_configuration=client_configuration), compiler=dummy_compiler)

    q0 = 2
    q1 = 3

    # bell state program
    p = Program()
    p += RESET()
    p += H(q0)
    p += CNOT(q0, q1)
    p.wrap_in_numshots_loop(10)

    # XX, YY, ZZ experiment
    sx = ExperimentSetting(in_state=_pauli_to_product_state(sZ(q0) * sZ(q1)), out_operator=sX(q0) * sX(q1))
    sy = ExperimentSetting(in_state=_pauli_to_product_state(sZ(q0) * sZ(q1)), out_operator=sY(q0) * sY(q1))
    sz = ExperimentSetting(in_state=_pauli_to_product_state(sZ(q0) * sZ(q1)), out_operator=sZ(q0) * sZ(q1))

    e = Experiment(settings=[sx, sy, sz], program=p)

    results = qc.run_experiment(e)

    # XX expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[0].expectation, 1)
    assert np.isclose(results[0].std_err, 0)
    assert results[0].total_counts == 40

    # YY expectation value for bell state |00> + |11> is -1
    assert np.isclose(results[1].expectation, -1)
    assert np.isclose(results[1].std_err, 0)
    assert results[1].total_counts == 40

    # ZZ expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[2].expectation, 1)
    assert np.isclose(results[2].std_err, 0)
    assert results[2].total_counts == 40


def asymmetric_ro_model(qubits: list, p00: float = 0.95, p11: float = 0.90) -> NoiseModel:
    aprobs = np.array([[p00, 1 - p00], [1 - p11, p11]])
    aprobs = {q: aprobs for q in qubits}
    return NoiseModel([], aprobs)


def test_qc_calibration_1q(client_configuration: QCSClientConfiguration):
    # noise model with 95% symmetrized readout fidelity per qubit
    noise_model = asymmetric_ro_model([0], 0.945, 0.955)
    qc = get_qc("1q-qvm", client_configuration=client_configuration)
    qc.qam.noise_model = noise_model

    # bell state program (doesn't matter)
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10000)

    # Z experiment
    sz = ExperimentSetting(in_state=_pauli_to_product_state(sZ(0)), out_operator=sZ(0))
    e = Experiment(settings=[sz], program=p)

    results = qc.calibrate(e)

    # Z expectation value should just be 1 - 2 * readout_error
    np.isclose(results[0].expectation, 0.9, atol=0.01)
    assert results[0].total_counts == 20000


def test_qc_calibration_2q(client_configuration: QCSClientConfiguration):
    # noise model with 95% symmetrized readout fidelity per qubit
    noise_model = asymmetric_ro_model([0, 1], 0.945, 0.955)
    qc = get_qc("2q-qvm", client_configuration=client_configuration)
    qc.qam.noise_model = noise_model

    # bell state program (doesn't matter)
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10000)

    # ZZ experiment
    sz = ExperimentSetting(in_state=_pauli_to_product_state(sZ(0) * sZ(1)), out_operator=sZ(0) * sZ(1))
    e = Experiment(settings=[sz], program=p)

    results = qc.calibrate(e)

    # ZZ expectation should just be (1 - 2 * readout_error_q0) * (1 - 2 * readout_error_q1)
    np.isclose(results[0].expectation, 0.81, atol=0.01)
    assert results[0].total_counts == 40000


def test_qc_joint_expectation(client_configuration: QCSClientConfiguration, dummy_compiler: DummyCompiler):
    qc = QuantumComputer(name="testy!", qam=QVM(client_configuration=client_configuration), compiler=dummy_compiler)

    # |01> state program
    p = Program()
    p += RESET()
    p += X(0)
    p.wrap_in_numshots_loop(10)

    # ZZ experiment
    sz = ExperimentSetting(
        in_state=_pauli_to_product_state(sZ(0) * sZ(1)), out_operator=sZ(0) * sZ(1), additional_expectations=[[0], [1]]
    )
    e = Experiment(settings=[sz], program=p)

    results = qc.run_experiment(e)

    # ZZ expectation value for state |01> is -1
    assert np.isclose(results[0].expectation, -1)
    assert np.isclose(results[0].std_err, 0)
    assert results[0].total_counts == 40
    # Z0 expectation value for state |01> is -1
    assert np.isclose(results[0].additional_results[0].expectation, -1)
    assert results[0].additional_results[1].total_counts == 40
    # Z1 expectation value for state |01> is 1
    assert np.isclose(results[0].additional_results[1].expectation, 1)
    assert results[0].additional_results[1].total_counts == 40


def test_get_qc_noisy_qpu_error(client_configuration: QCSClientConfiguration, dummy_compiler: DummyCompiler):
    expected_message = (
        "pyQuil currently does not support initializing a noisy QuantumComputer "
        "based on a QCSQuantumProcessor. Change noisy to False or specify the name of a QVM."
    )
    with pytest.raises(ValueError, match=expected_message):
        get_qc("Aspen-8", noisy=True)


def test_qc_joint_calibration(client_configuration: QCSClientConfiguration):
    # noise model with 95% symmetrized readout fidelity per qubit
    noise_model = asymmetric_ro_model([0, 1], 0.945, 0.955)
    qc = get_qc("2q-qvm", client_configuration=client_configuration)
    qc.qam.noise_model = noise_model

    # |01> state program
    p = Program()
    p += RESET()
    p += X(0)
    p.wrap_in_numshots_loop(10000)

    # ZZ experiment
    sz = ExperimentSetting(
        in_state=_pauli_to_product_state(sZ(0) * sZ(1)), out_operator=sZ(0) * sZ(1), additional_expectations=[[0], [1]]
    )
    e = Experiment(settings=[sz], program=p)

    results = qc.run_experiment(e)

    # ZZ expectation value for state |01> with 95% RO fid on both qubits is about -0.81
    assert np.isclose(results[0].expectation, -0.81, atol=0.01)
    assert results[0].total_counts == 40000
    # Z0 expectation value for state |01> with 95% RO fid on both qubits is about -0.9
    assert np.isclose(results[0].additional_results[0].expectation, -0.9, atol=0.01)
    assert results[0].additional_results[1].total_counts == 40000
    # Z1 expectation value for state |01> with 95% RO fid on both qubits is about 0.9
    assert np.isclose(results[0].additional_results[1].expectation, 0.9, atol=0.01)
    assert results[0].additional_results[1].total_counts == 40000


def test_qc_expectation_on_qvm(client_configuration: QCSClientConfiguration, dummy_compiler: DummyCompiler):
    # regression test for https://github.com/rigetti/forest-tutorials/issues/2
    qc = QuantumComputer(name="testy!", qam=QVM(client_configuration=client_configuration), compiler=dummy_compiler)

    p = Program()
    theta = p.declare("theta", "REAL")
    p += RESET()
    p += RY(theta, 0)
    p.wrap_in_numshots_loop(10000)

    sx = ExperimentSetting(in_state=_pauli_to_product_state(sZ(0)), out_operator=sX(0))
    e = Experiment(settings=[sx], program=p)

    thetas = [-np.pi / 2, 0.0, np.pi / 2]
    results = []

    # Verify that multiple calls to qc.experiment with the same experiment backed by a QVM that
    # requires_exectutable does not raise an exception.
    for theta in thetas:
        results.append(qc.run_experiment(e, memory_map={"theta": [theta]}))

    assert np.isclose(results[0][0].expectation, -1.0, atol=0.01)
    assert np.isclose(results[0][0].std_err, 0)
    assert results[0][0].total_counts == 20000

    # bounds on atol and std_err here are a little loose to try and avoid test flakiness.
    assert np.isclose(results[1][0].expectation, 0.0, atol=0.1)
    assert results[1][0].std_err < 0.01
    assert results[1][0].total_counts == 20000

    assert np.isclose(results[2][0].expectation, 1.0, atol=0.01)
    assert np.isclose(results[2][0].std_err, 0)
    assert results[2][0].total_counts == 20000
