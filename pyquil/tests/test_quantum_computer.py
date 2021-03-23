import itertools
import random

import networkx as nx
import numpy as np
import pytest
from rpcq.messages import ParameterAref

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.api import QVM, QuantumComputer, Client
from pyquil.api._quantum_computer import (
    _symmetrization,
    _flip_array_to_prog,
    _construct_orthogonal_array,
    _construct_strength_two_orthogonal_array,
    _construct_strength_three_orthogonal_array,
    _parse_name,
    _get_qvm_with_topology,
    _measure_bitstrings,
    _consolidate_symmetrization_outputs,
    _check_min_num_trials_for_symmetrized_readout,
)
from pyquil.quantum_processor import NxQuantumProcessor
from pyquil.gates import CNOT, H, I, MEASURE, RX, X
from pyquil.noise import decoherence_noise_with_asymmetric_ro
from pyquil.pyqvm import PyQVM
from pyquil.quilbase import Declare, MemoryReference
from pyquil.tests.utils import DummyCompiler


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


def test_measure_bitstrings(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(2))
    dummy_compiler = DummyCompiler(quantum_processor=quantum_processor, client=client)
    qc_pyqvm = QuantumComputer(name="testy!", qam=PyQVM(n_qubits=2), compiler=dummy_compiler)
    qc_forest = QuantumComputer(
        name="testy!",
        qam=QVM(client=client, gate_noise=(0.00, 0.00, 0.00)),
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


def test_quantum_processor_stuff(client: Client):
    topo = nx.from_edgelist([(0, 4), (0, 99)])
    qc = QuantumComputer(
        name="testy!",
        qam=None,  # not necessary for this test
        compiler=DummyCompiler(quantum_processor=NxQuantumProcessor(topo, gates_2q=["CPHASE"]), client=client),
    )
    assert nx.is_isomorphic(qc.qubit_topology(), topo)

    isa = qc.to_compiler_isa()

    assert isa.edges["0-4"].gates[0].operator == "CPHASE"
    assert isa.edges["0-4"].ids == [0, 4]


# We sometimes narrowly miss the np.mean(parity) < 0.15 assertion, below. Alternatively, that upper
# bound could be relaxed.
@pytest.mark.flaky(reruns=1)
def test_run(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client=client, gate_noise=(0.01, 0.01, 0.01)),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client=client),
    )
    bitstrings = qc.run(
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

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


def test_run_pyqvm_noiseless(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!", qam=PyQVM(n_qubits=3), compiler=DummyCompiler(quantum_processor=quantum_processor, client=client)
    )
    prog = Program(H(0), CNOT(0, 1), CNOT(1, 2))
    ro = prog.declare("ro", "BIT", 3)
    for q in range(3):
        prog += MEASURE(q, ro[q])
    bitstrings = qc.run(prog.wrap_in_numshots_loop(1000))

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert np.mean(parity) == 0


def test_run_pyqvm_noisy(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=PyQVM(n_qubits=3, post_gate_noise_probabilities={"relaxation": 0.01}),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client=client),
    )
    prog = Program(H(0), CNOT(0, 1), CNOT(1, 2))
    ro = prog.declare("ro", "BIT", 3)
    for q in range(3):
        prog += MEASURE(q, ro[q])
    bitstrings = qc.run(prog.wrap_in_numshots_loop(1000))

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


def test_readout_symmetrization(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    noise_model = decoherence_noise_with_asymmetric_ro(quantum_processor.to_compiler_isa())
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client=client, noise_model=noise_model),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client=client),
    )

    prog = Program(I(0), X(1), MEASURE(0, MemoryReference("ro", 0)), MEASURE(1, MemoryReference("ro", 1)))
    prog.wrap_in_numshots_loop(1000)

    bs1 = qc.run(prog)
    avg0_us = np.mean(bs1[:, 0])
    avg1_us = 1 - np.mean(bs1[:, 1])
    diff_us = avg1_us - avg0_us
    assert diff_us > 0.03

    bs2 = qc.run_symmetrized_readout(prog, 1000)
    avg0_s = np.mean(bs2[:, 0])
    avg1_s = 1 - np.mean(bs2[:, 1])
    diff_s = avg1_s - avg0_s
    assert diff_s < 0.05


@pytest.mark.slow
def test_run_symmetrized_readout_error(client: Client):
    # This test checks if the function runs for any possible input on a small number of qubits.
    # Locally this test was run on all 8 qubits, but it was slow.
    qc = get_qc("8q-qvm", client=client)
    sym_type_vec = [-1, 0, 1, 2, 3]
    prog_vec = [Program(I(x) for x in range(0, 3))[0:n] for n in range(0, 4)]
    trials_vec = list(range(0, 5))
    for prog, trials, sym_type in itertools.product(prog_vec, trials_vec, sym_type_vec):
        print(qc.run_symmetrized_readout(prog, trials, sym_type))


def test_list_qc():
    qc_names = list_quantum_computers(qpus=False)
    # TODO: update with deployed qpus
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


def test_qc(client):
    qc = get_qc("9q-square-noisy-qvm", client=client)
    assert isinstance(qc, QuantumComputer)
    assert isinstance(qc.qam, QVM)
    assert qc.qam.noise_model is not None
    assert qc.qubit_topology().number_of_nodes() == 9
    assert qc.qubit_topology().degree[0] == 2
    assert qc.qubit_topology().degree[4] == 4
    assert str(qc) == "9q-square-noisy-qvm"


def test_qc_run(client):
    qc = get_qc("9q-square-noisy-qvm", client=client)
    bs = qc.run_and_measure(Program(X(0)), trials=3)
    assert len(bs) == 9
    for _, bits in bs.items():
        assert bits.shape == (3,)


def test_nq_qvm_qc(client):
    for n_qubits in [2, 4, 7, 19]:
        qc = get_qc(f"{n_qubits}q-qvm", client=client)
        for q1, q2 in itertools.permutations(range(n_qubits), r=2):
            assert (q1, q2) in qc.qubit_topology().edges
        assert qc.name == f"{n_qubits}q-qvm"


def test_qc_noisy(client):
    qc = get_qc("5q", as_qvm=True, noisy=True, client=client)
    assert isinstance(qc, QuantumComputer)


def test_qc_compile(dummy_compiler: DummyCompiler, client):
    qc = get_qc("5q", as_qvm=True, noisy=True, client=client)
    qc.compiler = dummy_compiler
    prog = Program()
    prog += H(0)
    assert qc.compile(prog) == prog


def test_qc_error(client):
    # QVM is not a QPU
    with pytest.raises(ValueError):
        get_qc("9q-square-noisy-qvm", as_qvm=False, client=client)

    with pytest.raises(ValueError):
        get_qc("5q", as_qvm=False, client=client)


def test_run_and_measure(client):
    qc = get_qc("9q-square-qvm", client=client)
    prog = Program(I(8))
    trials = 11
    # note to devs: this is included as an example in the run_and_measure docstrings
    # so if you change it here ... change it there!
    bitstrings = qc.run_and_measure(prog, trials)
    bitstring_array = np.vstack([bitstrings[q] for q in qc.qubits()]).T
    assert bitstring_array.shape == (trials, len(qc.qubits()))


def test_run_and_measure_noiseless_qvm(client):
    """ Test that run_and_measure works as expected on a noiseless QVM. """
    qc = get_qc("9q-square-qvm", client=client)
    prog = Program(X(0))
    trials = 1
    bitstrings = qc.run_and_measure(prog, trials)
    bitstring_array = np.vstack([bitstrings[q] for q in qc.qubits()]).T
    # Test for appropriate shape
    assert bitstring_array.shape == (trials, len(qc.qubits()))
    # Test that X(0) flipped qubit 0.
    assert bitstring_array[0, 0] == 1
    # Test that all remaining qubits were measured and found to be in
    # state |0>.
    assert all(bitstring_array[0][1:] == np.zeros(len(qc.qubits()) - 1))


def test_run_with_parameters(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client=client),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client=client),
    )
    bitstrings = qc.run(
        executable=Program(
            Declare(name="theta", memory_type="REAL"),
            Declare(name="ro", memory_type="BIT"),
            RX(MemoryReference("theta"), 0),
            MEASURE(0, MemoryReference("ro")),
        ).wrap_in_numshots_loop(1000),
        memory_map={"theta": [np.pi]},
    )

    assert bitstrings.shape == (1000, 1)
    assert all([bit == 1 for bit in bitstrings])


def test_reset(client: Client):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    qc = QuantumComputer(
        name="testy!",
        qam=QVM(client=client),
        compiler=DummyCompiler(quantum_processor=quantum_processor, client=client),
    )
    p = Program(
        Declare(name="theta", memory_type="REAL"),
        Declare(name="ro", memory_type="BIT"),
        RX(MemoryReference("theta"), 0),
        MEASURE(0, MemoryReference("ro")),
    ).wrap_in_numshots_loop(1000)
    qc.run(executable=p, memory_map={"theta": [np.pi]})

    aref = ParameterAref(name="theta", index=0)
    assert qc.qam._variables_shim[aref] == np.pi
    assert qc.qam.executable == p
    assert qc.qam._memory_results["ro"].shape == (1000, 1)
    assert all([bit == 1 for bit in qc.qam._memory_results["ro"]])
    assert qc.qam.status == "done"

    qc.reset()

    assert qc.qam._variables_shim == {}
    assert qc.qam.executable is None
    assert qc.qam._memory_results["ro"] is None
    assert qc.qam.status == "connected"


def test_get_qvm_with_topology(client: Client):
    topo = nx.from_edgelist([(5, 6), (6, 7), (10, 11)])
    # Note to developers: perhaps make `get_qvm_with_topology` public in the future
    qc = _get_qvm_with_topology(name="test-qvm", topology=topo, client=client)
    assert len(qc.qubits()) == 5
    assert min(qc.qubits()) == 5


def test_get_qvm_with_topology_2(client: Client):
    topo = nx.from_edgelist([(5, 6), (6, 7)])
    qc = _get_qvm_with_topology(name="test-qvm", topology=topo, client=client)
    results = qc.run_and_measure(Program(X(5)), trials=5)
    assert sorted(results.keys()) == [5, 6, 7]
    assert all(x == 1 for x in results[5])


def test_parse_mix_qvm_and_noisy_flag():
    # https://github.com/rigetti/pyquil/issues/764
    name, qvm_type, noisy = _parse_name("1q-qvm", as_qvm=None, noisy=True)
    assert noisy


def test_noisy(client: Client):
    # https://github.com/rigetti/pyquil/issues/764
    p = Program(X(0))
    qc = get_qc("1q-qvm", noisy=True, client=client)
    result = qc.run_and_measure(p, trials=10000)
    assert result[0].mean() < 1.0


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
