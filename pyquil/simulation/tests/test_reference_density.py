import networkx as nx
import numpy as np
import pytest

import pyquil.simulation.matrices as qmats
from pyquil import Program
from pyquil.api import QuantumComputer, Client
from pyquil.quantum_processor import NxQuantumProcessor
from pyquil.experiment import ExperimentSetting, Experiment, zeros_state
from pyquil.gates import CNOT, H, I, MEASURE, PHASE, RX, RY, RZ, X
from pyquil.operator_estimation import measure_observables
from pyquil.paulis import sI, sX, sY, sZ
from pyquil.pyqvm import PyQVM
from pyquil.simulation._reference import ReferenceDensitySimulator, _is_valid_quantum_state
from pyquil.simulation.tools import lifted_gate_matrix
from pyquil.tests.utils import DummyCompiler


def test_qaoa_density():
    wf_true = [
        0.00167784 + 1.00210180e-05 * 1j,
        0.50000000 - 4.99997185e-01 * 1j,
        0.50000000 - 4.99997185e-01 * 1j,
        0.00167784 + 1.00210180e-05 * 1j,
    ]
    wf_true = np.reshape(np.array(wf_true), (4, 1))
    rho_true = np.dot(wf_true, np.conj(wf_true).T)
    prog = Program()
    prog.inst(
        [
            RY(np.pi / 2, 0),
            RX(np.pi, 0),
            RY(np.pi / 2, 1),
            RX(np.pi, 1),
            CNOT(0, 1),
            RX(-np.pi / 2, 1),
            RY(4.71572463191, 1),
            RX(np.pi / 2, 1),
            CNOT(0, 1),
            RX(-2 * 2.74973750579, 0),
            RX(-2 * 2.74973750579, 1),
        ]
    )

    qam = PyQVM(n_qubits=2, quantum_simulator_type=ReferenceDensitySimulator).execute(prog)
    rho = qam.wf_simulator.density
    np.testing.assert_allclose(rho_true, rho, atol=1e-8)


def test_larger_qaoa_density():
    prog = Program(
        H(0),
        H(1),
        H(2),
        H(3),
        X(0),
        PHASE(0.3928244130249029, 0),
        X(0),
        PHASE(0.3928244130249029, 0),
        CNOT(0, 1),
        RZ(0.78564882604980579, 1),
        CNOT(0, 1),
        X(0),
        PHASE(0.3928244130249029, 0),
        X(0),
        PHASE(0.3928244130249029, 0),
        CNOT(0, 3),
        RZ(0.78564882604980579, 3),
        CNOT(0, 3),
        X(0),
        PHASE(0.3928244130249029, 0),
        X(0),
        PHASE(0.3928244130249029, 0),
        CNOT(1, 2),
        RZ(0.78564882604980579, 2),
        CNOT(1, 2),
        X(0),
        PHASE(0.3928244130249029, 0),
        X(0),
        PHASE(0.3928244130249029, 0),
        CNOT(2, 3),
        RZ(0.78564882604980579, 3),
        CNOT(2, 3),
        H(0),
        RZ(-0.77868204192240842, 0),
        H(0),
        H(1),
        RZ(-0.77868204192240842, 1),
        H(1),
        H(2),
        RZ(-0.77868204192240842, 2),
        H(2),
        H(3),
        RZ(-0.77868204192240842, 3),
        H(3),
    )

    qam = PyQVM(n_qubits=4, quantum_simulator_type=ReferenceDensitySimulator).execute(prog)
    rho_test = qam.wf_simulator.density
    wf_true = np.array(
        [
            8.43771693e-05 - 0.1233845 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -2.50040954e-01 + 0.12661547 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -4.99915497e-01 - 0.12363516 * 1j,
            -2.50040954e-01 + 0.12661547 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -2.50040954e-01 + 0.12661547 * 1j,
            -4.99915497e-01 - 0.12363516 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -2.50040954e-01 + 0.12661547 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            -1.24927731e-01 + 0.00329533 * 1j,
            8.43771693e-05 - 0.1233845 * 1j,
        ]
    )

    wf_true = np.reshape(wf_true, (2 ** 4, 1))
    rho_true = np.dot(wf_true, np.conj(wf_true).T)
    np.testing.assert_allclose(rho_true, rho_test, atol=1e-8)


def _random_1q_density():
    state = np.random.random(2) + 1j * np.random.random()
    normalization = np.conj(state).T.dot(state)
    state /= np.sqrt(normalization)
    state = state.reshape((-1, 1))

    rho = state.dot(np.conj(state).T)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, np.conj(rho).T)
    return rho


def test_kraus_application_bitflip():
    p = 0.372
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"bit_flip": p},
    )
    initial_density = _random_1q_density()
    qam.wf_simulator.density = initial_density
    qam.execute(Program(I(0)))
    final_density = (1 - p) * initial_density + p * qmats.X.dot(initial_density).dot(qmats.X)
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


def test_kraus_application_phaseflip():
    p = 0.372
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"phase_flip": p},
    )
    initial_density = _random_1q_density()
    qam.wf_simulator.density = initial_density
    qam.execute(Program(I(0)))

    final_density = (1 - p) * initial_density + p * qmats.Z.dot(initial_density).dot(qmats.Z)
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


def test_kraus_application_bitphaseflip():
    p = 0.372
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"bitphase_flip": p},
    )
    initial_density = _random_1q_density()
    qam.wf_simulator.density = initial_density
    qam.execute(Program(I(0)))

    final_density = (1 - p) * initial_density + p * qmats.Y.dot(initial_density).dot(qmats.Y)
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


def test_kraus_application_relaxation():
    p = 0.372
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"relaxation": p},
    )
    rho = _random_1q_density()
    qam.wf_simulator.density = rho
    qam.execute(Program(I(0)))

    final_density = np.array(
        [
            [rho[0, 0] + rho[1, 1] * p, np.sqrt(1 - p) * rho[0, 1]],
            [np.sqrt(1 - p) * rho[1, 0], (1 - p) * rho[1, 1]],
        ]
    )
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


def test_kraus_application_dephasing():
    p = 0.372
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"dephasing": p},
    )
    rho = _random_1q_density()
    qam.wf_simulator.density = rho
    qam.execute(Program(I(0)))
    final_density = np.array([[rho[0, 0], (1 - p) * rho[0, 1]], [(1 - p) * rho[1, 0], rho[1, 1]]])
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


def test_kraus_application_depolarizing():
    p = 0.372
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"depolarizing": p},
    )
    rho = _random_1q_density()
    qam.wf_simulator.density = rho
    qam.execute(Program(I(0)))

    final_density = (1 - p) * rho + (p / 3) * (
        qmats.X.dot(rho).dot(qmats.X) + qmats.Y.dot(rho).dot(qmats.Y) + qmats.Z.dot(rho).dot(qmats.Z)
    )
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


def test_kraus_compound_T1T2_application():
    p1 = 0.372
    p2 = 0.45
    qam = PyQVM(
        n_qubits=1,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"relaxation": p1, "dephasing": p2},
    )
    rho = _random_1q_density()
    qam.wf_simulator.density = rho
    qam.execute(Program(I(0)))

    final_density = np.array(
        [
            [rho[0, 0] + rho[1, 1] * p1, (1 - p2) * np.sqrt(1 - p1) * rho[0, 1]],
            [(1 - p2) * np.sqrt(1 - p1) * rho[1, 0], (1 - p1) * rho[1, 1]],
        ]
    )
    np.testing.assert_allclose(final_density, qam.wf_simulator.density)


@pytest.mark.xfail(reason="We don't support different noise parameters for 2q vs 1q gates!")
def test_multiqubit_decay_bellstate():
    program = Program(RY(np.pi / 3, 0), CNOT(0, 1))

    # commence manually dotting the above program
    initial_density = np.zeros((4, 4), dtype=complex)
    initial_density[0, 0] = 1.0

    gate_time_1q = 50e-9
    T1 = 30e-6
    T2 = 15e-6
    p1 = 1 - np.exp(-gate_time_1q / T1)
    p2 = 1 - np.exp(-gate_time_1q / T2)

    # RY
    gate_1 = np.kron(np.eye(2), qmats.RY(np.pi / 3))
    state = gate_1.dot(initial_density).dot(np.conj(gate_1).T)

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in qmats.relaxation_operators(p1):
            operator = lifted_gate_matrix(matrix=kop, qubit_inds=[ii], n_qubits=2)
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in qmats.dephasing_operators(p2):
            operator = lifted_gate_matrix(matrix=kop, qubit_inds=[ii], n_qubits=2)
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    # CNOT
    # TODO: different 1q, 2q noise probabilities
    cnot_01 = np.kron(qmats.I, qmats.P0) + np.kron(qmats.X, qmats.P1)
    state = cnot_01.dot(state).dot(cnot_01.T)
    gate_time_2q = 150e-9
    p1 = 1 - np.exp(-gate_time_2q / T1)
    p2 = 1 - np.exp(-gate_time_2q / T2)

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in qmats.relaxation_operators(p1):
            operator = lifted_gate_matrix(matrix=kop, qubit_inds=[ii], n_qubits=2)
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    for ii in range(2):
        new_density = np.zeros_like(state)
        for kop in qmats.dephasing_operators(p2):
            operator = lifted_gate_matrix(matrix=kop, qubit_inds=[ii], n_qubits=2)
            new_density += operator.dot(state).dot(np.conj(operator).T)
        state = new_density

    qam = PyQVM(
        n_qubits=2,
        quantum_simulator_type=ReferenceDensitySimulator,
        post_gate_noise_probabilities={"relaxation": p1, "dephasing": p2},
    )
    qam.execute(program)

    assert np.allclose(qam.wf_simulator.density, state)


@pytest.mark.slow
def test_for_negative_probabilities(client: Client):
    # trivial program to do state tomography on
    prog = Program(I(0))

    # make an Experiment
    expt_settings = [ExperimentSetting(zeros_state([0]), pt) for pt in [sI(0), sX(0), sY(0), sZ(0)]]
    experiment_1q = Experiment(settings=expt_settings, program=prog)

    # make a quantum computer object
    device = NxQuantumProcessor(nx.complete_graph(1))
    qc_density = QuantumComputer(
        name="testy!",
        qam=PyQVM(n_qubits=1, quantum_simulator_type=ReferenceDensitySimulator),
        compiler=DummyCompiler(quantum_processor=device, client=client),
    )

    # initialize with a pure state
    initial_density = np.array([[1.0, 0.0], [0.0, 0.0]])
    qc_density.qam.wf_simulator.density = initial_density

    try:
        list(measure_observables(qc=qc_density, tomo_experiment=experiment_1q, n_shots=3000))
    except ValueError as e:
        # the error is from np.random.choice by way of self.rs.choice in ReferenceDensitySimulator
        assert str(e) != "probabilities are not non-negative"

    # initialize with a mixed state
    initial_density = np.array([[0.9, 0.0], [0.0, 0.1]])
    qc_density.qam.wf_simulator.density = initial_density

    try:
        list(measure_observables(qc=qc_density, tomo_experiment=experiment_1q, n_shots=3000))
    except ValueError as e:
        assert str(e) != "probabilities are not non-negative"


def test_set_initial_state(client: Client):
    # That is test the assigned state matrix in ReferenceDensitySimulator is persistent between
    # rounds of run.
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]])

    # run prog
    prog = Program(I(0))
    ro = prog.declare("ro", "BIT", 1)
    prog += MEASURE(0, ro[0])

    # make a quantum computer object
    device = NxQuantumProcessor(nx.complete_graph(1))
    qc_density = QuantumComputer(
        name="testy!",
        qam=PyQVM(n_qubits=1, quantum_simulator_type=ReferenceDensitySimulator),
        compiler=DummyCompiler(quantum_processor=device, client=client),
    )

    qc_density.qam.wf_simulator.set_initial_state(rho1).reset()

    out = [qc_density.run(prog) for _ in range(0, 4)]
    ans = [np.array([[1]]), np.array([[1]]), np.array([[1]]), np.array([[1]])]
    assert all([np.allclose(x, y) for x, y in zip(out, ans)])

    # Run and measure style
    progRAM = Program(I(0))

    results = qc_density.run_and_measure(progRAM, trials=10)
    ans = {0: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
    assert np.allclose(results[0], ans[0])

    # test reverting ReferenceDensitySimulator to the default state
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    qc_density.qam.wf_simulator.set_initial_state(rho0).reset()
    assert np.allclose(qc_density.qam.wf_simulator.density, rho0)
    assert np.allclose(qc_density.qam.wf_simulator.initial_density, rho0)


def test_is_valid_quantum_state():
    with pytest.raises(ValueError):
        # is Hermitian and PSD but not trace one
        _is_valid_quantum_state(np.array([[1, 0], [0, 1]]))
    with pytest.raises(ValueError):
        # negative eigenvalue
        _is_valid_quantum_state(np.array([[1.01, 0], [0, -0.01]]))
    with pytest.raises(ValueError):
        # imaginary eigenvalue
        _is_valid_quantum_state(np.array([[1, 0], [0, -0.0001j]]))
    with pytest.raises(ValueError):
        # not Hermitian
        _is_valid_quantum_state(np.array([[0, 1], [1, 0]]))
