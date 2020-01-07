import networkx as nx
import numpy as np

from pyquil import Program
from pyquil.api import QVM, QuantumComputer, get_qc
from pyquil.device import NxDevice
from pyquil.experiment import ExperimentSetting, TomographyExperiment
from pyquil.gates import CNOT, H, RESET, X
from pyquil.noise import NoiseModel
from pyquil.paulis import sX, sY, sZ
from pyquil.tests.utils import DummyCompiler


def test_qc_expectation(forest):
    device = NxDevice(nx.complete_graph(2))
    qc = QuantumComputer(
        name="testy!", qam=QVM(connection=forest), device=device, compiler=DummyCompiler()
    )

    # bell state program
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10)

    # XX, YY, ZZ experiment
    sx = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sX(0) * sX(1))
    sy = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sY(0) * sY(1))
    sz = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1))

    e = TomographyExperiment(settings=[sx, sy, sz], program=p)

    results = qc.experiment(e)

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


def test_qc_expectation_larger_lattice(forest):
    device = NxDevice(nx.complete_graph(4))
    qc = QuantumComputer(
        name="testy!", qam=QVM(connection=forest), device=device, compiler=DummyCompiler()
    )

    q0 = 2
    q1 = 3

    # bell state program
    p = Program()
    p += RESET()
    p += H(q0)
    p += CNOT(q0, q1)
    p.wrap_in_numshots_loop(10)

    # XX, YY, ZZ experiment
    sx = ExperimentSetting(in_state=sZ(q0) * sZ(q1), out_operator=sX(q0) * sX(q1))
    sy = ExperimentSetting(in_state=sZ(q0) * sZ(q1), out_operator=sY(q0) * sY(q1))
    sz = ExperimentSetting(in_state=sZ(q0) * sZ(q1), out_operator=sZ(q0) * sZ(q1))

    e = TomographyExperiment(settings=[sx, sy, sz], program=p)

    results = qc.experiment(e)

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


def test_qc_calibration_1q(forest):
    # noise model with 95% symmetrized readout fidelity per qubit
    noise_model = asymmetric_ro_model([0], 0.945, 0.955)
    qc = get_qc("1q-qvm")
    qc.qam.noise_model = noise_model

    # bell state program (doesn't matter)
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10000)

    # Z experiment
    sz = ExperimentSetting(in_state=sZ(0), out_operator=sZ(0))
    e = TomographyExperiment(settings=[sz], program=p)

    results = qc.calibrate(e)

    # Z expectation value should just be 1 - 2 * readout_error
    np.isclose(results[0].expectation, 0.9, atol=0.01)
    assert results[0].total_counts == 20000


def test_qc_calibration_2q(forest):
    # noise model with 95% symmetrized readout fidelity per qubit
    noise_model = asymmetric_ro_model([0, 1], 0.945, 0.955)
    qc = get_qc("2q-qvm")
    qc.qam.noise_model = noise_model

    # bell state program (doesn't matter)
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10000)

    # ZZ experiment
    sz = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1))
    e = TomographyExperiment(settings=[sz], program=p)

    results = qc.calibrate(e)

    # ZZ expectation should just be (1 - 2 * readout_error_q0) * (1 - 2 * readout_error_q1)
    np.isclose(results[0].expectation, 0.81, atol=0.01)
    assert results[0].total_counts == 40000


def test_qc_joint_expectation(forest):
    device = NxDevice(nx.complete_graph(2))
    qc = QuantumComputer(
        name="testy!", qam=QVM(connection=forest), device=device, compiler=DummyCompiler()
    )

    # |01> state program
    p = Program()
    p += RESET()
    p += X(0)
    p.wrap_in_numshots_loop(10)

    # ZZ experiment
    sz = ExperimentSetting(
        in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1), additional_expectations=[[0], [1]]
    )
    e = TomographyExperiment(settings=[sz], program=p)

    results = qc.experiment(e)

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


def test_qc_joint_calibration(forest):
    # noise model with 95% symmetrized readout fidelity per qubit
    noise_model = asymmetric_ro_model([0, 1], 0.945, 0.955)
    qc = get_qc("2q-qvm")
    qc.qam.noise_model = noise_model

    # |01> state program
    p = Program()
    p += RESET()
    p += X(0)
    p.wrap_in_numshots_loop(10000)

    # ZZ experiment
    sz = ExperimentSetting(
        in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1), additional_expectations=[[0], [1]]
    )
    e = TomographyExperiment(settings=[sz], program=p)

    results = qc.experiment(e)

    # ZZ expectation value for state |01> with 95% RO fid on both qubits is about -0.81
    assert np.isclose(results[0].expectation, -0.81, atol=0.01)
    assert results[0].total_counts == 40000
    # Z0 expectation value for state |01> with 95% RO fid on both qubits is about -0.9
    assert np.isclose(results[0].additional_results[0].expectation, -0.9, atol=0.01)
    assert results[0].additional_results[1].total_counts == 40000
    # Z1 expectation value for state |01> with 95% RO fid on both qubits is about 0.9
    assert np.isclose(results[0].additional_results[1].expectation, 0.9, atol=0.01)
    assert results[0].additional_results[1].total_counts == 40000
