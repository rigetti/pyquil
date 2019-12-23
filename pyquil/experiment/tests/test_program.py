import numpy as np

from pyquil import Program
from pyquil.experiment._program import (
    measure_qubits,
    parameterized_single_qubit_measurement_basis,
    parameterized_single_qubit_state_preparation,
    parameterized_readout_symmetrization,
)
from pyquil.gates import MEASURE, RX, RZ


def test_measure_qubits():
    p = Program()
    ro = p.declare("ro", "BIT", 2)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    assert measure_qubits([0, 1]).out() == p.out()


def test_parameterized_single_qubit_measurement_basis():
    p = Program()
    alpha = p.declare("measurement_alpha", "REAL", 2)
    beta = p.declare("measurement_beta", "REAL", 2)
    gamma = p.declare("measurement_gamma", "REAL", 2)
    for idx, q in enumerate(range(2)):
        p += RZ(alpha[idx], q)
        p += RX(np.pi / 2, q)
        p += RZ(beta[idx], q)
        p += RX(-np.pi / 2, q)
        p += RZ(gamma[idx], q)
    assert parameterized_single_qubit_measurement_basis([0, 1]).out() == p.out()


def test_parameterized_single_qubit_state_preparation():
    p = Program()
    alpha = p.declare("preparation_alpha", "REAL", 2)
    beta = p.declare("preparation_beta", "REAL", 2)
    gamma = p.declare("preparation_gamma", "REAL", 2)
    p += RZ(alpha[0], 0)
    p += RX(np.pi / 2, 0)
    p += RZ(beta[0], 0)
    p += RX(-np.pi / 2, 0)
    p += RZ(gamma[0], 0)
    p += RZ(alpha[1], 1)
    p += RX(np.pi / 2, 1)
    p += RZ(beta[1], 1)
    p += RX(-np.pi / 2, 1)
    p += RZ(gamma[1], 1)
    assert parameterized_single_qubit_state_preparation([0, 1]).out() == p.out()


def test_parameterized_readout_symmetrization():
    p = Program()
    symmetrization = p.declare("symmetrization", "REAL", 2)
    p += RX(symmetrization[0], 0)
    p += RX(symmetrization[1], 1)
    assert parameterized_readout_symmetrization([0, 1]).out() == p.out()
