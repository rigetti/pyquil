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


def test_adding_does_not_mutate():
    # https://github.com/rigetti/pyquil/issues/1476
    p1 = Program(
        """
DEFCAL RX(pi/2) 32:
    FENCE 32
    NONBLOCKING PULSE 32 "rf" drag_gaussian(duration: 3.2e-08, fwhm: 8e-09, t0: 1.6e-08, anh: -190000000.0, alpha: -1.8848698349348032, scale: 0.30631340170943533, phase: 0.0, detuning: 1622438.2425563578)
    FENCE 32

RX(pi/2) 32
"""
    )
    original_p1 = p1.copy()
    p2 = Program(
        """
DEFCAL RX(pi/2) 33:
    FENCE 33
    NONBLOCKING PULSE 33 "rf" drag_gaussian(duration: 2e-08, fwhm: 5e-09, t0: 1e-08, anh: -190000000.0, alpha: -0.9473497322033984, scale: 0.25680107985232403, phase: 0.0, detuning: 1322130.5458282642)
    FENCE 33

RX(pi/2) 33
"""
    )
    p_all = p1 + p2
    assert p1 == original_p1
    assert p1.calibrations != p_all.calibrations
