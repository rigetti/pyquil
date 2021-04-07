import numpy as np

from pyquil.simulation.matrices import (
    QUANTUM_GATES,
    relaxation_operators,
    dephasing_operators,
    depolarizing_operators,
    bit_flip_operators,
    phase_flip_operators,
    bitphase_flip_operators,
)


def test_singleq():
    assert np.isclose(QUANTUM_GATES["I"], np.eye(2)).all()
    assert np.isclose(QUANTUM_GATES["X"], np.array([[0, 1], [1, 0]])).all()
    assert np.isclose(QUANTUM_GATES["Y"], np.array([[0, -1j], [1j, 0]])).all()
    assert np.isclose(QUANTUM_GATES["Z"], np.array([[1, 0], [0, -1]])).all()

    assert np.isclose(QUANTUM_GATES["H"], (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])).all()
    assert np.isclose(QUANTUM_GATES["S"], np.array([[1.0, 0], [0, 1j]])).all()
    assert np.isclose(QUANTUM_GATES["T"], np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]])).all()


def test_parametric():
    phi_range = np.linspace(0, 2 * np.pi, 120)
    for phi in phi_range:
        assert np.isclose(QUANTUM_GATES["PHASE"](phi), np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])).all()
        assert np.isclose(
            QUANTUM_GATES["RX"](phi),
            np.array(
                [
                    [np.cos(phi / 2.0), -1j * np.sin(phi / 2.0)],
                    [-1j * np.sin(phi / 2.0), np.cos(phi / 2.0)],
                ]
            ),
        ).all()
        assert np.isclose(
            QUANTUM_GATES["RY"](phi),
            np.array([[np.cos(phi / 2.0), -np.sin(phi / 2.0)], [np.sin(phi / 2.0), np.cos(phi / 2.0)]]),
        ).all()
        assert np.isclose(
            QUANTUM_GATES["RZ"](phi),
            np.array(
                [
                    [np.cos(phi / 2.0) - 1j * np.sin(phi / 2.0), 0],
                    [0, np.cos(phi / 2.0) + 1j * np.sin(phi / 2.0)],
                ]
            ),
        ).all()


def test_multiq():
    assert np.isclose(QUANTUM_GATES["CNOT"], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])).all()
    assert np.isclose(
        QUANTUM_GATES["CCNOT"],
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        ),
    ).all()


def test_kraus_t1_normalization():
    kraus_ops = relaxation_operators(0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_t2_normalization():
    kraus_ops = dephasing_operators(0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_depolarizing_normalization():
    kraus_ops = depolarizing_operators(0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_bitflip_normalization():
    kraus_ops = bit_flip_operators(0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_phaseflip_normalization():
    kraus_ops = phase_flip_operators(0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))


def test_kraus_bitphaseflip_normalization():
    kraus_ops = bitphase_flip_operators(0.75)
    total = np.zeros((2, 2), dtype=complex)
    for kop in kraus_ops:
        total += np.conj(kop.T).dot(kop)
    assert np.allclose(total, np.eye(2))
