from pyquil.gate_matrices import QUANTUM_GATES, relaxation_operators, dephasing_operators, \
    depolarizing_operators, bit_flip_operators, phase_flip_operators, bitphase_flip_operators
import numpy as np


def test_singleq():
    I = QUANTUM_GATES['I']
    assert np.isclose(I, np.eye(2)).all()
    X = QUANTUM_GATES['X']
    assert np.isclose(X, np.array([[0, 1], [1, 0]])).all()
    Y = QUANTUM_GATES['Y']
    assert np.isclose(Y, np.array([[0, -1j], [1j, 0]])).all()
    Z = QUANTUM_GATES['Z']
    assert np.isclose(Z, np.array([[1, 0], [0, -1]])).all()

    H = QUANTUM_GATES['H']
    assert np.isclose(H, (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])).all()
    S = QUANTUM_GATES['S']
    assert np.isclose(S, np.array([[1.0, 0], [0, 1j]])).all()
    T = QUANTUM_GATES['T']
    assert np.isclose(T, np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]])).all()


def test_parametric():
    phi_range = np.linspace(0, 2 * np.pi, 120)
    for phi in phi_range:
        assert np.isclose(QUANTUM_GATES['PHASE'](phi),
                          np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])).all()
        assert np.isclose(QUANTUM_GATES['RX'](phi),
                          np.array([[np.cos(phi / 2.0), -1j * np.sin(phi / 2.0)],
                                    [-1j * np.sin(phi / 2.0), np.cos(phi / 2.0)]])).all()
        assert np.isclose(QUANTUM_GATES['RY'](phi),
                          np.array([[np.cos(phi / 2.0), -np.sin(phi / 2.0)],
                                    [np.sin(phi / 2.0), np.cos(phi / 2.0)]])).all()
        assert np.isclose(QUANTUM_GATES['RZ'](phi),
                          np.array([[np.cos(phi / 2.0) - 1j * np.sin(phi / 2.0), 0],
                                    [0, np.cos(phi / 2.0) + 1j * np.sin(phi / 2.0)]])).all()


def test_multiq():
    assert np.isclose(QUANTUM_GATES['CNOT'], np.array([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 0, 1],
                                                       [0, 0, 1, 0]])).all()
    assert np.isclose(QUANTUM_GATES['CCNOT'], np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 1, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 1, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 1, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 1, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 1],
                                                        [0, 0, 0, 0, 0, 0, 1, 0]])).all()


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
