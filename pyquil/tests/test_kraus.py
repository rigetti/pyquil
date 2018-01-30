import numpy as np
import pytest

from pyquil.gates import CZ, RZ, RX, I, H
from pyquil.noise import (damping_kraus_map, dephasing_kraus_map, tensor_kraus_maps,
                          combine_kraus_maps, damping_after_dephasing, _noisy_instruction)
from pyquil.quil import Pragma


def test_damping_kraus_map():
    p = 0.05
    k1, k2 = damping_kraus_map(p=p)
    assert k1[1, 1] == np.sqrt(1 - p)
    assert k2[0, 1] == np.sqrt(p)


def test_dephasing_kraus_map():
    p = 0.05
    k1, k2 = dephasing_kraus_map(p=p)
    np.testing.assert_allclose(np.diag(k1), [np.sqrt(1 - p)] * 2)
    np.testing.assert_allclose(np.abs(np.diag(k2)), [np.sqrt(p)] * 2)


def test_tensor_kraus_maps():
    damping = damping_kraus_map()
    k1, k2, k3, k4 = tensor_kraus_maps(damping, damping)
    assert k1.shape == (4, 4)
    assert k2.shape == (4, 4)
    assert k3.shape == (4, 4)
    assert k4.shape == (4, 4)
    np.testing.assert_allclose(k1[-1, -1], 1 - 0.1)


def test_combine_kraus_maps():
    damping = damping_kraus_map()
    dephasing = dephasing_kraus_map()
    k1, k2, k3, k4 = combine_kraus_maps(damping, dephasing)
    assert k1.shape == (2, 2)
    assert k2.shape == (2, 2)
    assert k3.shape == (2, 2)
    assert k4.shape == (2, 2)


def test_damping_after_dephasing():
    damping = damping_kraus_map()
    dephasing = dephasing_kraus_map()
    ks_ref = combine_kraus_maps(damping, dephasing)

    ks_actual = damping_after_dephasing(10, 10, 1)
    np.testing.assert_allclose(ks_actual, ks_ref)


def test_noisy_instruction():
    # Unaffected
    assert I(0) == _noisy_instruction(I(0))
    assert RZ(.234)(0) == _noisy_instruction(RZ(.234)(0))
    assert Pragma('lalala') == _noisy_instruction(Pragma('lalala'))

    # Noisified
    assert _noisy_instruction(CZ(0, 1)).out() == 'noisy-cz 0 1'
    assert _noisy_instruction(RX(np.pi / 2)(0)).out() == 'noisy-x-plus90 0'
    assert _noisy_instruction(RX(-np.pi / 2)(23)).out() == 'noisy-x-minus90 23'

    # Unsupported
    with pytest.raises(ValueError):
        _noisy_instruction(H(0))

    with pytest.raises(ValueError):
        _noisy_instruction(RX(2 * np.pi / 3)(0))
