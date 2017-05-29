import pytest
import numpy as np

from pyquil.wavefunction import get_bitstring_from_index, Wavefunction


@pytest.fixture()
def wvf():
    return Wavefunction(np.array([1.0, 1.j, 0.000005, 0.02]))


def test_get_bitstring_from_index():
    assert get_bitstring_from_index(0, 2) == '00'
    assert get_bitstring_from_index(3, 3) == '011'

    with pytest.raises(IndexError):
        get_bitstring_from_index(10, 2)


def test_parsers(wvf):
    outcome_probs = wvf.get_outcome_probs()
    assert len(outcome_probs.keys()) == 4

    pp_wvf = wvf.pretty_print()
    # this should round out one outcome
    assert pp_wvf == "(1+0j)|00> + 1j|01> + (0.02+0j)|11>"
    pp_wvf = wvf.pretty_print(1)
    assert pp_wvf == "(1+0j)|00> + 1j|01>"

    pp_probs = wvf.pretty_print_probabilities()
    # this should round out two outcomes
    assert len(pp_probs.keys()) == 2
    pp_probs = wvf.pretty_print_probabilities(5)
    assert len(pp_probs.keys()) == 3


def test_ground_state():
    ground = Wavefunction.ground(2)
    assert len(ground) == 2
    assert ground[0] == 1.0
