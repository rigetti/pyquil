import pytest
import numpy as np
import itertools

from pyquil.wavefunction import (
    get_bitstring_from_index,
    Wavefunction,
    _octet_bits,
)


@pytest.fixture()
def wvf():
    amps = np.array([1.0, 1.0j, 0.000005, 0.02])
    amps /= np.sqrt(np.sum(np.abs(amps) ** 2))
    return Wavefunction(amps)


def test_get_bitstring_from_index():
    assert get_bitstring_from_index(0, 2) == "00"
    assert get_bitstring_from_index(3, 3) == "011"

    with pytest.raises(IndexError):
        get_bitstring_from_index(10, 2)


def test_parsers(wvf):
    outcome_probs = wvf.get_outcome_probs()
    assert len(outcome_probs.keys()) == 4

    pp_wvf = wvf.pretty_print()
    # this should round out one outcome
    assert pp_wvf == "(0.71+0j)|00> + 0.71j|01> + (0.01+0j)|11>"
    pp_wvf = wvf.pretty_print(1)
    assert pp_wvf == "(0.7+0j)|00> + 0.7j|01>"

    pp_probs = wvf.pretty_print_probabilities()
    # this should round out two outcomes
    assert len(pp_probs.keys()) == 2
    pp_probs = wvf.pretty_print_probabilities(5)
    assert len(pp_probs.keys()) == 3


def test_ground_state():
    ground = Wavefunction.zeros(2)
    assert len(ground) == 2
    assert ground.amplitudes[0] == 1.0


def test_octet_bits():
    assert [0, 0, 0, 0, 0, 0, 0, 0] == _octet_bits(0b0)
    assert [1, 0, 0, 0, 0, 0, 0, 0] == _octet_bits(0b1)
    assert [0, 1, 0, 0, 0, 0, 0, 0] == _octet_bits(0b10)
    assert [1, 0, 1, 0, 0, 0, 0, 0] == _octet_bits(0b101)
    assert [1, 1, 1, 1, 1, 1, 1, 1] == _octet_bits(0b11111111)


def test_probabilities(wvf):
    n_qubits = 2
    bitstrings = [list(x) for x in itertools.product((0, 1), repeat=n_qubits)]
    prob_keys = ["".join(str(b) for b in bs) for bs in bitstrings]
    prob_dict = wvf.get_outcome_probs()
    probs1 = np.array([prob_dict[x] for x in prob_keys])
    probs2 = wvf.probabilities()
    np.testing.assert_array_equal(probs1, probs2)

    assert np.sum(probs2) == 1.0


def test_sample(wvf):
    bitstrings = wvf.sample_bitstrings(n_samples=100)
    assert bitstrings.shape == (100, 2)
    assert [0, 0] in bitstrings
