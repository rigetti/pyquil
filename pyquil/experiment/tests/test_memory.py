import numpy as np

from pyquil.experiment._memory import (
    merge_memory_map_lists,
    pauli_term_to_preparation_memory_map,
    pauli_term_to_measurement_memory_map,
)
from pyquil.paulis import sX, sY


def test_merge_memory_map_lists():
    mml = [
        {"a": [1.0], "b": [3.0], "c": [4.0]},
        {"a": [1.0], "b": [5.0], "c": [6.0]},
        {"a": [2.0], "b": [3.0], "c": [4.0]},
        {"a": [2.0], "b": [5.0], "c": [6.0]},
    ]
    assert mml == merge_memory_map_lists(
        [{"a": [1.0]}, {"a": [2.0]}], [{"b": [3.0], "c": [4.0]}, {"b": [5.0], "c": [6.0]}]
    )


def test_pauli_term_to_preparation_memory_map():
    assert pauli_term_to_preparation_memory_map(sX(0)) == {
        "preparation_alpha": [0.0],
        "preparation_beta": [np.pi / 2],
        "preparation_gamma": [0.0],
    }


def test_pauli_term_to_measurement_memory_map():
    assert pauli_term_to_measurement_memory_map(sY(1)) == {
        "measurement_alpha": [0.0, np.pi / 2],
        "measurement_beta": [0.0, np.pi / 2],
        "measurement_gamma": [0.0, -np.pi / 2],
    }
