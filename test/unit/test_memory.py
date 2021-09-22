import numpy as np

from pyquil.experiment._memory import (
    merge_memory_map_lists,
    pauli_term_to_preparation_memory_map,
    pauli_term_to_measurement_memory_map,
)
from pyquil.paulis import sX, sY
from pyquil.quilatom import (
    MemoryReference,
    quil_cis,
    quil_cos,
    quil_exp,
    quil_sin,
    quil_sqrt,
    substitute,
    substitute_array,
)


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


def test_substitute_memory_reference():
    x_0 = MemoryReference("x", 0, declared_size=2)
    x_1 = MemoryReference("x", 1, declared_size=2)

    # complete substitutions

    assert substitute(x_0, {x_0: 5}) == 5

    assert substitute(x_0 + x_1, {x_0: +5, x_1: -5}) == 0
    assert substitute(x_0 - x_1, {x_0: +5, x_1: -5}) == 10
    assert substitute(x_0 * x_1, {x_0: +5, x_1: -5}) == -25
    assert substitute(x_0 / x_1, {x_0: +5, x_1: -5}) == -1

    assert substitute(x_0 * x_0 ** 2 / x_1, {x_0: 5, x_1: 10}) == 12.5

    assert np.isclose(substitute(quil_exp(x_0), {x_0: 5, x_1: 10}), np.exp(5))
    assert np.isclose(substitute(quil_sin(x_0), {x_0: 5, x_1: 10}), np.sin(5))
    assert np.isclose(substitute(quil_cos(x_0), {x_0: 5, x_1: 10}), np.cos(5))
    assert np.isclose(substitute(quil_sqrt(x_0), {x_0: 5, x_1: 10}), np.sqrt(5))
    assert np.isclose(substitute(quil_cis(x_0), {x_0: 5, x_1: 10}), np.exp(1j * 5.0))

    # incomplete substitutions

    y = MemoryReference("y", 0, declared_size=1)
    z = MemoryReference("z", 0, declared_size=1)

    assert substitute(y + z, {y: 5}) == 5 + z

    assert substitute(quil_cis(z), {y: 5}) == quil_cis(z)

    # array substitution pass-through

    a = MemoryReference("a", 0, declared_size=1)

    assert np.allclose(substitute_array([quil_sin(a), quil_cos(a)], {a: 5}), [np.sin(5), np.cos(5)])
