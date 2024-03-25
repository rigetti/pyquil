##############################################################################
# Copyright 2016-2019 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import itertools
from typing import Dict, List, Tuple, cast

import numpy as np

from pyquil.paulis import PauliTerm


def euler_angles_RX(theta: float) -> Tuple[float, float, float]:
    """
    A tuple of angles which corresponds to a ZXZXZ-decomposed ``RX`` gate.

    :param theta: The angle parameter for the ``RX`` gate.
    :return: The corresponding Euler angles for that gate.
    """
    return (np.pi / 2, theta, -np.pi / 2)


def euler_angles_RY(theta: float) -> Tuple[float, float, float]:
    """
    A tuple of angles which corresponds to a ZXZXZ-decomposed ``RY`` gate.

    :param theta: The angle parameter for the ``RY`` gate.
    :return: The corresponding Euler angles for that gate.
    """
    return (0.0, theta, 0.0)


# euler angles for preparing the +1 eigenstate of X, Y, or Z
P_X = euler_angles_RY(np.pi / 2)
P_Y = euler_angles_RX(-np.pi / 2)
P_Z = (0.0, 0.0, 0.0)


# euler angles for measuring in the eigenbasis of X, Y, or Z
M_X = euler_angles_RY(-np.pi / 2)
M_Y = euler_angles_RX(np.pi / 2)
M_Z = (0.0, 0.0, 0.0)


def pauli_term_to_euler_memory_map(
    term: PauliTerm,
    *,
    prefix: str,
    tuple_x: Tuple[float, float, float],
    tuple_y: Tuple[float, float, float],
    tuple_z: Tuple[float, float, float],
    suffix_alpha: str = "alpha",
    suffix_beta: str = "beta",
    suffix_gamma: str = "gamma",
) -> Dict[str, List[float]]:
    """
    Given a ``PauliTerm``, create a memory map corresponding to a collection of ZXZXZ-decomposed
    single-qubit gates. The intent is that these gate are used to prepare an eigenstate of the
    ``PauliTerm`` or measure in the eigenbasis of the ``PauliTerm``, which is more clearly
    discernible from the calling functions ``pauli_term_to_preparation_memory_map`` (for state
    preparation) and ``pauli_term_to_measurement_memory_map`` (for measuring in different bases).
    This function is not really meant to be used by itself, but rather by the aforementioned
    calling functions.

    :param term: The ``PauliTerm`` in question.
    :param prefix: The prefix for the declared memory region labels. For example, if the prefix
        is "preparation" and the alpha, beta, and gamma suffixes are left as default, the labels
        would be "preparation_alpha", "preparation_beta", and "preparation_gamma".
    :param tuple_x: A tuple of Euler angles as (alpha, beta, gamma) to be used for the ``X``
        operators in the ``PauliTerm``.
    :param tuple_y: A tuple of Euler angles as (alpha, beta, gamma) to be used for the ``Y``
        operators in the ``PauliTerm``.
    :param tuple_z: A tuple of Euler angles as (alpha, beta, gamma) to be used for the ``Z``
        and ``I`` operators in the ``PauliTerm``.
    :param suffix_alpha: The suffix for the "alpha" memory region label, which corresponds to the
        first (rightmost) ``Z`` in the ZXZXZ decomposition. Defaults to "alpha".
    :param suffix_beta: The suffix for the "beta" memory region label, which corresponds to the
        second (middle) ``Z`` in the ZXZXZ decomposition. Defaults to "beta".
    :param suffix_gamma: The suffix for the "gamma" memory region label, which corresponds to the
        last (leftmost) ``Z`` in the ZXZXZ decomposition. Defaults to "gamma".
    :return: Memory map dictionary containing three entries (three labels as keys and three lists
        of angles as values).
    """
    # no need to provide a memory map when no rotations are necessary
    if ("X" not in term.pauli_string(cast(List[int], term.get_qubits()))) and (
        "Y" not in term.pauli_string(cast(List[int], term.get_qubits()))
    ):
        return {}

    alpha_label = f"{prefix}_{suffix_alpha}"
    beta_label = f"{prefix}_{suffix_beta}"
    gamma_label = f"{prefix}_{suffix_gamma}"

    # assume the pauli indices are equivalent to the memory region
    memory_size = max(cast(List[int], term.get_qubits())) + 1

    memory_map = {
        alpha_label: [0.0] * memory_size,
        beta_label: [0.0] * memory_size,
        gamma_label: [0.0] * memory_size,
    }

    tuples = {"X": tuple_x, "Y": tuple_y, "Z": tuple_z, "I": tuple_z}

    for qubit, operator in term:
        assert isinstance(qubit, int)
        if operator not in tuples:
            raise ValueError(f"Unknown operator {operator}")
        memory_map[alpha_label][qubit] = tuples[operator][0]
        memory_map[beta_label][qubit] = tuples[operator][1]
        memory_map[gamma_label][qubit] = tuples[operator][2]

    return memory_map


def pauli_term_to_preparation_memory_map(term: PauliTerm, label: str = "preparation") -> Dict[str, List[float]]:
    """
    Given a ``PauliTerm``, create a memory map corresponding to the ZXZXZ-decomposed single-qubit
    gates that prepare the plus one eigenstate of the ``PauliTerm``. For example, if we have the
    following program:

        RZ(preparation_alpha[0]) 0
        RX(pi/2) 0
        RZ(preparation_beta[0]) 0
        RX(-pi/2) 0
        RZ(preparation_gamma[0]) 0

    We can prepare the ``|+>`` state (by default we start in the ``|0>`` state) by providing the
    following memory map (which corresponds to ``RY(pi/2)``):

        {'preparation_alpha': [0.0], 'preparation_beta': [pi/2], 'preparation_gamma': [0.0]}

    :param term: The ``PauliTerm`` in question.
    :param label: The prefix to provide to ``pauli_term_to_euler_memory_map``, for labeling the
        declared memory regions. Defaults to "preparation".
    :return: Memory map for preparing the desired state.
    """
    return pauli_term_to_euler_memory_map(term, prefix=label, tuple_x=P_X, tuple_y=P_Y, tuple_z=P_Z)


def pauli_term_to_measurement_memory_map(term: PauliTerm, label: str = "measurement") -> Dict[str, List[float]]:
    """
    Given a ``PauliTerm``, create a memory map corresponding to the ZXZXZ-decomposed single-qubit
    gates that allow for measurement in the eigenbasis of the ``PauliTerm``. For example, if we
    have the following program:

        RZ(measurement_alpha[0]) 0
        RX(pi/2) 0
        RZ(measurement_beta[0]) 0
        RX(-pi/2) 0
        RZ(measurement_gamma[0]) 0
        MEASURE 0 ro[0]

    We can measure in the ``Y`` basis (by default we measure in the ``Z`` basis) by providing the
    following memory map (which corresponds to ``RX(pi/2)``):

        {'measurement_alpha': [pi/2], 'measurement_beta': [pi/2], 'measurement_gamma': [pi/2]}

    :param term: The ``PauliTerm`` in question.
    :param label: The prefix to provide to ``pauli_term_to_euler_memory_map``, for labeling the
        declared memory regions. Defaults to "measurement".
    :return: Memory map for measuring in the desired basis.
    """
    return pauli_term_to_euler_memory_map(term, prefix=label, tuple_x=M_X, tuple_y=M_Y, tuple_z=M_Z)


def merge_memory_map_lists(
    mml1: List[Dict[str, List[float]]], mml2: List[Dict[str, List[float]]]
) -> List[Dict[str, List[float]]]:
    """
    Given two lists of memory maps, produce the "cartesian product" of the memory maps:

        merge_memory_map_lists([{a: 1}, {a: 2}], [{b: 3, c: 4}, {b: 5, c: 6}])

        -> [{a: 1, b: 3, c: 4}, {a: 1, b: 5, c: 6}, {a: 2, b: 3, c: 4}, {a: 2, b: 5, c: 6}]

    :param mml1: The first memory map list.
    :param mml2: The second memory map list.
    :return: A list of the merged memory maps.
    """
    if not mml1:
        return mml2
    if not mml2:
        return mml1
    return [{**d1, **d2} for d1, d2 in itertools.product(mml1, mml2)]
