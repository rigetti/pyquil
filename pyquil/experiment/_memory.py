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
from typing import Dict, List, Tuple

import numpy as np

from pyquil.paulis import PauliTerm


def euler_angles_RX(theta: float) -> Tuple[float, float, float]:
    """

    :param theta:
    :return:
    """
    return (np.pi / 2, theta, np.pi / 2)


def euler_angles_RY(theta: float) -> Tuple[float, float, float]:
    """

    :param theta:
    :return:
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
        label_alpha: str = 'alpha',
        label_beta: str = 'beta',
        label_gamma: str = 'gamma',
) -> Dict[str, List[float]]:
    """

    :param term:
    :param prefix:
    :param tuple_x:
    :param tuple_y:
    :param tuple_z:
    :param label_alpha:
    :param label_beta:
    :param label_gamma:
    :return:
    """
    # no need to provide a memory map when no rotations are necessary
    if ('X' not in term.pauli_string()) and ('Y' not in term.pauli_string()):
        return {}

    alpha_label = f'{prefix}_{label_alpha}'
    beta_label = f'{prefix}_{label_beta}'
    gamma_label = f'{prefix}_{label_gamma}'

    # assume the pauli indices are equivalent to the memory region
    memory_size = max(term.get_qubits()) + 1

    memory_map = {alpha_label: [0.0] * memory_size,
                  beta_label: [0.0] * memory_size,
                  gamma_label: [0.0] * memory_size}

    for qubit, operator in term:
        if operator == 'X':
            memory_map[alpha_label][qubit] = tuple_x[0]
            memory_map[beta_label][qubit] = tuple_x[1]
            memory_map[gamma_label][qubit] = tuple_x[2]
        elif operator == 'Y':
            memory_map[alpha_label][qubit] = tuple_y[0]
            memory_map[beta_label][qubit] = tuple_y[1]
            memory_map[gamma_label][qubit] = tuple_y[2]
        elif operator == 'Z':
            memory_map[alpha_label][qubit] = tuple_z[0]
            memory_map[beta_label][qubit] = tuple_z[1]
            memory_map[gamma_label][qubit] = tuple_z[2]
        elif operator == 'I':
            memory_map[alpha_label][qubit] = tuple_z[0]
            memory_map[beta_label][qubit] = tuple_z[1]
            memory_map[gamma_label][qubit] = tuple_z[2]
        else:
            raise ValueError(f'Unknown operator {operator}')

    return memory_map


def pauli_term_to_preparation_memory_map(
        term: PauliTerm,
        label: str = 'preparation',
) -> Dict[str, List[float]]:
    """

    :param term:
    :param label:
    :return:
    """
    return pauli_term_to_euler_memory_map(term,
                                          prefix=label,
                                          tuple_x=P_X,
                                          tuple_y=P_Y,
                                          tuple_z=P_Z)


def pauli_term_to_measurement_memory_map(
        term: PauliTerm,
        label: str = 'measurement',
) -> Dict[str, List[float]]:
    """

    :param term:
    :param label:
    :return:
    """
    return pauli_term_to_euler_memory_map(term,
                                          prefix=label,
                                          tuple_x=M_X,
                                          tuple_y=M_Y,
                                          tuple_z=M_Z)


def build_symmetrization_memory_maps(
        memory_size: int,
        symmetrization_level: int = -1,
        label: str = 'symmetrization'
) -> List[Dict[str, List[float]]]:
    """

    :param size:
    :param level:
    :param label:
    :return:
    """
    if symmetrization_level == 0:
        return [{}]

    # TODO: add support for orthogonal arrays
    if symmetrization_level != -1:
        raise ValueError('We only support exhaustive symmetrization for now.')

    assignments = itertools.product(np.array([0, np.pi]), repeat=memory_size)
    memory_maps = []
    for a in assignments:
        memory_maps.append({f'{label}': a})
    return memory_maps


def merge_memory_map_lists(
        mml1: List[Dict[str, List[float]]],
        mml2: List[Dict[str, List[float]]]
) -> List[Dict[str, List[float]]]:
    """

    :param mml1:
    :param mml2:
    :return:
    """
    if not mml1:
        return mml2
    if not mml2:
        return mml1
    return [{**d1, **d2} for d1, d2 in itertools.product(mml1, mml2)]
