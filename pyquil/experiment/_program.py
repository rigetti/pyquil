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

import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, RX, RZ


def parameterized_euler_rotations(
        num_qubits: int,
        *,
        prefix: str,
        label_alpha: str = 'alpha',
        label_beta: str = 'beta',
        label_gamma: str = 'gamma',
) -> Program:
    """

    :param qubits:
    :param prefix:
    :param label_alpha:
    :param label_beta:
    :param label_gamma:
    :return:
    """
    alpha_label = f'{prefix}_{label_alpha}'
    beta_label = f'{prefix}_{label_beta}'
    gamma_label = f'{prefix}_{label_gamma}'

    p = Program()

    alpha = p.declare(alpha_label, 'REAL', num_qubits)
    beta = p.declare(beta_label, 'REAL', num_qubits)
    gamma = p.declare(gamma_label, 'REAL', num_qubits)

    for idx, q in enumerate(range(num_qubits)):
        p += RZ(alpha[idx], q)
        p += RX(np.pi / 2, q)
        p += RZ(beta[idx], q)
        p += RX(-np.pi / 2, q)
        p += RZ(gamma[idx], q)

    return p


def parameterized_single_qubit_state_preparation(
        num_qubits: int,
        label: str = 'preparation',
) -> Program:
    """

    :param qubits:
    :param label:
    :return:
    """
    return parameterized_euler_rotations(num_qubits, prefix=label)


def parameterized_single_qubit_measurement_basis(
        num_qubits: int,
        label: str = 'measurement',
) -> Program:
    """

    :param qubits:
    :param label:
    :return:
    """
    return parameterized_euler_rotations(num_qubits, prefix=label)


def parameterized_readout_symmetrization(
        num_qubits: int,
        label: str = 'symmetrization',
) -> Program:
    """

    :param qubits:
    :param label:
    :return:
    """
    p = Program()
    symmetrization = p.declare(f'{label}', 'REAL', num_qubits)
    for idx, q in enumerate(range(num_qubits)):
        p += RX(symmetrization[idx], q)
    return p


def measure_qubits(num_qubits: int) -> Program:
    """

    :param qubits:
    :return:
    """
    p = Program()
    ro = p.declare('ro', 'BIT', num_qubits)
    for idx, q in enumerate(range(num_qubits)):
        p += MEASURE(q, ro[idx])
    return p
