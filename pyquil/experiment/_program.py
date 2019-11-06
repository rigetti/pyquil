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
        qubits: set,
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

    expectation_alpha = p.declare(alpha_label, 'REAL', len(qubits))
    expectation_beta = p.declare(beta_label, 'REAL', len(qubits))
    expectation_gamma = p.declare(gamma_label, 'REAL', len(qubits))

    for idx, q in enumerate(qubits):
        p += RZ(expectation_alpha[idx], q)
        p += RX(np.pi / 2, q)
        p += RZ(expectation_beta[idx], q)
        p += RX(-np.pi / 2, q)
        p += RZ(expectation_gamma[idx], q)

    return p


def parameterized_single_qubit_state_preparation(
        qubits: set,
        label: str = 'preparation',
) -> Program:
    """

    :param qubits:
    :param label:
    :return:
    """
    return parameterized_euler_rotations(qubits, prefix=label)


def parameterized_single_qubit_measurement_basis(
        qubits: set,
        label: str = 'measurement',
) -> Program:
    """

    :param qubits:
    :param label:
    :return:
    """
    return parameterized_euler_rotations(qubits, prefix=label)


def parameterized_readout_symmetrization(
        qubits: set,
        label: str = 'symmetrization',
) -> Program:
    """

    :param qubits:
    :param label:
    :return:
    """
    p = Program()
    symmetrization = p.declare(f'{label}', 'REAL', len(qubits))
    for idx, q in enumerate(qubits):
        p += RX(symmetrization[idx], q)
    return p


def measure_qubits(qubits: set) -> Program:
    """

    :param qubits:
    :return:
    """
    p = Program()
    ro = p.declare('ro', 'BIT', len(qubits))
    for idx, q in enumerate(qubits):
        p += MEASURE(q, ro[idx])
    return p
