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

from collections.abc import Sequence

import numpy as np

from pyquil.gates import MEASURE, RX, RZ
from pyquil.quil import Program


def parameterized_euler_rotations(
    qubits: Sequence[int],
    *,
    prefix: str,
    suffix_alpha: str = "alpha",
    suffix_beta: str = "beta",
    suffix_gamma: str = "gamma",
) -> Program:
    """Given a number of qubits (n), build a ``Program`` containing a ZXZXZ-decomposed gate on each qubit.

    Each ``RZ`` is parameterized by declared values with labels given by the "prefix" and "suffix" arguments. Put more
    plainly, the resulting Quil program on n qubits is::

        RZ(alpha_label[0]) 0
        RX(pi/2) 0
        RZ(beta_label[0]) 0
        RX(-pi/2) 0
        RZ(gamma_label[0]) 0
        ...
        RZ(alpha_label[n-1]) n-1
        RX(pi/2) n-1
        RZ(beta_label[0]) n-1
        RX(-pi/2) n-1
        RZ(gamma_label[n-1]) n-1

    :param qubits: The number of qubits (n).
    :param prefix: The prefix for the declared memory region labels. For example, if the prefix
        is "preparation" and the alpha, beta, and gamma suffixes are left as default, the labels
        would be "preparation_alpha", "preparation_beta", and "preparation_gamma".
    :param suffix_alpha: The suffix for the "alpha" memory region label, which corresponds to the
        first (rightmost) ``Z`` in the ZXZXZ decomposition. Defaults to "alpha".
    :param suffix_beta: The suffix for the "beta" memory region label, which corresponds to the
        second (middle) ``Z`` in the ZXZXZ decomposition. Defaults to "beta".
    :param suffix_gamma: The suffix for the "gamma" memory region label, which corresponds to the
        last (leftmost) ``Z`` in the ZXZXZ decomposition. Defaults to "gamma".
    :return: A ``Program`` containing a 3 parameterized ``RZ``s and 2 fixed ``RX``s per qubit.
    """
    alpha_label = f"{prefix}_{suffix_alpha}"
    beta_label = f"{prefix}_{suffix_beta}"
    gamma_label = f"{prefix}_{suffix_gamma}"

    p = Program()

    alpha = p.declare(alpha_label, "REAL", len(qubits))
    beta = p.declare(beta_label, "REAL", len(qubits))
    gamma = p.declare(gamma_label, "REAL", len(qubits))

    for idx, q in enumerate(qubits):
        p += RZ(alpha[idx], q)
        p += RX(np.pi / 2, q)
        p += RZ(beta[idx], q)
        p += RX(-np.pi / 2, q)
        p += RZ(gamma[idx], q)

    return p


def parameterized_single_qubit_state_preparation(qubits: Sequence[int], label: str = "preparation") -> Program:
    """Produce a program as in ``parameterized_euler_rotations`` where each memory region is prefixed by "preparation".

    :param qubits: The number of qubits (n).
    :param label: The prefix to use when declaring memory in ``parameterized_euler_rotations``. (default: "preparation")
    :return: A parameterized ``Program`` that can be used to prepare a product state.
    """
    return parameterized_euler_rotations(qubits, prefix=label)


def parameterized_single_qubit_measurement_basis(qubits: Sequence[int], label: str = "measurement") -> Program:
    """Produce a program as in ``parameterized_euler_rotations`` where each memory region is prefixed by "measurement".

    :param qubits: The number of qubits (n).
    :param label: The prefix to use when declaring memory in ``parameterized_euler_rotations``. (default: "measurement")
    :return: A parameterized ``Program`` that can be used to prepare a product state.
    """
    return parameterized_euler_rotations(qubits, prefix=label)


def parameterized_readout_symmetrization(qubits: Sequence[int], label: str = "symmetrization") -> Program:
    """Given a number of qubits (n), produce a parameterized ``Program`` with an ``RX`` instruction on qubits [0, n-1].

    Qubits 0 through n-1 are parameterized by memory regions label[0] through label[n-1], where "label" defaults to
    "symmetrization".

    :param qubits: The number of qubits (n).
    :param label: The name of the declared memory region. (default: "symmetrization")
    :return: A ``Program`` with parameterized ``RX`` gates on n qubits.
    """
    p = Program()
    symmetrization = p.declare(f"{label}", "REAL", len(qubits))
    for idx, q in enumerate(qubits):
        p += RX(symmetrization[idx], q)
    return p


def measure_qubits(qubits: Sequence[int]) -> Program:
    """Given a number of qubits (n), produce a ``Program`` with a ``MEASURE`` instruction on qubits [0, n-1].

    Each MEASURE is written to corresponding readout registers ro[0] through ro[n-1].

    :param qubits: The number of qubits (n).
    :return: A ``Program`` that measures n qubits.
    """
    p = Program()
    ro = p.declare("ro", "BIT", len(qubits))
    for idx, q in enumerate(qubits):
        p += MEASURE(q, ro[idx])
    return p
