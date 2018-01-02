##############################################################################
# Copyright 2018 Rigetti Computing
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
"""
Module for creating and verifying noisy gate definitions in terms of Kraus maps.
"""

import numpy as np
from scipy.linalg import expm

from pyquil.parameters import format_parameter
from pyquil.quilbase import Pragma, Gate


def _check_kraus_ops(n, kraus_ops):
    """
    Verify that the Kraus operators are of the correct shape and satisfy the correct normalization.

    :param int n: Number of qubits
    :param list|tuple kraus_ops: The Kraus operators as numpy.ndarrays.
    """
    for k in kraus_ops:
        if not np.shape(k) == (2 ** n, 2 ** n):
            raise ValueError(
                "Kraus operators for {0} qubits must have shape {1}x{1}: {2}".format(n, 2 ** n, k))

    kdk_sum = sum(np.transpose(k).conjugate().dot(k) for k in kraus_ops)
    if not np.allclose(kdk_sum, np.eye(2 ** n), atol=1e-5):
        raise ValueError(
            "Kraus operator not correctly normalized: sum_j K_j^*K_j == {}".format(kdk_sum))


def _create_kraus_pragmas(name, qubit_indices, kraus_ops):
    """
    Generate the pragmas to define a Kraus map for a specific gate on some qubits.

    :param str name: The name of the gate.
    :param list|tuple qubit_indices: The qubits
    :param list|tuple kraus_ops: The Kraus operators as matrices.
    :return: A QUIL string with PRAGMA ADD-KRAUS ... statements.
    :rtype: str
    """

    pragmas = [Pragma("ADD-KRAUS",
                      [name] + list(qubit_indices),
                      "({})".format(" ".join(map(format_parameter, np.ravel(k)))))
               for k in kraus_ops]
    return pragmas


def append_kraus_to_gate(kraus_ops, gate_matrix):
    """
    Follow a gate ``gate_matrix`` by a Kraus map described by ``kraus_ops``.

    :param list kraus_ops: The Kraus operators.
    :param numpy.ndarray gate_matrix: The unitary gate.
    :return: A list of transformed Kraus operators.
    """
    return [kj.dot(gate_matrix) for kj in kraus_ops]


def damping_kraus_map(p=0.10):
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :param float p: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    damping_op = np.sqrt(p) * np.array([[0, 1],
                                        [0, 0]])

    residual_kraus = np.diag([1, np.sqrt(1 - p)])
    return [residual_kraus, damping_op]


def dephasing_kraus_map(p=0.10):
    """
    Generate the Kraus operators corresponding to a dephasing channel.

    :params float p: The one-step dephasing probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    return [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * np.diag([1, -1])]


def tensor_kraus_maps(k1, k2):
    """
    Generate the Kraus map corresponding to the composition
    of two maps on different qubits.

    :param list k1: The Kraus operators for the first qubit.
    :param list k2: The Kraus operators for the second qubit.
    :return: A list of tensored Kraus operators.
    """
    return [np.kron(k1j, k2l) for k1j in k1 for k2l in k2]


def combine_kraus_maps(k1, k2):
    """
    Generate the Kraus map corresponding to the composition
    of two maps on the same qubits with k1 being applied to the state
    after k2.

    :param list k1: The list of Kraus operators that are applied second.
    :param list k2: The list of Kraus operators that are applied first.
    :return: A combinatorially generated list of composed Kraus operators.
    """
    return [np.dot(k1j, k2l) for k1j in k1 for k2l in k2]


def damping_after_dephasing(T1, T2, gate_time):
    """
    Generate the Kraus map corresponding to the composition
    of a dephasing channel followed by an amplitude damping channel.

    :param float T1: The amplitude damping time
    :param float T2: The dephasing time
    :param float gate_time: The gate duration.
    :return: A list of Kraus operators.
    """
    damping = damping_kraus_map(p=gate_time / T1)
    dephasing = dephasing_kraus_map(p=gate_time / T2)
    return combine_kraus_maps(damping, dephasing)


# You can only apply gate-noise to non-parametrized gates,
# so we need to define placeholders for RX(+/- pi/2)
SINGLE_Q = {
    "noisy-x-plus90": expm(-1j * np.pi / 4 * np.array([[0, 1],
                                                       [1, 0]])),
    "noisy-x-minus90": expm(+1j * np.pi / 4 * np.array([[0, 1],
                                                        [1, 0]])),
}

TWO_Q = {
    "noisy-cz": np.diag([1, 1, 1, -1]),
}


def noisy_instruction(instruction):
    if isinstance(instruction, Pragma):
        return instruction

    if instruction.name == 'RZ':
        return instruction

    if instruction.name == 'I':
        return instruction

    if instruction.name == 'RX':
        assert len(instruction.params) == 1
        assert len(instruction.qubits) == 1
        if instruction.params[0] == np.pi / 2.0:
            return Gate('noisy-x-plus90', [], instruction.qubits)
        if instruction.params[0] == -np.pi / 2.0:
            return Gate('noisy-x-minus90', [], instruction.qubits)
        raise ValueError("Can't add noise to a parametric gate. "
                         "Try compiling to RX(pi/2) or RX(-pi/2)")

    if instruction.name == 'CZ':
        return Gate('noisy-cz', [], instruction.qubits)

    raise ValueError('Gate {} is not in the native instruction set'.format(instruction))


def _get_program_topology(prog):
    """
    Get the graph of two-qubit gates used in a program.

    :param prog: The program
    :return: (qubits, edges). ``qubits`` is a set of all integer qubits.
        ``edges`` is a set of tuples of two-qubit pairings.
    """
    qubits = set()
    edges = set()
    for instruction in prog:
        if isinstance(instruction, Pragma):
            continue
        qb = [q.index for q in instruction.qubits]
        qubits.update(qb)
        if len(qb) == 2:
            edges.add(tuple(sorted(qb)))
        if len(qb) > 2:
            raise ValueError("Encountered a >2 qubit instruction")

    return qubits, edges


def add_noise_to_program(prog, T1=30e-6, T2=None, gate_time_1q=50e-9, gate_time_2q=150e-09):
    """
    Add generic damping and dephasing noise to a program.

    This high-level function is provided as a convenience to investigate the effects of a
    generic noise model on a program. For more fine-grained control, please investigate
    the other methods available in the ``pyquil.kraus`` module.

    In an attempt to closely model the QPU, noisy versions of RX(+-pi/2) and CZ are provided;
    I and parametric RZ are noiseless, and other gates are not allowed. To use this function,
    you need to compile your program to this native gate set.

    The default noise parameters

        T1 = 30 us
        T2 = T1 / 2
        1q gate time = 50 ns
        2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param prog: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param T1: The T1 amplitude damping time. By default, this is 30 us
    :param T2: The T2 dephasing time. By default, this is one-half of the T1 time.
    :param gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :return: A new program with noisy operators.
    """

    if T2 is None:
        T2 = T1 / 2.0

    from pyquil.quil import Program  # Avoid circular dependency
    qubits, edges = _get_program_topology(prog)
    new_prog = Program()

    # Define noisy 1q gates
    for name, matrix in SINGLE_Q.items():
        new_prog.defgate(name, matrix)
        k_ops = append_kraus_to_gate(damping_after_dephasing(T1=T1, T2=T2, gate_time=gate_time_1q),
                                     gate_matrix=matrix)
        for qubit in qubits:
            new_prog.define_noisy_gate(name, (qubit,), k_ops)

    # Define noisy 2q gates
    for name, matrix in TWO_Q.items():
        new_prog.defgate(name, matrix)
        k1q = damping_after_dephasing(T1=T1, T2=T2, gate_time=gate_time_2q)
        k_total = append_kraus_to_gate(tensor_kraus_maps(k1q, k1q), gate_matrix=matrix)
        for q1, q2 in edges:
            # Note! q1 must be less than q2. This is done in _get_program_topology
            assert q1 < q2
            new_prog.define_noisy_gate(name, (q1, q2), k_total)

    # Translate noiseless gates to noisy gates.
    new_instrs = [noisy_instruction(instruction) for instruction in prog]
    new_prog.inst(new_instrs)

    return new_prog
