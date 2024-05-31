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
"""Miscellaneous tools that are helpful for simulation."""

from collections.abc import Sequence
from typing import Union, cast

import numpy as np

from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import QUANTUM_GATES, STATES, SWAP


def all_bitstrings(n_bits: int) -> np.ndarray:
    """All bitstrings in lexicographical order as a 2d np.ndarray.

    This should be the same as ``np.array(list(itertools.product([0,1], repeat=n_bits)))``
    but faster.
    """
    n_bitstrings = 2**n_bits
    out = np.zeros(shape=(n_bitstrings, n_bits), dtype=np.int8)

    tf = np.array([False, True])
    for i in range(n_bits):
        # Lexicographical ordering gives a pattern of 1's
        # where runs of 1s of length 2**j are tiled 2**i times

        # i indexes from the *left*
        # j indexes from the *right*
        j = n_bits - i - 1

        out[np.tile(np.repeat(tf, 2**j), 2**i), i] = 1
    return out


def qubit_adjacent_lifted_gate(i: int, matrix: np.ndarray, n_qubits: int) -> np.ndarray:
    """Lift k-qubit gate on adjacent qubits from qubit i to complete Hilbert space of dimension 2 ** num_qubits.

    Ex: 1-qubit gate, lifts from qubit i
    Ex: 2-qubit gate, lifts from qubits (i+1, i)
    Ex: 3-qubit gate, lifts from qubits (i+2, i+1, i), operating in that order

    In general, this takes a k-qubit gate (2D matrix 2^k x 2^k) and lifts
    it to the complete Hilbert space of dim 2^num_qubits, as defined by
    the right-to-left tensor product (1) in arXiv:1608.03355.

    Developer note: Quil and the QVM like qubits to be ordered such that qubit 0 is on the right.
    Therefore, in ``qubit_adjacent_lifted_gate``, ``lifted_pauli``, and ``lifted_state_operator``,
    we build up the lifted matrix by performing the kronecker product from right to left.

    Note that while the qubits are addressed in decreasing order,
    starting with num_qubit - 1 on the left and ending with qubit 0 on the
    right (in a little-endian fashion), gates are still lifted to apply
    on qubits in increasing index (right-to-left) order.

    :param i: starting qubit to lift matrix from (incr. index order)
    :param matrix: the matrix to be lifted
    :param n_qubits: number of overall qubits present in space

    :return: matrix representation of operator acting on the
        complete Hilbert space of all num_qubits.
    """
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("Matrix must be square")
    gate_size = np.log2(n_rows)
    if gate_size != int(gate_size):
        raise ValueError("Matrix must be 2^n by 2^n")
    gate_size = int(gate_size)

    # Outer-product to lift gate to complete Hilbert space

    # bottom: i qubits below target
    bottom_matrix = np.eye(2**i, dtype=np.complex128)
    # top: Nq - i (bottom) - gate_size (gate) qubits above target
    top_qubits = n_qubits - i - gate_size
    top_matrix = np.eye(2**top_qubits, dtype=np.complex128)

    return np.kron(top_matrix, np.kron(matrix, bottom_matrix))


def two_swap_helper(j: int, k: int, num_qubits: int, qubit_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate the permutation matrix that permutes two single-particle Hilbert spaces into adjacent positions.

    ALWAYS swaps j TO k. Recall that Hilbert spaces are ordered in decreasing
    qubit index order. Hence, j > k implies that j is to the left of k.

    End results:
        j == k: nothing happens
        j > k: Swap j right to k, until j at ind (k) and k at ind (k+1).
        j < k: Swap j left to k, until j at ind (k) and k at ind (k-1).

    Done in preparation for arbitrary 2-qubit gate application on ADJACENT
    qubits.

    :param j: starting qubit index
    :param k: ending qubit index
    :param num_qubits: number of qubits in Hilbert space
    :param qubit_map: current index mapping of qubits
    :return: tuple of swap matrix for the specified permutation,
             and the new qubit_map, after permutation is made
    """
    if not (0 <= j < num_qubits and 0 <= k < num_qubits):
        raise ValueError("Permutation SWAP index not valid")

    perm = np.eye(2**num_qubits, dtype=np.complex128)
    new_qubit_map = np.copy(qubit_map)

    if j == k:
        # nothing happens
        return perm, new_qubit_map
    elif j > k:
        # swap j right to k, until j at ind (k) and k at ind (k+1)
        for i in range(j, k, -1):
            perm = qubit_adjacent_lifted_gate(i - 1, SWAP, num_qubits).dot(perm)
            new_qubit_map[i - 1], new_qubit_map[i] = new_qubit_map[i], new_qubit_map[i - 1]
    elif j < k:
        # swap j left to k, until j at ind (k) and k at ind (k-1)
        for i in range(j, k, 1):
            perm = qubit_adjacent_lifted_gate(i, SWAP, num_qubits).dot(perm)
            new_qubit_map[i], new_qubit_map[i + 1] = new_qubit_map[i + 1], new_qubit_map[i]

    return perm, new_qubit_map


def permutation_arbitrary(qubit_inds: Sequence[int], n_qubits: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate the permutation matrix that permutes an arbitrary number of single-particle Hilbert spaces into adjacent positions.

    Transposes the qubit indices in the order they are passed to a
    contiguous region in the complete Hilbert space, in increasing
    qubit index order (preserving the order they are passed in).

    Gates are usually defined as `GATE 0 1 2`, with such an argument ordering
    dictating the layout of the matrix corresponding to GATE. If such an
    instruction is given, actual qubits (0, 1, 2) need to be swapped into the
    positions (2, 1, 0), because the lifting operation taking the 8 x 8 matrix
    of GATE is done in the little-endian (reverse) addressed qubit space.

    For example, suppose I have a Quil command CCNOT 20 15 10.
    The median of the qubit indices is 15 - hence, we permute qubits
    [20, 15, 10] into the final map [16, 15, 14] to minimize the number of
    swaps needed, and so we can directly operate with the final CCNOT, when
    lifted from indices [16, 15, 14] to the complete Hilbert space.

    Notes: assumes qubit indices are unique (assured in parent call).

    See documentation for further details and explanation.

    Done in preparation for arbitrary gate application on
    adjacent qubits.

    :param qubit_inds: Qubit indices in the order the gate is applied to.
    :param n_qubits: Number of qubits in system
    :return:
        perm - permutation matrix providing the desired qubit reordering
        qubit_arr - new indexing of qubits presented in left to right decreasing index order.
        start_i - starting index to lift gate from
    """  # noqa: E501
    # Begin construction of permutation
    perm = np.eye(2**n_qubits, dtype=np.complex128)

    # First, sort the list and find the median.
    sorted_inds = np.sort(qubit_inds)
    med_i = len(qubit_inds) // 2
    med = sorted_inds[med_i]

    # The starting position of all specified Hilbert spaces begins at
    # the qubit at (median - med_i)
    start = med - med_i
    # Array of final indices the arguments are mapped to, from
    # high index to low index, left to right ordering
    final_map = np.arange(start, start + len(qubit_inds))[::-1]
    start_i = final_map[-1]

    # Note that the lifting operation takes a k-qubit gate operating
    # on the qubits i+k-1, i+k-2, ... i (left to right).
    # two_swap_helper can be used to build the
    # permutation matrix by filling out the final map by sweeping over
    # the qubit_inds from left to right and back again, swapping qubits into
    # position. we loop over the qubit_inds until the final mapping matches
    # the argument.
    qubit_arr = np.arange(n_qubits)  # current qubit indexing

    made_it = False
    right = True
    while not made_it:
        array = range(len(qubit_inds)) if right else range(len(qubit_inds))[::-1]
        for i in array:
            pmod, qubit_arr = two_swap_helper(
                np.where(qubit_arr == qubit_inds[i])[0][0], final_map[i], n_qubits, qubit_arr
            )

            # update permutation matrix
            perm = pmod.dot(perm)
            if np.allclose(qubit_arr[final_map[-1] : final_map[0] + 1][::-1], qubit_inds):
                made_it = True
                break

        # for next iteration, go in opposite direction
        right = not right

    if not np.allclose(qubit_arr[final_map[-1] : final_map[0] + 1][::-1], qubit_inds):
        segment_values = qubit_arr[final_map[-1] : final_map[0] + 1][::-1]
        raise ValueError(
            f"Permutation matrix not constructed correctly. "
            f"Expected array segment {segment_values} to match qubit indices {qubit_inds}. "
            f"Segment values: {segment_values}, Expected values: {qubit_inds}"
        )

    return perm, qubit_arr[::-1], start_i


def lifted_gate_matrix(matrix: np.ndarray, qubit_inds: Sequence[int], n_qubits: int) -> np.ndarray:
    """Lift a unitary matrix to act on the specified qubits in a full ``n_qubits``-qubit Hilbert space.

    For 1-qubit gates, this is easy and can be achieved with appropriate kronning of identity
    matrices. For 2-qubit gates acting on adjacent qubit indices, it is also easy. However,
    for a multiqubit gate acting on non-adjactent qubit indices, we must first apply a permutation
    matrix to make the qubits adjacent and then apply the inverse permutation.

    :param matrix: A 2^k by 2^k matrix encoding an n-qubit operation, where ``k == len(qubit_inds)``
    :param qubit_inds: The qubit indices we wish the matrix to act on.
    :param n_qubits: The total number of qubits.
    :return: A 2^n by 2^n lifted version of the unitary matrix acting on the specified qubits.
    """
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("Matrix must be square")
    gate_size = np.log2(n_rows)
    if gate_size != int(gate_size):
        raise ValueError("Matrix must be 2^n by 2^n")
    gate_size = int(gate_size)

    pi_permutation_matrix, final_map, start_i = permutation_arbitrary(qubit_inds, n_qubits)
    if start_i > 0:
        check = final_map[-gate_size - start_i : -start_i]
    else:
        # Python can't deal with `arr[:-0]`
        check = final_map[-gate_size - start_i :]
    np.testing.assert_allclose(check, qubit_inds)

    v_matrix = qubit_adjacent_lifted_gate(start_i, matrix, n_qubits)
    return np.dot(np.conj(pi_permutation_matrix.T), np.dot(v_matrix, pi_permutation_matrix))  # type: ignore


def lifted_gate(gate: Gate, n_qubits: int) -> np.ndarray:
    """Lift a pyquil :py:class:`Gate` in a full ``n_qubits``-qubit Hilbert space.

    This function looks up the matrix form of the gate and then dispatches to
    :py:func:`lifted_gate_matrix` with the target qubits.

    :param gate: A gate
    :param n_qubits: The total number of qubits.
    :return: A 2^n by 2^n lifted version of the gate acting on its specified qubits.
    """
    zero = np.eye(2)
    zero[1, 1] = 0
    one = np.eye(2)
    one[0, 0] = 0

    if any(isinstance(param, Parameter) for param in gate.params):
        raise TypeError("Cannot produce a matrix from a gate with non-constant parameters.")

    # The main source of complexity is in handling handling FORKED gates. Given
    # a gate with modifiers, such as `FORKED CONTROLLED FORKED RX(a,b,c,d) 0 1
    # 2 3`, we get a tree, as in
    #
    #               FORKED CONTROLLED FORKED RX(a,b,c,d) 0 1 2 3
    #                 /                                      \
    #    CONTROLLED FORKED RX(a,b) 1 2 3       CONTROLLED FORKED RX(c,d) 1 2 3
    #                |                                        |
    #         FORKED RX(a,b) 2 3                      FORKED RX(c,d) 2 3
    #          /          \                            /          \
    #      RX(a) 3      RX(b) 3                    RX(c) 3      RX(d) 3
    #
    # We recurse on this structure using _gate_matrix below.

    def _gate_matrix(gate: Gate) -> np.ndarray:
        if len(gate.modifiers) == 0:  # base case
            if len(gate.params) > 0:
                return QUANTUM_GATES[gate.name](*gate.params)  # type: ignore
            else:
                return QUANTUM_GATES[gate.name]  # type: ignore
        else:
            mod = gate.modifiers[0]
            if mod == "DAGGER":
                child = _strip_modifiers(gate, limit=1)
                return _gate_matrix(child).conj().T
            elif mod == "CONTROLLED":
                child = _strip_modifiers(gate, limit=1)
                matrix = _gate_matrix(child)
                return np.kron(zero, np.eye(*matrix.shape)) + np.kron(one, matrix)  # type: ignore
            elif mod == "FORKED":
                if len(gate.params) % 2 != 0:
                    raise ValueError("FORKED gates must have an even number of parameters.")
                p0, p1 = gate.params[: len(gate.params) // 2], gate.params[len(gate.params) // 2 :]
                child = _strip_modifiers(gate, limit=1)
                # handle the first half of the FORKED params
                child.params = p0
                mat0 = _gate_matrix(child)
                # handle the second half of the FORKED params
                child.params = p1
                mat1 = _gate_matrix(child)
                return np.kron(zero, mat0) + np.kron(one, mat1)
            else:
                raise TypeError(f"Unsupported gate modifier {mod}")

    matrix = _gate_matrix(gate)

    return lifted_gate_matrix(matrix=matrix, qubit_inds=gate.get_qubit_indices(), n_qubits=n_qubits)


def program_unitary(program: Program, n_qubits: int) -> np.ndarray:
    """Return the unitary of a pyQuil program.

    :param program: A program consisting only of :py:class:`Gate`.:
    :return: a unitary corresponding to the composition of the program's gates.
    """
    umat: np.ndarray = np.eye(2**n_qubits)
    for instruction in program:
        if isinstance(instruction, Gate):
            unitary = lifted_gate(gate=instruction, n_qubits=n_qubits)
            umat = unitary.dot(umat)
        elif isinstance(instruction, Halt):
            pass
        else:
            raise ValueError(
                "Can only compute program unitary for programs composed of `Gate`s. "
                f"Found unsupported instruction: {instruction}"
            )
    return umat


def lifted_pauli(pauli_sum: Union[PauliSum, PauliTerm], qubits: list[int]) -> np.ndarray:
    """Return a matrix corresponding to the tensor representation of the given PauliSum and qubits.

    Useful for generating the full Hamiltonian after a particular fermion to
    pauli transformation. For example:

    Converting a PauliSum X0Y1 + Y1X0 into the matrix

    .. code-block:: python

       [
           [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 2.0j],
           [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
           [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
           [0.0 + 2.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
       ]

    Developer note: Quil and the QVM like qubits to be ordered such that qubit 0 is on the right.
    Therefore, in ``qubit_adjacent_lifted_gate``, ``lifted_pauli``, and ``lifted_state_operator``,
    we build up the lifted matrix by performing the kronecker product from right to left.

    :param pauli_sum: Pauli representation of an operator
    :param qubits: list of qubits in the order they will be represented in the resultant matrix.
    :return: matrix representation of the pauli_sum operator
    """
    if isinstance(pauli_sum, PauliTerm):
        pauli_sum = PauliSum([pauli_sum])

    n_qubits = len(qubits)
    result_hilbert = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    # left kronecker product corresponds to the correct basis ordering
    for term in pauli_sum.terms:
        term_hilbert = np.array([1])
        for qubit in qubits:
            term_hilbert = np.kron(QUANTUM_GATES[term[qubit]], term_hilbert)

        result_hilbert += term_hilbert * cast(complex, term.coefficient)

    return result_hilbert


def tensor_up(pauli_sum: Union[PauliSum, PauliTerm], qubits: list[int]) -> np.ndarray:
    """Return a matrix corresponding to the tensor representation of the given PauliSum and qubits.

    This is the same as :py:func:`lifted_pauli`. Nick R originally wrote this functionality
    and really likes the name ``tensor_up``. Who can blame him?

    :param pauli_sum: Pauli representation of an operator
    :param qubits: list of qubits in the order they will be represented in the resultant matrix.
    :return: matrix representation of the pauli_sum operator
    """
    return lifted_pauli(pauli_sum=pauli_sum, qubits=qubits)


def lifted_state_operator(state: TensorProductState, qubits: list[int]) -> np.ndarray:
    """Return a matrix corresponding to the tensored-up representation of the given state and qubits.

    Developer note: Quil and the QVM like qubits to be ordered such that qubit 0 is on the right.
    Therefore, in ``qubit_adjacent_lifted_gate``, ``lifted_pauli``, and ``lifted_state_operator``,
    we build up the lifted matrix by using the *left* kronecker product.

    :param state: The state
    :param qubits: list of qubits in the order they will be represented in the resultant matrix.
    """
    mat: np.ndarray = np.eye(1)
    for qubit in qubits:
        oneq_state = state[qubit]
        if oneq_state.qubit != qubit:
            raise ValueError(f"State qubit mismatch: {oneq_state.qubit} != {qubit}")
        state_vector = STATES[oneq_state.label][oneq_state.index][:, np.newaxis]
        state_matrix = state_vector @ state_vector.conj().T
        mat = np.kron(state_matrix, mat)
    return mat


def scale_out_phase(unitary1: np.ndarray, unitary2: np.ndarray) -> np.ndarray:
    """Return a matrix m equal to unitary1/θ where ɑ satisfies unitary2 = e^(iθ)·unitary1.

    :param unitary1: The unitary matrix from which the constant of
        proportionality should be scaled-out.
    :param unitary2: The reference matrix.

    :return: A matrix (same shape as the input matrices) with the
             constant of proportionality scaled-out.
    """
    # h/t quilc
    rescale_value = 1.0
    goodness_value = 0.0

    for j in range(unitary1.shape[0]):
        if np.abs(unitary1[j, 0]) > goodness_value:
            goodness_value = np.abs(unitary1[j, 0])
            rescale_value = unitary2[j, 0] / unitary1[j, 0]

    return rescale_value * unitary1


def unitary_equal(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if two matrices are unitarily equal."""
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")
    dim = A.shape[0]
    return np.allclose(np.abs(np.trace(A.T.conjugate() @ B) / dim), 1.0)
