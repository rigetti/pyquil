import numpy as np
import pytest

from pyquil import Program
from pyquil.experiment import plusX, minusZ
from pyquil.gates import CCNOT, CNOT, CZ, H, MEASURE, PHASE, RX, RY, RZ, X, Y, Z
from pyquil.paulis import sX, sY, sZ
from pyquil.quilatom import MemoryReference, Parameter
from pyquil.quilbase import Declare
from pyquil.simulation import matrices as mat
from pyquil.simulation.tools import (
    qubit_adjacent_lifted_gate,
    program_unitary,
    lifted_gate_matrix,
    lifted_gate,
    lifted_pauli,
    lifted_state_operator,
)


def test_random_gates():
    p = Program().inst([H(0), H(1), H(0)])
    test_unitary = program_unitary(p, n_qubits=2)
    actual_unitary = np.kron(mat.H, np.eye(2 ** 1))
    assert np.allclose(test_unitary, actual_unitary)


def test_random_gates_2():
    p = Program().inst([H(0), X(1), Y(2), Z(3)])
    test_unitary = program_unitary(p, n_qubits=4)
    actual_unitary = np.kron(mat.Z, np.kron(mat.Y, np.kron(mat.X, mat.H)))
    assert np.allclose(test_unitary, actual_unitary)


def test_random_gates_3():
    p = Program(X(2), CNOT(2, 1), CNOT(1, 0))
    test_unitary = program_unitary(p, n_qubits=3)
    # gates are multiplied in 'backwards' order
    actual_unitary = (
        np.kron(np.eye(2 ** 1), mat.QUANTUM_GATES["CNOT"])
        .dot(np.kron(mat.QUANTUM_GATES["CNOT"], np.eye(2 ** 1)))
        .dot(np.kron(mat.QUANTUM_GATES["X"], np.eye(2 ** 2)))
    )
    np.testing.assert_allclose(actual_unitary, test_unitary)


def test_identity():
    p = Program()
    test_unitary = program_unitary(p, 0)
    assert np.allclose(test_unitary, np.eye(2 ** 0))


def test_qaoa_unitary():
    wf_true = [
        0.00167784 + 1.00210180e-05 * 1j,
        0.50000000 - 4.99997185e-01 * 1j,
        0.50000000 - 4.99997185e-01 * 1j,
        0.00167784 + 1.00210180e-05 * 1j,
    ]

    prog = Program(
        [
            RY(np.pi / 2, 0),
            RX(np.pi, 0),
            RY(np.pi / 2, 1),
            RX(np.pi, 1),
            CNOT(0, 1),
            RX(-np.pi / 2, 1),
            RY(4.71572463191, 1),
            RX(np.pi / 2, 1),
            CNOT(0, 1),
            RX(-2 * 2.74973750579, 0),
            RX(-2 * 2.74973750579, 1),
        ]
    )

    test_unitary = program_unitary(prog, n_qubits=2)
    wf_test = np.zeros(4)
    wf_test[0] = 1.0
    wf_test = test_unitary.dot(wf_test)
    assert np.allclose(wf_test, wf_true)


def test_unitary_measure():
    prog = Program(Declare("ro", "BIT"), H(0), H(1), MEASURE(0, MemoryReference("ro", 0)))
    with pytest.raises(ValueError):
        program_unitary(prog, n_qubits=2)


def test_lifted_swap():
    # SWAP indexed at 0
    test_matrix = qubit_adjacent_lifted_gate(0, mat.SWAP, 2)
    result = mat.SWAP
    assert np.allclose(test_matrix, result)


def test_lifted_swap_2():
    test_matrix = qubit_adjacent_lifted_gate(0, mat.SWAP, 3)
    result = np.kron(np.eye(2 ** 1), mat.SWAP)
    assert np.allclose(test_matrix, result)


def test_lifted_swap_3():
    test_matrix = qubit_adjacent_lifted_gate(0, mat.SWAP, 4)
    result = np.kron(np.eye(2 ** 2), mat.SWAP)
    assert np.allclose(test_matrix, result)


def test_lifted_swap_4():
    # SWAP indexed at max num_qubits
    test_matrix = qubit_adjacent_lifted_gate(1, mat.SWAP, 3)
    result = np.kron(mat.SWAP, np.eye(2))
    assert np.allclose(test_matrix, result)


def test_lifted_swap_6():
    test_matrix = qubit_adjacent_lifted_gate(1, mat.SWAP, 4)
    result = np.kron(np.eye(2 ** 1), np.kron(mat.SWAP, np.eye(2 ** 1)))
    assert np.allclose(test_matrix, result)


def test_lifted_swap_7():
    test_matrix = qubit_adjacent_lifted_gate(2, mat.SWAP, 4)
    result = np.kron(np.eye(2 ** 0), np.kron(mat.SWAP, np.eye(2 ** 2)))
    assert np.allclose(test_matrix, result)


def test_lifted_swap_8():
    test_matrix = qubit_adjacent_lifted_gate(8, mat.SWAP, 10)
    result = np.kron(np.eye(2 ** 0), np.kron(mat.SWAP, np.eye(2 ** 8)))
    assert np.allclose(test_matrix, result)


def test_two_qubit_gates_1():
    unitary_test = lifted_gate_matrix(mat.CNOT, [1, 0], 2)
    unitary_true = np.kron(mat.P0, np.eye(2)) + np.kron(mat.P1, mat.X)
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_2():
    unitary_test = lifted_gate_matrix(mat.CNOT, [0, 1], 2)
    unitary_true = np.kron(np.eye(2), mat.P0) + np.kron(mat.X, mat.P1)
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_3():
    unitary_test = lifted_gate_matrix(mat.CNOT, [2, 1], 3)
    unitary_true = np.kron(mat.CNOT, np.eye(2 ** 1))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_4():
    with pytest.raises(IndexError):
        lifted_gate_matrix(mat.CNOT, [2, 1], 2)


def test_two_qubit_gates_5():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [0, 1], 3)
    unitary_true = np.kron(np.eye(2), mat.ISWAP)
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_6():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [1, 0], 3)
    unitary_true = np.kron(np.eye(2), mat.ISWAP)
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_7():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [1, 2], 4)
    unitary_true = np.kron(np.eye(2), np.kron(mat.ISWAP, np.eye(2)))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_8():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [3, 2], 4)
    unitary_true = np.kron(mat.ISWAP, np.eye(4))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_9():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [2, 3], 4)
    unitary_true = np.kron(mat.ISWAP, np.eye(4))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_10():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [0, 3], 4)
    swap_01 = np.kron(np.eye(4), mat.SWAP)
    swap_12 = np.kron(np.eye(2), np.kron(mat.SWAP, np.eye(2)))
    swapper = swap_12.dot(swap_01)
    V = np.kron(mat.ISWAP, np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_11():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [3, 0], 4)
    swap_01 = np.kron(np.eye(4), mat.SWAP)
    swap_12 = np.kron(np.eye(2), np.kron(mat.SWAP, np.eye(2)))
    swapper = swap_12.dot(swap_01)
    V = np.kron(mat.ISWAP, np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_12():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [1, 3], 4)
    swap_12 = np.kron(np.eye(2), np.kron(mat.SWAP, np.eye(2)))
    swapper = swap_12
    V = np.kron(mat.ISWAP, np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_13():
    unitary_test = lifted_gate_matrix(mat.ISWAP, [3, 1], 4)
    swap_12 = np.kron(np.eye(2), np.kron(mat.SWAP, np.eye(2)))
    swapper = swap_12
    V = np.kron(mat.ISWAP, np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_14():
    unitary_test = lifted_gate_matrix(mat.CNOT, [3, 1], 4)
    swap_12 = np.kron(np.eye(2), np.kron(mat.SWAP, np.eye(2)))
    swapper = swap_12
    V = np.kron(mat.CNOT, np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_two_qubit_gates_15():
    unitary_test = lifted_gate_matrix(mat.SWAP, [3, 1], 4)
    swap_12 = np.kron(np.eye(2), np.kron(mat.SWAP, np.eye(2)))
    swapper = swap_12
    V = np.kron(mat.SWAP, np.eye(4))
    unitary_true = np.dot(np.conj(swapper.T), np.dot(V, swapper))
    assert np.allclose(unitary_test, unitary_true)


def test_single_qubit_gates_1():
    test_unitary = lifted_gate_matrix(mat.H, [0], 4)
    true_unitary = np.kron(np.eye(8), mat.H)
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_2():
    test_unitary = lifted_gate_matrix(mat.H, [1], 4)
    true_unitary = np.kron(np.eye(4), np.kron(mat.H, np.eye(2)))
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_3():
    test_unitary = lifted_gate_matrix(mat.H, [2], 4)
    true_unitary = np.kron(np.eye(2), np.kron(mat.H, np.eye(4)))
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_4():
    test_unitary = lifted_gate_matrix(mat.H, [3], 4)
    true_unitary = np.kron(mat.H, np.eye(8))
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_5():
    test_unitary = lifted_gate_matrix(mat.H, [0], 5)
    true_unitary = np.kron(np.eye(2 ** 4), mat.H)
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_6():
    test_unitary = lifted_gate_matrix(mat.H, [1], 5)
    true_unitary = np.kron(np.eye(2 ** 3), np.kron(mat.H, np.eye(2)))
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_7():
    test_unitary = lifted_gate_matrix(mat.H, [2], 5)
    true_unitary = np.kron(np.eye(2 ** 2), np.kron(mat.H, np.eye(2 ** 2)))
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_8():
    test_unitary = lifted_gate_matrix(mat.H, [3], 5)
    true_unitary = np.kron(np.eye(2 ** 1), np.kron(mat.H, np.eye(2 ** 3)))
    assert np.allclose(test_unitary, true_unitary)


def test_single_qubit_gates_9():
    test_unitary = lifted_gate_matrix(mat.H, [4], 5)
    true_unitary = np.kron(np.eye(2 ** 0), np.kron(mat.H, np.eye(2 ** 4)))
    assert np.allclose(test_unitary, true_unitary)


def test_lifted_gate_single_qubit():
    test_unitary = lifted_gate(H(0), 1)
    true_unitary = mat.H
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(H(0), 5)
    true_unitary = np.kron(np.eye(2 ** 4), mat.H)
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(RX(0.2, 3), 5)
    true_unitary = np.kron(np.eye(2 ** 1), np.kron(mat.RX(0.2), np.eye(2 ** 3)))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(RX(0.5, 4), 5)
    true_unitary = np.kron(np.eye(2 ** 0), np.kron(mat.RX(0.5), np.eye(2 ** 4)))
    assert np.allclose(test_unitary, true_unitary)


def test_lifted_gate_two_qubit():
    test_unitary = lifted_gate(CNOT(0, 1), 4)
    true_unitary = lifted_gate_matrix(mat.CNOT, [0, 1], 4)
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(CNOT(1, 0), 4)
    true_unitary = lifted_gate_matrix(mat.CNOT, [1, 0], 4)
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(CNOT(1, 3), 4)
    true_unitary = lifted_gate_matrix(mat.CNOT, [1, 3], 4)
    assert np.allclose(test_unitary, true_unitary)


def test_lifted_gate_modified():
    test_unitary = lifted_gate(RZ(np.pi / 4, 0).dagger(), 1)
    true_unitary = mat.RZ(-np.pi / 4)
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(X(0).dagger().controlled(1), 2)
    true_unitary = lifted_gate(CNOT(1, 0), 2)
    other_true = mat.CNOT
    assert np.allclose(test_unitary, true_unitary)
    assert np.allclose(other_true, true_unitary)

    test_unitary = lifted_gate(X(1).dagger().controlled(0).dagger().dagger(), 2)
    true_unitary = lifted_gate(CNOT(0, 1), 2)
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(X(0).dagger().controlled(1).dagger().dagger().controlled(2), 3)
    true_unitary = lifted_gate(CCNOT(1, 2, 0), 3)
    other_true = mat.CCNOT
    assert np.allclose(test_unitary, true_unitary)
    assert np.allclose(other_true, true_unitary)

    test_unitary = lifted_gate(RY(np.pi / 4, 0).dagger().controlled(2).dagger().dagger(), 3)
    ry_part = lifted_gate(RY(-np.pi / 4, 0), 1)
    zero = np.eye(2)
    zero[1, 1] = 0
    one = np.eye(2)
    one[0, 0] = 0
    true_unitary = np.kron(zero, np.eye(4)) + np.kron(one, np.kron(np.eye(2), ry_part))
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(PHASE(0.0, 1).forked(0, [np.pi]), 2)
    true_unitary = lifted_gate(CZ(0, 1), 2)
    assert np.allclose(test_unitary, true_unitary)

    test_unitary = lifted_gate(PHASE(0.0, 2).forked(1, [0.0]).forked(0, [0.0, np.pi]), 3)
    true_unitary = lifted_gate(CZ(1, 2).controlled(0), 3)
    assert np.allclose(test_unitary, true_unitary)


def test_lifted_pauli():
    qubits = [0, 1]
    xy_term = sX(0) * sY(1)

    # test correctness
    trial_matrix = lifted_pauli(xy_term, qubits)
    true_matrix = np.kron(mat.Y, mat.X)
    np.testing.assert_allclose(trial_matrix, true_matrix)

    x1_term = sX(1)
    trial_matrix = lifted_pauli(x1_term, qubits)
    true_matrix = np.kron(mat.X, mat.I)
    np.testing.assert_allclose(trial_matrix, true_matrix)

    zpz_term = sZ(0) + sZ(1)
    trial_matrix = lifted_pauli(zpz_term, qubits)
    true_matrix = np.zeros((4, 4))
    true_matrix[0, 0] = 2
    true_matrix[-1, -1] = -2
    np.testing.assert_allclose(trial_matrix, true_matrix)


def test_lifted_state_operator():
    xz_state = plusX(5) * minusZ(6)

    plus = np.array([1, 1]) / np.sqrt(2)
    plus = plus[:, np.newaxis]
    proj_plus = plus @ plus.conj().T
    assert proj_plus.shape == (2, 2)

    one = np.array([0, 1])
    one = one[:, np.newaxis]
    proj_one = one @ one.conj().T
    assert proj_one.shape == (2, 2)

    np.testing.assert_allclose(np.kron(proj_one, proj_plus), lifted_state_operator(xz_state, qubits=[5, 6]))


def test_lifted_state_operator_backwards_qubits():
    xz_state = plusX(5) * minusZ(6)
    plus = np.array([1, 1]) / np.sqrt(2)
    plus = plus[:, np.newaxis]
    proj_plus = plus @ plus.conj().T
    assert proj_plus.shape == (2, 2)

    one = np.array([0, 1])
    one = one[:, np.newaxis]
    proj_one = one @ one.conj().T
    assert proj_one.shape == (2, 2)

    np.testing.assert_allclose(np.kron(proj_plus, proj_one), lifted_state_operator(xz_state, qubits=[6, 5]))


def test_lifted_gate_with_nonconstant_params():
    gate = RX(Parameter("theta"), 0)

    with pytest.raises(TypeError):
        lifted_gate(gate, 1)
