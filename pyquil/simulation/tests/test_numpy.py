import itertools

import numpy as np
import pytest

from pyquil import Program
from pyquil.gates import CCNOT, CNOT, H, MEASURE, RX, X
from pyquil.paulis import sZ, sX
from pyquil.pyqvm import PyQVM
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare, DefGate
from pyquil.simulation._numpy import (
    targeted_einsum,
    NumpyWavefunctionSimulator,
    all_bitstrings,
    targeted_tensordot,
    _term_expectation,
)
from pyquil.simulation._reference import ReferenceWavefunctionSimulator
from pyquil.simulation.matrices import QUANTUM_GATES as GATES
from pyquil.simulation.tests.test_reference_wavefunction import (
    _generate_random_program,
    _generate_random_pauli,
)


def test_H_einsum():
    h_mat = GATES["H"]
    one_q_wfn = np.zeros((2,), dtype=np.complex128)
    one_q_wfn[0] = 1 + 0.0j
    one_q_wfn = targeted_einsum(gate=h_mat, wf=one_q_wfn, wf_target_inds=[0])
    np.testing.assert_allclose(one_q_wfn, 1 / np.sqrt(2) * np.ones(2))


def test_H_tensordot():
    h_mat = GATES["H"]
    one_q_wfn = np.zeros((2,), dtype=np.complex128)
    one_q_wfn[0] = 1 + 0.0j
    one_q_wfn = targeted_tensordot(gate=h_mat, wf=one_q_wfn, wf_target_inds=[0])
    np.testing.assert_allclose(one_q_wfn, 1 / np.sqrt(2) * np.ones(2))


def test_wfn_ordering_einsum():
    h_mat = GATES["H"]
    two_q_wfn = np.zeros((2, 2), dtype=np.complex128)
    two_q_wfn[0, 0] = 1 + 0.0j
    two_q_wfn = targeted_einsum(gate=h_mat, wf=two_q_wfn, wf_target_inds=[0])
    np.testing.assert_allclose(two_q_wfn[:, 0], 1 / np.sqrt(2) * np.ones(2))


def test_wfn_ordering_tensordot():
    h_mat = GATES["H"]
    two_q_wfn = np.zeros((2, 2), dtype=np.complex128)
    two_q_wfn[0, 0] = 1 + 0.0j
    two_q_wfn = targeted_tensordot(gate=h_mat, wf=two_q_wfn, wf_target_inds=[0])
    np.testing.assert_allclose(two_q_wfn[:, 0], 1 / np.sqrt(2) * np.ones(2))


def test_einsum_simulator_H():
    prog = Program(H(0))
    qam = PyQVM(n_qubits=1, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    np.testing.assert_allclose(wf, 1 / np.sqrt(2) * np.ones(2))


def test_einsum_simulator_1():
    prog = Program(H(0), CNOT(0, 1))
    qam = PyQVM(n_qubits=2, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    np.testing.assert_allclose(wf, 1 / np.sqrt(2) * np.reshape([1, 0, 0, 1], (2, 2)))


def test_einsum_simulator_CNOT():
    prog = Program(X(0), CNOT(0, 1))
    qam = PyQVM(n_qubits=2, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    np.testing.assert_allclose(wf, np.reshape([0, 0, 0, 1], (2, 2)))


def test_einsum_simulator_CCNOT():
    prog = Program(X(2), X(0), CCNOT(2, 1, 0))
    qam = PyQVM(n_qubits=3, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    should_be = np.zeros((2, 2, 2))
    should_be[1, 0, 1] = 1
    np.testing.assert_allclose(wf, should_be)


def test_einsum_simulator_10q():
    prog = Program(H(0))
    for i in range(10 - 1):
        prog += CNOT(i, i + 1)
    qam = PyQVM(n_qubits=10, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    should_be = np.zeros((2,) * 10)
    should_be[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1 / np.sqrt(2)
    should_be[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] = 1 / np.sqrt(2)
    np.testing.assert_allclose(wf, should_be)


def test_measure():
    qam = PyQVM(n_qubits=3, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(Program(Declare("ro", "BIT", 64), H(0), CNOT(0, 1), MEASURE(0, MemoryReference("ro", 63))))
    measured_bit = qam.ram["ro"][-1]
    should_be = np.zeros((2, 2, 2))
    if measured_bit == 1:
        should_be[1, 1, 0] = 1
    else:
        should_be[0, 0, 0] = 1

    np.testing.assert_allclose(qam.wf_simulator.wf, should_be)


@pytest.fixture(params=list(range(3, 5)))
def n_qubits(request):
    return request.param


@pytest.fixture(params=[2, 50, 100])
def prog_length(request):
    return request.param


@pytest.fixture(params=[True, False])
def include_measures(request):
    return request.param


def test_vs_ref_simulator(n_qubits, prog_length, include_measures):
    if include_measures:
        seed = 52
    else:
        seed = None

    for _ in range(10):
        prog = _generate_random_program(n_qubits=n_qubits, length=prog_length, include_measures=include_measures)
        ref_qam = PyQVM(n_qubits=n_qubits, seed=seed, quantum_simulator_type=ReferenceWavefunctionSimulator)
        ref_qam.execute(prog)
        ref_wf = ref_qam.wf_simulator.wf

        es_qam = PyQVM(n_qubits=n_qubits, seed=seed, quantum_simulator_type=NumpyWavefunctionSimulator)
        es_qam.execute(prog)
        es_wf = es_qam.wf_simulator.wf
        # einsum has its wavefunction as a vector of shape (2, 2, 2, ...) where qubits are indexed
        # from left to right. We transpose then flatten.
        es_wf = es_wf.transpose().reshape(-1)

        np.testing.assert_allclose(ref_wf, es_wf, atol=1e-15)


def test_all_bitstrings():
    for n_bits in range(2, 10):
        bitstrings_ref = np.array(list(itertools.product((0, 1), repeat=n_bits)))
        bitstrings_new = all_bitstrings(n_bits)
        np.testing.assert_array_equal(bitstrings_ref, bitstrings_new)


def test_sample_bitstrings():
    prog = Program(H(0), H(1))
    qam = PyQVM(n_qubits=3, quantum_simulator_type=NumpyWavefunctionSimulator, seed=52)
    qam.execute(prog)
    bitstrings = qam.wf_simulator.sample_bitstrings(10000)
    assert bitstrings.shape == (10000, 3)
    np.testing.assert_allclose([0.5, 0.5, 0], np.mean(bitstrings, axis=0), rtol=1e-2)


def test_expectation_helper():
    n_qubits = 3
    wf = np.zeros(shape=((2,) * n_qubits), dtype=np.complex)
    wf[0, 0, 0] = 1
    z0 = _term_expectation(wf, 0.4 * sZ(0))
    assert z0 == 0.4

    x0 = _term_expectation(wf, sX(2))
    assert x0 == 0


def test_expectation():
    wfn = NumpyWavefunctionSimulator(n_qubits=3)
    val = wfn.expectation(0.4 * sZ(0) + sX(2))
    assert val == 0.4


def test_expectation_vs_ref_qvm(n_qubits):
    for _ in range(20):
        prog = _generate_random_program(n_qubits=n_qubits, length=10)
        operator = _generate_random_pauli(n_qubits=n_qubits, n_terms=5)
        print(prog)
        print(operator)

        ref_wf = ReferenceWavefunctionSimulator(n_qubits=n_qubits).do_program(prog)
        ref_exp = ref_wf.expectation(operator=operator)

        np_wf = NumpyWavefunctionSimulator(n_qubits=n_qubits).do_program(prog)
        np_exp = np_wf.expectation(operator=operator)
        np.testing.assert_allclose(ref_exp, np_exp, atol=1e-15)


def test_defgate():
    # regression test for https://github.com/rigetti/pyquil/issues/1059
    theta = np.pi / 2
    U = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )

    gate_definition = DefGate("U_test", U)
    U_test = gate_definition.get_constructor()

    p = Program()
    p += gate_definition
    p += X(1)
    p += U_test(1, 0)
    qam = PyQVM(n_qubits=2, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(p)
    wf1 = qam.wf_simulator.wf
    should_be = np.zeros((2, 2), dtype=np.complex128)
    one_over_sqrt2 = 1 / np.sqrt(2)
    should_be[0, 1] = one_over_sqrt2
    should_be[1, 1] = -1j * one_over_sqrt2
    np.testing.assert_allclose(wf1, should_be)

    # Ensure the output of the custom U_test gate matches the standard RX gate. Something like
    # RX(theta, 0).controlled(1) would be a more faithful reproduction of U_test, but
    # NumpyWavefunctionSimulator doesn't (yet) support gate modifiers, so just apply the RX gate
    # unconditionally.
    p = Program()
    p += X(1)
    p += RX(theta, 0)
    qam = PyQVM(n_qubits=2, quantum_simulator_type=NumpyWavefunctionSimulator)
    qam.execute(p)
    wf2 = qam.wf_simulator.wf
    np.testing.assert_allclose(wf1, wf2)


# The following tests are lovingly copied with light modification from the Cirq project
# https://github.com/quantumlib/Cirq
#
# With the original copyright disclaimer:

# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def kron(*matrices: np.ndarray) -> np.ndarray:
    """Computes the kronecker product of a sequence of matrices.

    A *args version of lambda args: functools.reduce(np.kron, args).

    Args:
        *matrices: The matrices and controls to combine with the kronecker
            product.

    Returns:
        The resulting matrix.
    """
    product = np.eye(1)
    for m in matrices:
        product = np.kron(product, m)
    return np.array(product)


def test_einsum_matches_kron_then_dot():
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = np.array([[2, 3], [5, 7]])
    i = np.eye(2)

    np.testing.assert_allclose(
        targeted_einsum(gate=m, wf=t.reshape((2, 2, 2)), wf_target_inds=[0]),
        np.dot(kron(m, i, i), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        targeted_einsum(gate=m, wf=t.reshape((2, 2, 2)), wf_target_inds=[1]),
        np.dot(kron(i, m, i), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        targeted_einsum(gate=m, wf=t.reshape((2, 2, 2)), wf_target_inds=[2]),
        np.dot(kron(i, i, m), t).reshape((2, 2, 2)),
        atol=1e-8,
    )


def test_tensordot_matches_kron_then_dot():
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = np.array([[2, 3], [5, 7]])
    i = np.eye(2)

    np.testing.assert_allclose(
        targeted_tensordot(m, t.reshape((2, 2, 2)), [0]),
        np.dot(kron(m, i, i), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        targeted_tensordot(m, t.reshape((2, 2, 2)), [1]),
        np.dot(kron(i, m, i), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        targeted_tensordot(m, t.reshape((2, 2, 2)), [2]),
        np.dot(kron(i, i, m), t).reshape((2, 2, 2)),
        atol=1e-8,
    )


def test_einsum_reorders_matrices():
    t = np.eye(4).reshape((2, 2, 2, 2))
    m = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]).reshape((2, 2, 2, 2))

    np.testing.assert_allclose(targeted_einsum(gate=m, wf=t, wf_target_inds=[0, 1]), m, atol=1e-8)

    np.testing.assert_allclose(
        targeted_einsum(gate=m, wf=t, wf_target_inds=[1, 0]),
        np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]).reshape((2, 2, 2, 2)),
        atol=1e-8,
    )


def test_tensordot_reorders_matrices():
    t = np.eye(4).reshape((2, 2, 2, 2))
    m = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]).reshape((2, 2, 2, 2))

    np.testing.assert_allclose(targeted_tensordot(gate=m, wf=t, wf_target_inds=[0, 1]), m, atol=1e-8)

    np.testing.assert_allclose(
        targeted_tensordot(gate=m, wf=t, wf_target_inds=[1, 0]),
        np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]).reshape((2, 2, 2, 2)),
        atol=1e-8,
    )
