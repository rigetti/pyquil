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
from typing import Any, Optional, Union, cast

import numpy as np
from numpy.random.mtrand import RandomState

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES

# The following function is lovingly copied from the Cirq project
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
from pyquil.simulation.tools import all_bitstrings


def targeted_einsum(gate: np.ndarray, wf: np.ndarray, wf_target_inds: list[int]) -> np.ndarray:
    """Left-multiplies the given axes of the wf tensor by the given gate matrix.

    Note that the matrix must have a compatible tensor structure.
    For example, if you have an 6-qubit state vector ``wf`` with shape
    (2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation ``op`` with shape
    (2, 2, 2, 2), and you want to apply ``op`` to the 5th and 3rd qubits
    within ``input_state``, then the output state vector is computed as follows::

        output_state = targeted_einsum(op, input_state, [5, 3])

    This method also works when the right hand side is a matrix instead of a
    vector. If a unitary circuit's matrix is ``old_effect``, and you append
    a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
    at offset 1 and the target q4 is the qubit at offset 4, then the appended
    circuit's unitary matrix is computed as follows::

        new_effect = targeted_left_multiply(CNOT.reshape((2, 2, 2, 2)), old_effect, [1, 4])

    :param gate: What to left-multiply the target tensor by.
    :param wf: A tensor to carefully broadcast a left-multiply over.
    :param wf_target_inds: Which axes of the target are being operated on.
    :returns: The output tensor.
    """
    k = len(wf_target_inds)
    d = len(wf.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in wf_target_inds)
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, wf_target_inds):
        output_indices[t] = w

    # TODO: `out` does not work if input matrices share memory with outputs, as is usually
    # TODO: the case when propagating a wavefunction. This might be fixed in numpy 1.15
    # https://github.com/numpy/numpy/pull/11286
    # It might be worth re-investigating memory savings with `out` when numpy 1.15 becomes
    # commonplace.

    return np.einsum(gate, input_indices, wf, data_indices, output_indices)  # type: ignore


def targeted_tensordot(gate: np.ndarray, wf: np.ndarray, wf_target_inds: Sequence[int]) -> np.ndarray:
    """Left-multiplies the given axes of the wf tensor by the given gate matrix.

    Compare with :py:func:`targeted_einsum`. The semantics of these two functions should be
    identical, except this uses ``np.tensordot`` instead of ``np.einsum``.

    :param gate: What to left-multiply the target tensor by.
    :param wf: A tensor to carefully broadcast a left-multiply over.
    :param wf_target_inds: Which axes of the target are being operated on.
    :returns: The output tensor.
    """
    gate_n_qubits = gate.ndim // 2
    n_qubits = wf.ndim

    # the indices we want to sum over are the final half
    gate_inds = np.arange(gate_n_qubits, 2 * gate_n_qubits)
    if len(wf_target_inds) != len(gate_inds):
        raise ValueError(f"Length mismatch: wf_target_inds={wf_target_inds}, gate_inds={gate_inds}")
    wf = np.tensordot(gate, wf, (gate_inds, wf_target_inds))

    # tensordot dumps "output" indices into 0, 1, .. gate_n_qubits
    # we need to move them to the right place.

    # First create a list of all the "unaffected" indices which is everything but the
    # first `gate_n_qubits`
    axes_ordering = list(range(gate_n_qubits, n_qubits))

    # We want to "insert" the affected indices into the right place. This means
    # we have to be extra careful about calling list.insert in the correct order.
    # Namely, we have to insert low target indices first.
    where_td_put_them = np.arange(gate_n_qubits)
    sorty = np.argsort(wf_target_inds)
    where_td_put_them = where_td_put_them[sorty]
    sorted_targets = np.asarray(wf_target_inds)[sorty]
    # now that everything is sorted, we can do the insertion.
    for target_ind, from_ind in zip(sorted_targets, where_td_put_them):
        axes_ordering.insert(target_ind, from_ind)

    # A quick call to transpose gives us the right thing.
    return wf.transpose(axes_ordering)


def get_measure_probabilities(wf: np.ndarray, qubit: int) -> np.ndarray:
    """Get the probabilities of measuring a qubit.

    :param wf: The statevector with a dimension for each qubit
    :param qubit: The qubit to measure. We will sum over every axis except this one.
    :return: A vector of classical probabilities.
    """
    n_qubits = len(wf.shape)
    all_inds = list(range(n_qubits))

    return np.einsum(np.conj(wf), all_inds, wf, all_inds, [int(qubit)])  # type: ignore


def _get_gate_tensor_and_qubits(gate: Gate) -> tuple[np.ndarray, list[int]]:
    """Given a gate ``Instruction``, turn it into a matrix and extract qubit indices.

    :param gate: the instruction
    :return: tensor, qubit_inds.
    """
    if len(gate.params) > 0:
        matrix = QUANTUM_GATES[gate.name](*gate.params)
    else:
        matrix = QUANTUM_GATES[gate.name]

    qubit_inds = gate.get_qubit_indices()

    # e.g. 2-qubit matrix is 4x4; turns into (2,2,2,2) tensor.
    tensor = np.reshape(matrix, (2,) * len(qubit_inds) * 2)

    return tensor, qubit_inds


def _term_expectation(wf: np.ndarray, term: PauliTerm) -> Any:
    # Computes <psi|XYZ..XXZ|psi>
    wf2 = wf
    for qubit_i, op_str in term._ops.items():
        if not isinstance(qubit_i, int):
            raise ValueError("Only PauliTerms with integer qubits are supported.")
        # Re-use QUANTUM_GATES since it has X, Y, Z
        op_mat = QUANTUM_GATES[op_str]
        wf2 = targeted_tensordot(gate=op_mat, wf=wf2, wf_target_inds=[qubit_i])

    # `wf2` is XYZ..XXZ|psi>
    # hit it with a <psi| i.e. `wf.dag`
    return cast(complex, term.coefficient) * np.tensordot(wf.conj(), wf2, axes=len(wf.shape))


class NumpyWavefunctionSimulator(AbstractQuantumSimulator):
    def __init__(self, n_qubits: int, rs: Optional[RandomState] = None):
        """Initialize a wavefunction simulator that uses numpy's tensordot or einsum to update a state vector.

        Please consider using
        :py:class:`PyQVM(..., quantum_simulator_type=NumpyWavefunctionSimulator)` rather
        than using this class directly.

        This class uses a n_qubit-dim ndarray to store wavefunction
        amplitudes. The array is indexed into with a tuple of n_qubits 1's and 0's, with
        qubit 0 as the leftmost bit. This is the opposite convention of the Rigetti Lisp QVM.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (should be shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic. A value of ``None`` disallows doing anything stochastic.
        """
        self.n_qubits = n_qubits
        self.rs = rs

        self.wf = np.zeros((2,) * n_qubits, dtype=np.complex128)
        self.wf[(0,) * n_qubits] = complex(1.0, 0)

    def sample_bitstrings(self, n_samples: int) -> np.ndarray:
        """Sample bitstrings from the distribution defined by the wavefunction.

        Qubit 0 is at ``out[:, 0]``.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        """
        if self.rs is None:
            raise ValueError(
                "You have tried to perform a stochastic operation without setting the "
                "random state of the simulator. Might I suggest using a PyQVM object?"
            )

        # note on reshape: it puts bitstrings in lexicographical order.
        # would you look at that .. _all_bitstrings returns things in lexicographical order!
        # reminder: qubit 0 is on the left in einsum simulator.
        probabilities = np.abs(self.wf.reshape(-1)) ** 2
        possible_bitstrings = all_bitstrings(self.n_qubits)
        inds = self.rs.choice(2**self.n_qubits, n_samples, p=probabilities)
        return possible_bitstrings[inds, :]

    def do_measurement(self, qubit: int) -> int:
        """Measure a qubit, collapse the wavefunction, and return the measurement result.

        :param qubit: Index of the qubit to measure.
        :return: measured bit
        """
        if self.rs is None:
            raise ValueError(
                "You have tried to perform a stochastic operation without setting the "
                "random state of the simulator. Might I suggest using a PyQVM object?"
            )

        # Get probabilities
        measurement_probs = get_measure_probabilities(self.wf, qubit)

        # Flip a coin and record the result
        measured_bit = int(np.argmax(self.rs.uniform() < np.cumsum(measurement_probs)))

        # Zero out amplitudes corresponding to non-measured bistrings
        other_bit = (measured_bit + 1) % 2
        other_bit_indices = (slice(None),) * qubit + (other_bit,) + (slice(None),) * (self.n_qubits - qubit - 1)
        self.wf[other_bit_indices] = 0

        # Re-normalize amplitudes corresponding to measured bistrings
        meas_bit_indices = (slice(None),) * qubit + (measured_bit,) + (slice(None),) * (self.n_qubits - qubit - 1)
        self.wf[meas_bit_indices] /= np.sqrt(measurement_probs[measured_bit])
        return measured_bit

    def do_gate(self, gate: Gate) -> "NumpyWavefunctionSimulator":
        """Perform a gate.

        :return: ``self`` to support method chaining.
        """
        gate_matrix, qubit_inds = _get_gate_tensor_and_qubits(gate=gate)
        # Note to developers: you can use either einsum- or tensordot- based functions.
        # tensordot seems a little faster, but feel free to experiment.
        # self.wf = targeted_einsum(gate=gate_matrix, wf=self.wf, wf_target_inds=qubit_inds)
        self.wf = targeted_tensordot(gate=gate_matrix, wf=self.wf, wf_target_inds=qubit_inds)
        return self

    def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> "NumpyWavefunctionSimulator":
        """Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """
        # e.g. 2-qubit matrix is 4x4; turns into (2,2,2,2) tensor.
        tensor = np.reshape(matrix, (2,) * len(qubits) * 2)

        # Note to developers: you can use either einsum- or tensordot- based functions.
        # tensordot seems a little faster, but feel free to experiment.
        # self.wf = targeted_einsum(gate=gate_matrix, wf=self.wf, wf_target_inds=qubits)
        self.wf = targeted_tensordot(gate=tensor, wf=self.wf, wf_target_inds=qubits)
        return self

    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> float:
        """Compute the expectation of an operator.

        :param operator: The operator
        :return: The operator's expectation value
        """
        if not isinstance(operator, PauliSum):
            operator = PauliSum([operator])

        return sum(_term_expectation(self.wf, term) for term in operator)  # type: ignore

    def reset(self) -> "NumpyWavefunctionSimulator":
        """Reset the wavefunction to the ``|000...00>`` state.

        :return: ``self`` to support method chaining.
        """
        self.wf.fill(0)
        self.wf[(0,) * self.n_qubits] = complex(1.0, 0)
        return self

    def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: list[int]) -> "AbstractQuantumSimulator":
        raise NotImplementedError("The numpy simulator cannot handle noise")
