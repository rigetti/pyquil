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
import warnings
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from numpy.random.mtrand import RandomState

from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import P0, P1, KRAUS_OPS, QUANTUM_GATES
from pyquil.simulation.tools import lifted_gate_matrix, lifted_gate, all_bitstrings


def _term_expectation(wf: np.ndarray, term: PauliTerm, n_qubits: int) -> Any:
    # Computes <psi|XYZ..XXZ|psi>
    wf2 = wf
    for qubit_i, op_str in term._ops.items():
        assert isinstance(qubit_i, int)
        # Re-use QUANTUM_GATES since it has X, Y, Z
        op_mat = QUANTUM_GATES[op_str]
        op_mat = lifted_gate_matrix(matrix=op_mat, qubit_inds=[qubit_i], n_qubits=n_qubits)
        wf2 = op_mat @ wf2

    # `wf2` is XYZ..XXZ|psi>
    # hit it with a <psi| i.e. `wf.dag`
    return term.coefficient * (wf.conj().T @ wf2)


def _is_valid_quantum_state(state_matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if a quantum state is valid, i.e. the matrix is Hermitian; trace one, and that the
    eigenvalues are non-negative.

    :param state_matrix: a D by D np.ndarray representing a quantum state
    :param rtol: The relative tolerance parameter in np.allclose and np.isclose
    :param atol: The absolute tolerance parameter in np.allclose and np.isclose
    :return: bool
    """
    hermitian = np.allclose(state_matrix, np.conjugate(state_matrix.transpose()), rtol, atol)
    if not hermitian:
        raise ValueError("The state matrix is not Hermitian.")
    trace_one = np.isclose(np.trace(state_matrix), 1, rtol, atol)
    if not trace_one:
        raise ValueError("The state matrix is not trace one.")
    evals = np.linalg.eigvals(state_matrix)  # type: ignore
    non_neg_eigs = all([False if val < -atol else True for val in evals])
    if not non_neg_eigs:
        raise ValueError("The state matrix has negative Eigenvalues of order -" + str(atol) + ".")
    return hermitian and trace_one and non_neg_eigs


class ReferenceWavefunctionSimulator(AbstractQuantumSimulator):
    def __init__(self, n_qubits: int, rs: Optional[RandomState] = None):
        """
        A wavefunction simulator that prioritizes readability over performance.

        Please consider using
        :py:class:`PyQVM(..., wf_simulator_type=ReferenceWavefunctionSimulator)` rather
        than using this class directly.

        This class uses a flat state-vector of length 2^n_qubits to store wavefunction
        amplitudes. The basis is taken to be bitstrings ordered lexicographically with
        qubit 0 as the rightmost bit. This is the same as the Rigetti Lisp QVM.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (should be shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic. A value of ``None`` disallows doing anything stochastic.
        """
        super().__init__(n_qubits=n_qubits, rs=rs)  # type: ignore

        self.n_qubits = n_qubits
        self.rs = rs

        self.wf = np.zeros(2**n_qubits, dtype=np.complex128)
        self.wf[0] = complex(1.0, 0)

    def sample_bitstrings(self, n_samples: int) -> np.ndarray:
        """
        Sample bitstrings from the distribution defined by the wavefunction.

        Qubit 0 is at ``out[:, 0]``.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        """
        if self.rs is None:
            raise ValueError(
                "You have tried to perform a stochastic operation without setting the "
                "random state of the simulator. Might I suggest using a PyQVM object?"
            )
        probabilities = np.abs(self.wf) ** 2
        possible_bitstrings = all_bitstrings(self.n_qubits)
        inds = self.rs.choice(2**self.n_qubits, n_samples, p=probabilities)
        bitstrings = possible_bitstrings[inds, :]
        bitstrings = np.flip(bitstrings, axis=1)  # type: ignore # qubit ordering: 0 on the left.
        return bitstrings  # type: ignore

    def do_gate(self, gate: Gate) -> "ReferenceWavefunctionSimulator":
        """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """
        unitary = lifted_gate(gate=gate, n_qubits=self.n_qubits)
        self.wf = unitary.dot(self.wf)
        return self

    def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> "ReferenceWavefunctionSimulator":
        """
        Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done.
        :param qubits: The qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """
        unitary = lifted_gate_matrix(matrix, list(qubits), n_qubits=self.n_qubits)
        self.wf = unitary.dot(self.wf)
        return self

    def do_measurement(self, qubit: int) -> int:
        """
        Measure a qubit, collapse the wavefunction, and return the measurement result.

        :param qubit: Index of the qubit to measure.
        :return: measured bit
        """
        if self.rs is None:
            raise ValueError(
                "You have tried to perform a stochastic operation without setting the "
                "random state of the simulator. Might I suggest using a PyQVM object?"
            )
        # lift projective measure operator to Hilbert space
        # prob(0) = <psi P0 | P0 psi> = psi* . P0* . P0 . psi
        measure_0 = lifted_gate_matrix(matrix=P0, qubit_inds=[qubit], n_qubits=self.n_qubits)
        proj_psi = measure_0 @ self.wf
        prob_zero = np.conj(proj_psi).T @ proj_psi

        # generate random number to 'roll' for measure
        if self.rs.uniform() < prob_zero:
            # decohere state using the measure_0 operator
            unitary = measure_0 @ (np.eye(2**self.n_qubits) / np.sqrt(prob_zero))
            self.wf = unitary.dot(self.wf)
            return 0
        else:  # measure one
            measure_1 = lifted_gate_matrix(matrix=P1, qubit_inds=[qubit], n_qubits=self.n_qubits)
            unitary = measure_1 @ (np.eye(2**self.n_qubits) / np.sqrt(1 - prob_zero))
            self.wf = unitary.dot(self.wf)
            return 1

    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> float:
        """
        Compute the expectation of an operator.

        :param operator: The operator
        :return: The operator's expectation value
        """
        if not isinstance(operator, PauliSum):
            operator = PauliSum([operator])

        return sum(_term_expectation(self.wf, term, n_qubits=self.n_qubits) for term in operator)  # type: ignore

    def reset(self) -> "ReferenceWavefunctionSimulator":
        """
        Reset the wavefunction to the ``|000...00>`` state.

        :return: ``self`` to support method chaining.
        """
        self.wf.fill(0)
        self.wf[0] = complex(1.0, 0)
        return self

    def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: List[int]) -> "AbstractQuantumSimulator":
        raise NotImplementedError("The reference wavefunction simulator cannot handle noise")


def zero_state_matrix(n_qubits: int) -> np.ndarray:
    """
    Construct a matrix corresponding to the tensor product of `n` ground states ``|0><0|``.

    :param n_qubits: The number of qubits.
    :return: The state matrix  ``|000...0><000...0|`` for `n_qubits`.
    """
    state_matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    state_matrix[0, 0] = complex(1.0, 0)
    return state_matrix


class ReferenceDensitySimulator(AbstractQuantumSimulator):
    """
    A density matrix simulator that prioritizes readability over performance.

    Please consider using
    :py:class:`PyQVM(..., wf_simulator_type=ReferenceDensitySimulator)` rather
    than using this class directly.

    This class uses a dense matrix of shape ``(2^n_qubits, 2^n_qubits)`` to store the
    density matrix.

    :param n_qubits: Number of qubits to simulate.
    :param rs: a RandomState (should be shared with the owning :py:class:`PyQVM`) for
        doing anything stochastic. A value of ``None`` disallows doing anything stochastic.
    """

    def __init__(self, n_qubits: int, rs: Optional[RandomState] = None):
        super().__init__(n_qubits=n_qubits, rs=rs)  # type: ignore

        self.n_qubits = n_qubits
        self.rs = rs
        self.density: Optional[np.ndarray] = None
        self.set_initial_state(zero_state_matrix(n_qubits)).reset()

    def set_initial_state(self, state_matrix: np.ndarray) -> "ReferenceDensitySimulator":
        """
        This method is the correct way (TM) to update the initial state matrix that is
        initialized every time reset() is called. The default initial state of
        ReferenceDensitySimulator is ``|000...00>``.

        Note that the current state matrix, i.e. ``self.density`` is not affected by this
        method; you must change it directly or else call reset() after calling this method.

        To restore default state initialization behavior of ReferenceDensitySimulator pass in
        a ``state_matrix`` equal to the default initial state on `n_qubits` (i.e. ``|000...00>``)
        and then call ``reset()``. We have provided a helper function ``n_qubit_zero_state``
        in the ``_reference.py`` module to simplify this step.

        :param state_matrix: numpy.ndarray or None.
        :return: ``self`` to support method chaining.
        """
        rows, cols = state_matrix.shape
        if rows != cols:
            raise ValueError("The state matrix is not square.")
        if self.n_qubits != int(np.log2(rows)):
            raise ValueError("The state matrix is not defined on the same numbers of qubits as the QVM.")
        if _is_valid_quantum_state(state_matrix):
            self.initial_density = state_matrix
        else:
            raise ValueError(
                "The state matrix is not valid. It must be Hermitian, trace one, " "and have non-negative eigenvalues."
            )
        return self

    def sample_bitstrings(self, n_samples: int, tol_factor: float = 1e8) -> np.ndarray:
        """
        Sample bitstrings from the distribution defined by the wavefunction.

        Qubit 0 is at ``out[:, 0]``.

        :param n_samples: The number of bitstrings to sample
        :param tol_factor: Tolerance to set imaginary probabilities to zero, relative to
            machine epsilon.
        :return: An array of shape (n_samples, n_qubits)
        """
        if self.rs is None:
            raise ValueError(
                "You have tried to perform a stochastic operation without setting the "
                "random state of the simulator. Might I suggest using a PyQVM object?"
            )

        # for np.real_if_close the actual tolerance is (machine_eps * tol_factor),
        # where `machine_epsilon = np.finfo(float).eps`. If we use tol_factor = 1e8, then the
        # overall tolerance is \approx 2.2e-8.
        probabilities = np.real_if_close(np.diagonal(self.density), tol=tol_factor)  # type: ignore
        # Next set negative probabilities to zero
        probabilities = np.array([0 if p < 0.0 else p for p in probabilities])
        # Ensure they sum to one
        probabilities = probabilities / np.sum(probabilities)
        possible_bitstrings = all_bitstrings(self.n_qubits)
        inds = self.rs.choice(2**self.n_qubits, n_samples, p=probabilities)
        bitstrings = possible_bitstrings[inds, :]
        bitstrings = np.flip(bitstrings, axis=1)  # type: ignore  # qubit ordering: 0 on the left.
        return bitstrings  # type: ignore

    def do_gate(self, gate: Gate) -> "AbstractQuantumSimulator":
        """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """
        unitary = lifted_gate(gate=gate, n_qubits=self.n_qubits)
        self.density = unitary.dot(self.density).dot(np.conj(unitary).T)  # type: ignore
        return self

    def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> "AbstractQuantumSimulator":
        """
        Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """
        unitary = lifted_gate_matrix(matrix=matrix, qubit_inds=qubits, n_qubits=self.n_qubits)
        self.density = unitary.dot(self.density).dot(np.conj(unitary).T)  # type: ignore
        return self

    def do_measurement(self, qubit: int) -> int:
        """
        Measure a qubit and collapse the wavefunction

        :return: The measurement result. A 1 or a 0.
        """
        if self.rs is None:
            raise ValueError(
                "You have tried to perform a stochastic operation without setting the "
                "random state of the simulator. Might I suggest using a PyQVM object?"
            )
        measure_0 = lifted_gate_matrix(matrix=P0, qubit_inds=[qubit], n_qubits=self.n_qubits)
        prob_zero = np.trace(measure_0 @ self.density)

        # generate random number to 'roll' for measurement
        if self.rs.uniform() < prob_zero:
            # decohere state using the measure_0 operator
            unitary = measure_0 @ (np.eye(2**self.n_qubits) / np.sqrt(prob_zero))
            self.density = unitary.dot(self.density).dot(np.conj(unitary.T))
            return 0
        else:  # measure one
            measure_1 = lifted_gate_matrix(matrix=P1, qubit_inds=[qubit], n_qubits=self.n_qubits)
            unitary = measure_1 @ (np.eye(2**self.n_qubits) / np.sqrt(1 - prob_zero))
            self.density = unitary.dot(self.density).dot(np.conj(unitary.T))
            return 1

    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> complex:
        raise NotImplementedError("To implement")

    def reset(self) -> "AbstractQuantumSimulator":
        """
        Resets the current state of ReferenceDensitySimulator ``self.density`` to
        ``self.initial_density``.

        :return: ``self`` to support method chaining.
        """
        self.density = self.initial_density
        return self

    def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: List[int]) -> "ReferenceDensitySimulator":
        kraus_ops = KRAUS_OPS[noise_type](p=noise_prob)
        if np.isclose(noise_prob, 0.0):
            warnings.warn(f"Skipping {noise_type} post-gate noise because noise_prob is close to 0")
            return self

        for q in qubits:
            new_density = np.zeros_like(self.density)  # type: ignore
            for kraus_op in kraus_ops:
                lifted_kraus_op = lifted_gate_matrix(matrix=kraus_op, qubit_inds=[q], n_qubits=self.n_qubits)
                new_density += lifted_kraus_op.dot(self.density).dot(np.conj(lifted_kraus_op.T))  # type: ignore
            self.density = new_density
        return self
