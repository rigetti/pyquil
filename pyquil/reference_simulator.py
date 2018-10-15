import warnings

import numpy as np
from numpy.random.mtrand import RandomState

from pyquil.gate_matrices import P0, P1, KRAUS_OPS
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.unitary_tools import lifted_gate_matrix, lifted_gate, _all_bitstrings


class ReferenceWavefunctionSimulator(AbstractQuantumSimulator):
    def __init__(self, n_qubits: int, rs: RandomState):
        """
        A wavefunction simulator that prioritizes readability over performance.

        Please consider using
        :py:class:`PyQVM(..., wf_simulator_type=ReferenceWavefunctionSimulator)` rather
        than using this class directly.

        This class uses a flat state-vector of length 2^n_qubits to store wavefunction
        amplitudes. The basis is taken to be bitstrings ordered lexicographically with
        qubit 0 as the rightmost bit. This is the same as the Rigetti Lisp QVM.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic.
        """
        self.n_qubits = n_qubits
        self.rs = rs

        self.wf = np.zeros(2 ** n_qubits, dtype=np.complex128)
        self.wf[0] = complex(1.0, 0)

    def do_gate(self, gate: Gate):
        """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """
        unitary = lifted_gate(gate=gate, n_qubits=self.n_qubits)
        self.wf = unitary.dot(self.wf)
        return self

    def do_measurement(self, qubit: int) -> int:
        """
        Measure a qubit, collapse the wavefunction, and return the measurement result.

        :param qubit: Index of the qubit to measure.
        :return: measured bit
        """
        # lift projective measure operator to Hilbert space
        # prob(0) = <psi P0 | P0 psi> = psi* . P0* . P0 . psi
        measure_0 = lifted_gate_matrix(matrix=P0, qubit_inds=[qubit], n_qubits=self.n_qubits)
        proj_psi = measure_0 @ self.wf
        prob_zero = np.conj(proj_psi).T @ proj_psi

        # generate random number to 'roll' for measure
        if self.rs.uniform() < prob_zero:
            # decohere state using the measure_0 operator
            unitary = measure_0 @ (np.eye(2 ** self.n_qubits) / np.sqrt(prob_zero))
            self.wf = unitary.dot(self.wf)
            return 0
        else:  # measure one
            measure_1 = lifted_gate_matrix(matrix=P1, qubit_inds=[qubit], n_qubits=self.n_qubits)
            unitary = measure_1 @ (np.eye(2 ** self.n_qubits) / np.sqrt(1 - prob_zero))
            self.wf = unitary.dot(self.wf)
            return 1

    def reset(self):
        """
        Reset the wavefunction to the |000...00> state.

        :return: ``self`` to support method chaining.
        """
        self.wf.fill(0)
        self.wf[0] = complex(1.0, 0)
        return self

    def do_post_gate_noise(self, noise_type: str, noise_prob: float):
        raise NotImplementedError("Reference simulator can't do noise.")


class ReferenceDensitySimulator(AbstractQuantumSimulator):
    def __init__(self, n_qubits: int, rs: RandomState):
        self.n_qubits = n_qubits
        self.rs = rs
        self.density = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=np.complex128)
        self.density[0, 0] = complex(1.0, 0)

    def sample_bitstrings(self, n_samples):
        """
        Sample bitstrings from the distribution defined by the wavefunction.

        Qubit 0 is at ``out[:, 0]``.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        """
        # TODO: port to AbstractQuantumSimulator abstraction
        probabilities = np.real_if_close(np.diagonal(self.density))
        possible_bitstrings = _all_bitstrings(self.n_qubits)
        inds = self.rs.choice(2 ** self.n_qubits, n_samples, p=probabilities)
        bitstrings = possible_bitstrings[inds, :]
        bitstrings = np.flip(bitstrings, axis=1)  # qubit ordering: 0 on the left.
        return bitstrings

    def do_gate(self, gate: Gate) -> 'AbstractQuantumSimulator':
        unitary = lifted_gate(gate=gate, n_qubits=self.n_qubits)
        self.density = unitary.dot(self.density).dot(np.conj(unitary).T)
        return self

    def do_measurement(self, qubit: int) -> int:
        measure_0 = lifted_gate_matrix(matrix=P0, qubit_inds=[qubit], n_qubits=self.n_qubits)
        prob_zero = np.trace(measure_0 @ self.density)

        # generate random number to 'roll' for measurement
        if self.rs.uniform() < prob_zero:
            # decohere state using the measure_0 operator
            unitary = measure_0 @ (np.eye(2 ** self.n_qubits) / np.sqrt(prob_zero))
            self.density = unitary.dot(self.density).dot(np.conj(unitary.T))
            return 0
        else:  # measure one
            measure_1 = lifted_gate_matrix(matrix=P1, qubit_inds=[qubit], n_qubits=self.n_qubits)
            unitary = measure_1 @ (np.eye(2 ** self.n_qubits) / np.sqrt(1 - prob_zero))
            self.density = unitary.dot(self.density).dot(np.conj(unitary.T))
            return 1

    def reset(self) -> 'AbstractQuantumSimulator':
        self.density.fill(0)
        self.density[0, 0] = complex(1.0, 0)
        return self

    def do_post_gate_noise(self, noise_type: str, noise_prob: float):
        kraus_ops = KRAUS_OPS[noise_type](p=noise_prob)
        if np.isclose(noise_prob, 1.0):
            warnings.warn(f"Skipping {noise_type} post-gate noise because noise_prob is close to 1")

        for q in range(self.n_qubits):
            new_density = np.zeros_like(self.density)
            for kraus_op in kraus_ops:
                lifted_kraus_op = lifted_gate_matrix(matrix=kraus_op, qubit_inds=[q],
                                                     n_qubits=self.n_qubits)
                new_density += lifted_kraus_op.dot(self.density).dot(np.conj(lifted_kraus_op.T))
            self.density = new_density
