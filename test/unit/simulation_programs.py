##############################################################################
# Copyright 2026 Rigetti Computing
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
"""Shared program corpus and helpers for the quax-based simulator tests.

``test_state_vector.py``, ``test_density_matrix.py`` and ``test_qutrit_simulation.py`` all
exercise the same simulators; this module holds what they share so a new case is added once
and checked by every backend.

.. note::
    The simulators are **big-endian**: ``qubits[0]`` is the most significant subsystem, so
    ``X 0`` on a two-qubit register gives ``|10>``. The rest of pyQuil is little-endian; use
    :func:`reverse_endianness` before comparing against the reference simulators.
"""

import jax.numpy as jnp
import numpy as np

from pyquil.gates import CCNOT, CNOT, CSWAP, CZ, RX, RY, RZ, SWAP, H, I, S, T, X, Y, Z
from pyquil.quil import Program
from pyquil.simulation._simulator import DensityMatrixSimulator, PureStateVectorSimulator

EMPTY_PARAMS = jnp.array([], dtype=float)

# Gate-only programs reused across the noiseless equivalence tests. Deliberately includes gates
# whose operands are *not* in ascending order -- the compressor advertises a sorted subsystem for
# merged groups but an unmerged operation keeps its own operand order, and getting that wrong
# silently permutes qudits in the density-matrix path only.
PROGRAMS = {
    "empty": Program(),
    "single_x": Program(X(0)),
    "hadamard": Program(H(0)),
    "bell": Program(H(0), CNOT(0, 1)),
    "ghz3": Program(H(0), CNOT(0, 1), CNOT(1, 2)),
    "clifford_chain": Program(H(0), S(0), T(0), H(0), Z(0), Y(0)),
    "rotations": Program(RX(0.3, 0), RY(0.7, 0), RZ(1.1, 0)),
    "two_qubit_mixed": Program(RX(0.4, 0), RY(0.9, 1), CNOT(0, 1), RZ(1.1, 0), CZ(0, 1)),
    "ccnot_sorted": Program(H(0), H(1), CCNOT(0, 1, 2)),
    "ccnot_reversed": Program(X(1), X(2), CCNOT(2, 1, 0)),
    "cnot_reversed": Program(X(1), CNOT(1, 0)),
    "cswap_reversed": Program(X(0), X(2), CSWAP(2, 1, 0)),
    "swap_reversed": Program(X(1), SWAP(2, 0)),
    "deep": Program(RX(0.4, 0), RY(0.9, 1), CNOT(0, 1), RZ(1.1, 0), CZ(1, 2), RX(0.2, 2), SWAP(0, 2)),
    "idle_qubit": Program(X(0), I(1)),
}

# The subset of PROGRAMS cross-checked against pyQuil's NumPy reference simulators.
COMPARABLE = [
    "single_x",
    "hadamard",
    "bell",
    "ghz3",
    "clifford_chain",
    "rotations",
    "two_qubit_mixed",
    "ccnot_sorted",
    "ccnot_reversed",
    "cnot_reversed",
    "cswap_reversed",
    "deep",
]


def simulate_state_vector(program, qubits=None, memory_map=None, **kwargs):
    """Run ``PureStateVectorSimulator`` and return the final ``qx.StateVector``."""
    sim = PureStateVectorSimulator(program, qubits=qubits, **kwargs)
    params = sim.linearize(memory_map) if memory_map else EMPTY_PARAMS
    return sim.compute(params)


def simulate_density_matrix(program, qubits=None, noise_model=None, memory_map=None, **kwargs):
    """Run ``DensityMatrixSimulator`` and return the final ``qx.DensityMatrix``."""
    sim = DensityMatrixSimulator(program, qubits=qubits, noise_model=noise_model, **kwargs)
    params = sim.linearize(memory_map) if memory_map else EMPTY_PARAMS
    return sim.compute(params)


def reverse_endianness(array, n_qubits):
    """Reverse the qubit order of a big-endian array so it can be compared with pyQuil's little-endian ones.

    Works for a state vector (1-D) or a density matrix (2-D) over ``n_qubits`` qubits.
    """
    axes = tuple(reversed(range(n_qubits)))
    if array.ndim == 1:
        return array.reshape((2,) * n_qubits).transpose(axes).reshape(-1)
    tensor = array.reshape((2,) * (2 * n_qubits))
    perm = axes + tuple(n + n_qubits for n in axes)
    return tensor.transpose(perm).reshape(2**n_qubits, 2**n_qubits)


def assert_pure(rho, psi, atol=1e-10):
    """Assert that density matrix ``rho`` equals ``|psi><psi|``."""
    np.testing.assert_allclose(rho, np.outer(psi, np.conj(psi)), atol=atol)


def assert_physical(rho, atol=1e-9):
    """Assert that ``rho`` is a valid density matrix: unit trace, Hermitian, positive semi-definite."""
    assert abs(np.trace(rho).real - 1.0) < atol
    np.testing.assert_allclose(rho, rho.conj().T, atol=atol)
    assert np.linalg.eigvalsh(rho).min() > -atol
