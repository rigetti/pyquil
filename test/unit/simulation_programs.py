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

Three collections are exported:

* :data:`PROGRAMS` -- gate-only programs every backend must agree on.  They deliberately include
  gates whose operands are *not* in ascending order (an unmerged operation keeps its own operand
  order, and getting that wrong silently permutes qudits in the density-matrix path only),
  parametric two-qubit gates, three-qubit gates, and registers whose qubit indices are not
  ``0..n-1``.
* :data:`COMPARABLE` -- the subset of :data:`PROGRAMS` also cross-checked against pyQuil's NumPy
  reference simulators.  Those take a contiguous register, so a sparse program is relabelled with
  :func:`relabel_contiguous` first; the two-qubit ``SWAP`` and ``I`` cases are left out only
  because the reference simulators do not accept them.
* :data:`DM_ONLY_PROGRAMS` -- programs with mid-circuit ``MEASURE`` / ``RESET``, which only the
  density-matrix backend can run, with their analytic final states in :data:`DM_ONLY_EXPECTED`.

.. note::
    The simulators are **big-endian**: ``qubits[0]`` is the most significant subsystem, so
    ``X 0`` on a two-qubit register gives ``|10>``. The rest of pyQuil is little-endian; use
    :func:`reverse_endianness` before comparing against the reference simulators.
"""

import jax
import numpy as np

from pyquil.gates import CCNOT, CNOT, CPHASE, CSWAP, CZ, FSIM, MEASURE, RESET, RX, RY, RZ, SWAP, XY, H, I, S, T, X, Y, Z
from pyquil.quil import Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import Declare, Gate, Measurement, ResetQubit
from pyquil.simulation._simulator import (
    DensityMatrixSimulator,
    PureStateVectorSimulator,
    TrajectorySimulator,
    _build_trajectory_kernel,
    _KrausStackLayout,
)

PROGRAMS: dict[str, Program] = {}

PROGRAMS["empty"] = Program()

program = Program()
program += X(0)
PROGRAMS["single_x"] = program

program = Program()
program += H(0)
PROGRAMS["hadamard"] = program

program = Program()
program += H(0)
program += CNOT(0, 1)
PROGRAMS["bell"] = program

program = Program()
program += H(0)
program += CNOT(0, 1)
program += CNOT(1, 2)
PROGRAMS["ghz3"] = program

program = Program()
program += H(0)
program += S(0)
program += T(0)
program += H(0)
program += Z(0)
program += Y(0)
PROGRAMS["clifford_chain"] = program

program = Program()
program += RX(0.3, 0)
program += RY(0.7, 0)
program += RZ(1.1, 0)
PROGRAMS["rotations"] = program

program = Program()
program += RX(0.4, 0)
program += RY(0.9, 1)
program += CNOT(0, 1)
program += RZ(1.1, 0)
program += CZ(0, 1)
PROGRAMS["two_qubit_mixed"] = program

# Parametric two-qubit gates.
program = Program()
program += RX(0.4, 0)
program += RY(0.9, 1)
program += XY(0.7, 0, 1)
PROGRAMS["xy"] = program

program = Program()
program += H(0)
program += RX(0.3, 1)
program += FSIM(0.4, 1.3, 0, 1)
program += RZ(0.5, 0)
PROGRAMS["fsim"] = program

program = Program()
program += H(0)
program += H(1)
program += CPHASE(0.9, 1, 0)
PROGRAMS["cphase_reversed"] = program

# Three-qubit gates, with operands both ascending and descending.
program = Program()
program += H(0)
program += H(1)
program += CCNOT(0, 1, 2)
PROGRAMS["ccnot_sorted"] = program

program = Program()
program += X(1)
program += X(2)
program += CCNOT(2, 1, 0)
PROGRAMS["ccnot_reversed"] = program

program = Program()
program += X(1)
program += CNOT(1, 0)
PROGRAMS["cnot_reversed"] = program

program = Program()
program += X(0)
program += X(2)
program += CSWAP(2, 1, 0)
PROGRAMS["cswap_reversed"] = program

program = Program()
program += X(1)
program += SWAP(2, 0)
PROGRAMS["swap_reversed"] = program

program = Program()
program += RX(0.4, 0)
program += RY(0.9, 1)
program += CNOT(0, 1)
program += RZ(1.1, 0)
program += CZ(1, 2)
program += RX(0.2, 2)
program += SWAP(0, 2)
PROGRAMS["deep"] = program

program = Program()
program += X(0)
program += I(1)
PROGRAMS["idle_qubit"] = program

# Registers whose qubit indices are not 0..n-1: the simulator numbers subsystems by sorted
# program qubit, so index 0 is qubit 2 here, not qubit 0.
program = Program()
program += H(2)
program += CNOT(2, 5)
program += RZ(0.3, 5)
PROGRAMS["sparse_register"] = program

program = Program()
program += X(7)
program += RY(0.6, 3)
program += CNOT(7, 3)
PROGRAMS["sparse_reversed"] = program

# The subset of PROGRAMS cross-checked against pyQuil's NumPy reference simulators.
COMPARABLE = [
    "single_x",
    "hadamard",
    "bell",
    "ghz3",
    "clifford_chain",
    "rotations",
    "two_qubit_mixed",
    "xy",
    "fsim",
    "cphase_reversed",
    "ccnot_sorted",
    "ccnot_reversed",
    "cnot_reversed",
    "cswap_reversed",
    "deep",
    "sparse_register",
    "sparse_reversed",
]

# Programs with mid-circuit measurement or reset, and the density matrix they must produce.
DM_ONLY_PROGRAMS: dict[str, Program] = {}
DM_ONLY_EXPECTED: dict[str, np.ndarray] = {}

_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

program = Program()
program += Declare("ro", "BIT", 1)
program += H(0)
program += MEASURE(0, ("ro", 0))
program += H(0)
DM_ONLY_PROGRAMS["measure_then_hadamard"] = program
DM_ONLY_EXPECTED["measure_then_hadamard"] = np.eye(2, dtype=complex) / 2

program = Program()
program += X(0)
program += RESET(0)
program += H(0)
DM_ONLY_PROGRAMS["reset_midcircuit"] = program
DM_ONLY_EXPECTED["reset_midcircuit"] = np.outer(_plus, _plus.conj())

program = Program()
program += Declare("ro", "BIT", 1)
program += H(0)
program += CNOT(0, 1)
program += MEASURE(0, ("ro", 0))
DM_ONLY_PROGRAMS["bell_measure_one"] = program
DM_ONLY_EXPECTED["bell_measure_one"] = np.diag([0.5, 0.0, 0.0, 0.5]).astype(complex)

program = Program()
program += Declare("ro", "BIT", 1)
program += H(0)
program += MEASURE(0, ("ro", 0))
program += RESET(0)
program += X(0)
program += CNOT(0, 1)
DM_ONLY_PROGRAMS["measure_reset_reuse"] = program
DM_ONLY_EXPECTED["measure_reset_reuse"] = np.diag([0.0, 0.0, 0.0, 1.0]).astype(complex)

del program


def relabel_contiguous(program: Program) -> Program:
    """Relabel a program's qubits so that its sorted qubits become ``0..n-1``.

    The simulators number subsystems by sorted program qubit, so the relabelled program's qubit
    ``i`` is exactly the simulator's subsystem ``i``; this is what lets a sparse program be
    compared with a reference simulator that only accepts a contiguous register.
    """
    mapping = {q: i for i, q in enumerate(sorted(program.get_qubit_indices()))}

    def relabel(qubit) -> Qubit:
        return Qubit(mapping[qubit.index if isinstance(qubit, Qubit) else int(qubit)])

    relabelled = Program()
    for inst in program.instructions:
        if isinstance(inst, Gate):
            relabelled += Gate(inst.name, inst.params, [relabel(q) for q in inst.qubits])
        elif isinstance(inst, Measurement):
            relabelled += Measurement(relabel(inst.qubit), inst.classical_reg)
        elif isinstance(inst, ResetQubit):
            relabelled += ResetQubit(relabel(inst.qubit))
        else:
            relabelled += inst
    return relabelled


def simulate_state_vector(program, qubits=None, memory_map=None, **kwargs):
    """Run ``PureStateVectorSimulator`` and return the final ``qx.StateVector``."""
    sim = PureStateVectorSimulator(program, qubits=qubits, **kwargs)
    return sim.compute(sim.linearize(memory_map) if memory_map else None)


def simulate_density_matrix(program, qubits=None, noise_model=None, memory_map=None, **kwargs):
    """Run ``DensityMatrixSimulator`` and return the final ``qx.DensityMatrix``."""
    sim = DensityMatrixSimulator(program, qubits=qubits, noise_model=noise_model, **kwargs)
    return sim.compute(sim.linearize(memory_map) if memory_map else None)


def simulate_trajectories(program, noise_model=None, qubits=None, num_trajectories=1, seed=0, **kwargs):
    """Run ``TrajectorySimulator.compute`` and return ``(state_vector, outcomes)``.

    Every trajectory is kept, so this is for the small ensembles the correctness tests use;
    :meth:`TrajectorySimulator.sample` is the scalable path and discards the states.
    """
    sim = TrajectorySimulator(program, qubits=qubits, noise_model=noise_model, **kwargs)
    return sim.compute(None, jax.random.split(jax.random.key(seed), num_trajectories))


def apply_trajectory_operations(operations, psi, key):
    """Build a one-off trajectory kernel for a bare operation sequence and apply it to *psi*.

    The simulators compile their kernel once at construction from the merge plan; this is for
    tests and benchmarks holding placements with no simulator around them.
    """
    layout = _KrausStackLayout.from_operations(operations, psi.dims)
    return _build_trajectory_kernel(layout)(layout.stack(operations), psi, key)


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
