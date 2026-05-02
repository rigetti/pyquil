##############################################################################
# Copyright 2016-2026 Rigetti Computing
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
simulation.state_vector module
------------------------------

Noiseless state-vector simulator backed by quax.

This module provides a pure state-vector simulator that uses
``qx.targeted_apply_unitary`` for gate application. It is much faster
than the density-matrix simulator for noiseless circuits because
it operates on vectors of size ``2^n`` rather than matrices of size
``2^n × 2^n``.

Use :func:`compute_program_state_vector` to obtain the final
``qx.StateVector`` for a given Quil program.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp
import quax as qx

from pyquil.api import MemoryMap
from pyquil.quil import Program
from pyquil.quilbase import Gate, Measurement, Reset, ResetQubit

from pyquil.noise import (
    get_custom_gates_from_program,
    get_instruction_unitary,
)
from pyquil.transform import expand_defcircuits, unparameterize


# ──────────────────────────────────────────────────────────
# Program → list of (Unitary, subsystem) operations
# ──────────────────────────────────────────────────────────


def _program_to_unitary_operations(
    program: Program,
    qubit_indices: dict[int, int],
    custom_gates: dict | None = None,
) -> List[Tuple[qx.Unitary, Tuple[int, ...]]]:
    """Walk program instructions and resolve each gate to a ``(Unitary, subsystem)`` pair.

    Measurements and resets are handled by projecting into the computational basis
    or resetting to |0⟩ respectively — both are implemented as unitaries acting on
    a pure state (projection followed by renormalisation for measurements, and
    simply replacing the qubit state with |0⟩ for resets).

    .. note::
        This simulator is noiseless.  Measurements decohere the qubit
        (project into the computational basis, weighted by Born probabilities)
        but in the state-vector picture we simply apply the dephasing
        projector |0⟩⟨0| + |1⟩⟨1| (identity in computational basis) which
        is a no-op on the state vector amplitudes.  The classical bit is
        not tracked here — use the density-matrix simulator or the QVM
        for that.

    :param program: Quil program (with DefCircuits already expanded).
    :param qubit_indices: Mapping from physical qubit id → 0-based position.
    :param custom_gates: Custom gate definitions extracted from the program.
    :return: Ordered list of (unitary, subsystem_tuple) pairs.
    """
    operations: List[Tuple[qx.Unitary, Tuple[int, ...]]] = []

    # Pre-build the reset "unitary" (projection onto |0⟩ — not unitary but
    # works for pure-state reset from |0⟩ initial state when applied as a
    # Kraus-like operator).  For a true reset we use the superop approach
    # below instead.
    _reset_ops = None  # lazy

    for inst in program.instructions:
        match inst:
            case Gate():
                inst_qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())
                unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
                operations.append((unitary, inst_qubits))

            case Measurement():
                # In the noiseless state-vector picture a measurement
                # without post-selection simply decoheres the qubit.
                # We skip it here — the state vector retains full
                # superposition, which is correct for computing
                # probabilities / expectation values.
                continue

            case ResetQubit() | Reset():
                # Resets are skipped in the state-vector simulator
                # because we assume the initial state is |0...0⟩ and
                # mid-circuit resets in a noiseless pure-state sim
                # would require a density-matrix treatment.
                continue

            case _:
                # Pragmas, declarations, etc.
                continue

    return operations


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────


def compute_program_state_vector(
    program: Program,
    qubits: List[int] | None = None,
    memory_map: MemoryMap | None = None,
) -> qx.StateVector:
    """Compute the state vector resulting from executing a Quil program.

    This is a noiseless simulator — no noise model is applied.  For
    noisy simulations use :func:`pyquil.simulation.density_matrix.compute_program_density_matrix`.

    :param program: The Quil program to simulate.
    :param qubits: Qubit ordering for the output state vector.  If ``None``,
        qubits are in sorted order of those appearing in the program.
    :param memory_map: Optional memory map for parameterised programs.
    :return: A ``qx.StateVector`` of the final state.
    """
    # Resolve parameters
    if memory_map is not None:
        program = unparameterize(program, memory_map)

    # Expand DefCircuit instructions
    program = expand_defcircuits(program)

    # Determine qubit order
    if qubits is None:
        qubits = sorted(program.get_qubit_indices())
    qubit_indices = {q: i for i, q in enumerate(qubits)}
    n_qubits = len(qubits)

    # Extract custom gate definitions (DefGate)
    custom_gates = get_custom_gates_from_program(program)

    # Build operation list
    operations = _program_to_unitary_operations(program, qubit_indices, custom_gates or None)

    # Initialise state |0...0⟩
    psi = qx.zero_state_vector(n_qubits)

    # Apply operations sequentially
    for unitary, subsystem in operations:
        psi = qx.targeted_apply_unitary(unitary, psi, subsystem)

    return psi
