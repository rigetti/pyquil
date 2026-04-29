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
simulation.density_matrix module
--------------------------------

Density-matrix simulator backed by quax.

This module provides a density-matrix simulator that uses quax for all quantum
operations. It supports the ``NoiseModel`` from :mod:`pyquil.noise`
in which every channel stores a ``qx.SuperOp`` (or ``qx.QuantumInstrument``) that
*includes* the gate unitary.

Key features:

* No manual einsum string generation — ``qx.targeted_apply_superop`` handles
  subsystem targeting internally.
* Ensemble (batch) simulation via quax broadcasting — pass ``ensemble_size``
  to compute the evolution of many initial states in parallel.
* Superoperator-only code path — no Kraus decomposition needed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp
import quax as qx

from pyquil.api import MemoryMap
from pyquil.quil import Program
from pyquil.quilbase import Gate, Measurement, Reset, ResetQubit

from pyquil.noise import (
    Channel,
    MeasurementChannel,
    NoiseModel,
    ResetChannel,
    get_custom_gates_from_program,
    get_instruction_unitary,
)
from pyquil.transform import expand_defcircuits, unparameterize


# ──────────────────────────────────────────────────────────
# Ideal reset / measurement superoperators (cached singletons)
# ──────────────────────────────────────────────────────────

def _ideal_reset_superop(dim: int = 2) -> qx.SuperOp:
    """Return the superoperator for an ideal reset to |0⟩.

    Kraus operators: K_j = |0⟩⟨j| for j in {0, ..., dim-1}.
    """
    kraus_list = []
    for j in range(dim):
        k = jnp.zeros((dim, dim), dtype=complex)
        k = k.at[0, j].set(1.0)
        kraus_list.append(k)
    choi_matrix = sum(
        (jnp.kron(k, jnp.conj(k)) for k in kraus_list),
        jnp.zeros((dim * dim, dim * dim), dtype=complex),
    )
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]] = ((dim,), (dim,))
    return qx.to_superop(qx.Choi.from_matrix(choi_matrix, dims))


def _ideal_measurement_superop(dim: int = 2) -> qx.SuperOp:
    """Return the superoperator that averages over measurement outcomes (decoherence).

    This is the total channel of an ideal projective measurement: ρ ↦ Σ_j |j⟩⟨j| ρ |j⟩⟨j|,
    which projects the state into the computational basis (removes off-diagonal elements).
    """
    choi_matrix = jnp.zeros((dim * dim, dim * dim), dtype=complex)
    for j in range(dim):
        proj = jnp.zeros((dim, dim), dtype=complex).at[j, j].set(1.0)
        choi_matrix = choi_matrix + jnp.kron(proj, jnp.conj(proj))
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]] = ((dim,), (dim,))
    return qx.to_superop(qx.Choi.from_matrix(choi_matrix, dims))


_RESET_SUPEROP = _ideal_reset_superop(2)
_MEASURE_SUPEROP = _ideal_measurement_superop(2)


# ──────────────────────────────────────────────────────────
# Program → list of (SuperOp, subsystem) operations
# ──────────────────────────────────────────────────────────

def _program_to_operations(
    program: Program,
    noise_model: NoiseModel | None,
    qubit_indices: dict[int, int],
    custom_gates: dict | None = None,
) -> List[Tuple[qx.SuperOp, Tuple[int, ...]]]:
    """Walk program instructions and resolve each to a ``(SuperOp, subsystem)`` pair.

    :param program: Quil program (with DefCircuits already expanded).
    :param noise_model: Optional noise model.
    :param qubit_indices: Mapping from physical qubit id → 0-based position.
    :param custom_gates: Custom gate definitions extracted from the program.
    :return: Ordered list of (superoperator, subsystem_tuple) pairs.
    """
    operations: List[Tuple[qx.SuperOp, Tuple[int, ...]]] = []

    for inst in program.instructions:
        match inst:
            case Gate():
                inst_qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())

                if noise_model is not None:
                    channel = noise_model.get_channel(inst)
                else:
                    channel = None

                if channel is not None and isinstance(channel, Channel):
                    # Channel.process already includes the gate unitary
                    superop = channel.process
                else:
                    # Noiseless: convert ideal unitary to superoperator
                    unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
                    superop = qx.to_superop(unitary)

                operations.append((superop, inst_qubits))

            case Measurement():
                inst_qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())

                if noise_model is not None:
                    meas_channel = noise_model.get_channel(inst)
                else:
                    meas_channel = None

                if meas_channel is not None and isinstance(meas_channel, MeasurementChannel):
                    # Use the total CPTP channel (averaged over outcomes)
                    superop = qx.to_superop(meas_channel.process.total_channel())
                else:
                    superop = _MEASURE_SUPEROP

                operations.append((superop, inst_qubits))

            case ResetQubit():
                inst_qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())  # type: ignore[union-attr]

                if noise_model is not None:
                    reset_channel = noise_model.get_channel(inst)
                else:
                    reset_channel = None

                if reset_channel is not None and isinstance(reset_channel, ResetChannel):
                    superop = reset_channel.process
                else:
                    superop = _RESET_SUPEROP

                operations.append((superop, inst_qubits))

            case Reset():
                # Global reset — apply to every qubit
                for _, idx in sorted(qubit_indices.items()):
                    operations.append((_RESET_SUPEROP, (idx,)))

            case _:
                # Pragmas, declarations, etc. are ignored
                continue

    return operations


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────

def compute_program_density_matrix(
    program: Program,
    noise_model: NoiseModel | None = None,
    qubits: List[int] | None = None,
    memory_map: MemoryMap | None = None,
) -> qx.DensityMatrix:
    """Compute the density matrix resulting from executing a Quil program.

    :param program: The Quil program to simulate.
    :param noise_model: Optional noise model (new-style ``NoiseModel`` from
        :mod:`pyquil.noise`).
    :param qubits: Qubit ordering for the output density matrix. If ``None``,
        qubits are in sorted order.
    :param memory_map: Optional memory map for parameterised programs.
    :return: A ``qx.DensityMatrix`` of the final state.
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
    operations = _program_to_operations(program, noise_model, qubit_indices, custom_gates or None)

    # Initialise state
    rho = qx.zero_state_matrix(n_qubits)

    # Apply operations sequentially
    for superop, subsystem in operations:
        rho = qx.targeted_apply_superop(superop, rho, subsystem)

    return rho
