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

* Quax offers built-in methods for applying superoperators which handle
  automatic promotion and conversion where necessary. We take advantage
  of those.
* Ensemble (batch) simulation via broadcasting. It's possible to run the
  simulation on an ensemble of initial states.
* Exclusively for simple programs without classical control flow. No support for
  conditionals, loops, or dynamic circuits. Midcircuit measurements are permitted,
  but only for effect - the outcomes are not recorded.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

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
# Program → list of (SuperOp, subsystem) operations
# ──────────────────────────────────────────────────────────


def _program_to_operations(
    program: Program,
    noise_model: NoiseModel | None,
    qubit_indices: Dict[int, int],
) -> List[Tuple[qx.SuperOp, Tuple[int, ...]]]:
    """Walk program instructions and resolve each to a ``(SuperOp, subsystem)`` pair.

    :param program: Quil program (with DefCircuits already expanded).
    :param noise_model: Optional noise model.
    :param qubit_indices: Mapping from physical qubit id → 0-based position.
    :param custom_gates: Custom gate definitions extracted from the program.
    :return: Ordered list of (superoperator, subsystem_tuple) pairs.
    """
    operations: List[Tuple[qx.SuperOp, Tuple[int, ...]]] = []

    # Extract custom gate definitions (DefGate)
    custom_gates = get_custom_gates_from_program(program)

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
                    superop = qx.gates.MEASURE().total_channel()

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
                    superop = qx.gates.RESET()

                operations.append((superop, inst_qubits))

            case Reset():
                # Global reset — apply to every qubit
                for _, idx in sorted(qubit_indices.items()):
                    operations.append((qx.gates.RESET(), (idx,)))

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
    :param noise_model: Optional noise model.
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

    # Build operation list
    operations = _program_to_operations(program, noise_model, qubit_indices)

    # Determine per-qudit dimensions: max over all ops applied to each slot.
    # A qutrit superop has dims[0] = (3,); this auto-promotes qubit slots to qutrit
    # when a larger-dimensional gate is encountered.
    qudit_dims: List[int] = [2] * n_qubits
    for superop, subsystem in operations:
        for slot, dim in zip(subsystem, superop.dims[0]):
            if dim > qudit_dims[slot]:
                qudit_dims[slot] = dim

    # Initialise state with correct per-qudit dimensions
    rho = qx.zero_state_matrix(dims=tuple(qudit_dims))

    # Apply operations sequentially
    for superop, subsystem in operations:
        rho = qx.targeted_apply_superop(superop, rho, subsystem)

    return rho
