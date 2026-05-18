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
"""Unified program simulators backed by quax.

Three simulators share a common preprocessing pipeline:

* :class:`PureStateVectorSimulator` — gate-only programs (no noise,
  measurements, or resets).  Jit- and grad-friendly.
* :class:`DensityMatrixSimulator` — any program, optionally with noise.
  Jit- and grad-friendly.
* :class:`TrajectorySimulator` — Monte Carlo trajectory simulation for
  programs with measurements and resets, optionally with noise.

Each simulator is constructed from a :class:`~pyquil.quil.Program` and
exposes ``linearize``, ``resolve``, ``compress``, and ``compute`` methods.
The ``compute`` method is the main entry point and can be passed directly
to ``jax.jit`` or ``jax.grad``.
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import jax
import jax.numpy as jnp
import quax as qx
from jax import Array

from pyquil.api import MemoryMap
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Reset, ResetQubit

from pyquil.noise._noise_model import NoiseModelLike
from pyquil.noise._channels import CycleChannel, get_custom_gates_from_program

from pyquil.transform import expand_defcircuits

from pyquil.simulation._resolver import (
    Linearizer,
    Resolver,
    ResolvedOp,
    TrajectoryOp,
    DensityMatrixOp,
    adapt_for_density_matrix,
    adapt_for_trajectory,
    compressor_from_dag,
    linearizer_from_program,
    dag_from_program,
    resolver_from_program,
)

logger = logging.getLogger(__name__)


def _get_cycle_channel_names(noise_model: NoiseModelLike | None) -> frozenset:
    """Extract DefCircuit names from CycleChannels in the noise model."""
    if noise_model is None:
        return frozenset()
    from pyquil.noise._noise_model import NoiseModel
    if isinstance(noise_model, NoiseModel):
        names = frozenset(
            ch.inst.name for ch in noise_model.channels
            if isinstance(ch, CycleChannel)
        )
        return names
    return frozenset()


# ══════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════


class ProgramSimulator:
    """Base class for program simulators.

    Handles all shared preprocessing: circuit expansion, qubit ordering,
    building the linearizer, resolver, and compressor closures, and
    inferring per-qudit dimensions.

    Subclasses override :meth:`_validate` and :meth:`compute`.

    Instances are immutable after construction.
    """

    __slots__ = ("n_qubits", "qubits", "dims", "_linearize_fn", "_resolve_fn", "_compress_fn")

    def __init__(
        self,
        program: Program,
        qubits: List[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 0,
    ) -> None:
        # Only expand DefCircuits that don't correspond to CycleChannels in the
        # noise model.  CycleChannels are keyed by the cycle Gate instruction,
        # so expanding their DefCircuit would destroy the match.
        cycle_names = _get_cycle_channel_names(noise_model)
        if cycle_names:
            program = expand_defcircuits(program, expand_names_except=cycle_names)
        else:
            program = expand_defcircuits(program)
        self._validate(program)

        if qubits is None:
            qubits = sorted(program.get_qubit_indices())
        self.qubits = qubits
        self.n_qubits = len(qubits)
        qubit_indices = {q: i for i, q in enumerate(qubits)}

        custom_gates = get_custom_gates_from_program(program)

        self._linearize_fn = linearizer_from_program(program)

        dag, node_order = dag_from_program(program, qubit_indices)

        self._resolve_fn = resolver_from_program(
            program, noise_model, qubit_indices, custom_gates or None,
            dag, node_order,
        )

        # Dims are inferred during resolver construction from gate/channel inspection.
        self.dims = self._resolve_fn.dims

        self._compress_fn = compressor_from_dag(dag, node_order, max_subsystem_size, dims=self.dims)

    # -- hook for subclass validation ---------------------

    def _validate(self, program: Program) -> None:
        """Override to reject unsupported instructions."""

    # -- public pipeline methods --------------------------

    def linearize(self, memory_map: MemoryMap) -> Array:
        """Convert a memory map to a flat JAX parameter vector."""
        return self._linearize_fn(memory_map)

    def resolve(self, params: Array) -> List[ResolvedOp]:
        """Resolve parameters into one operator per DAG node."""
        return self._resolve_fn(params)

    def compress(self, resolved: List[ResolvedOp]) -> List[ResolvedOp]:
        """Merge operators via greedy edge contraction."""
        return self._compress_fn(resolved)

    def compute(self, params: Array, **kwargs):
        """Compute the simulation result.  Subclasses must override."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════
# Pure state-vector simulator
# ══════════════════════════════════════════════════════════


class PureStateVectorSimulator(ProgramSimulator):
    """Simulator for gate-only programs (no noise, measurements, or resets).

    All methods are jit- and grad-friendly::

        sim = PureStateVectorSimulator(program)
        params = sim.linearize(memory_map)
        psi = jax.jit(sim.compute)(params)
        U   = jax.jit(sim.unitary)(params)
    """

    __slots__ = ("_psi0",)

    def __init__(
        self,
        program: Program,
        qubits: List[int] | None = None,
        *,
        max_subsystem_size: int = 0,
    ) -> None:
        super().__init__(program, qubits, noise_model=None, max_subsystem_size=max_subsystem_size)
        self._psi0 = qx.zero_state_vector(dims=self.dims)

    def _validate(self, program: Program) -> None:
        for inst in program.instructions:
            if isinstance(inst, Measurement):
                raise ValueError(
                    "PureStateVectorSimulator does not support measurements.  "
                    f"Found: {inst}"
                )
            if isinstance(inst, (Reset, ResetQubit)):
                raise ValueError(
                    "PureStateVectorSimulator does not support resets.  "
                    f"Found: {inst}"
                )

    def compute(self, params: Array) -> qx.StateVector:
        """Compute the final state vector.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: The final state vector.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        psi = self._psi0
        for unitary, subsystem in compressed:
            psi = qx.targeted_apply_unitary(unitary, psi, subsystem)
        return psi

    def __call__(self, params: Array) -> qx.StateVector:
        return self.compute(params)

    def unitary(self, params: Array) -> qx.Unitary:
        """Compute the full program unitary.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: The full unitary matrix.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)

        accumulated: qx.Unitary | None = None
        for op, subsystem in compressed:
            embedded = qx.embed(op, target_dims=self.dims, positions=subsystem)
            if accumulated is None:
                accumulated = embedded
            else:
                accumulated = embedded @ accumulated

        if accumulated is None:
            d = 1
            for dim in self.dims:
                d *= dim
            return qx.Unitary.from_matrix(jnp.eye(d, dtype=complex), self.dims)

        return accumulated


# ══════════════════════════════════════════════════════════
# Density-matrix simulator
# ══════════════════════════════════════════════════════════


class DensityMatrixSimulator(ProgramSimulator):
    """Density-matrix simulator for any program, optionally with noise.

    All methods are jit- and grad-friendly::

        sim = DensityMatrixSimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)
        rho = jax.jit(sim.compute)(params)
    """

    __slots__ = ("_rho0",)

    def __init__(
        self,
        program: Program,
        qubits: List[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 0,
    ) -> None:
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._rho0 = qx.zero_state_matrix(dims=self.dims)

    def compute(self, params: Array) -> qx.DensityMatrix:
        """Compute the final density matrix.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: The final density matrix.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        operations = adapt_for_density_matrix(compressed)
        rho = self._rho0
        for superop, subsystem in operations:
            rho = qx.targeted_apply_superop(superop, rho, subsystem)
        return rho

    def __call__(self, params: Array) -> qx.DensityMatrix:
        return self.compute(params)


# ══════════════════════════════════════════════════════════
# Trajectory simulator
# ══════════════════════════════════════════════════════════


class TrajectorySimulator(ProgramSimulator):
    """Monte Carlo trajectory simulator for programs with measurements and resets.

    The ``compute`` method requires a JAX PRNG key.  The number of
    trajectories is determined by the key shape: a scalar key runs one
    trajectory; a batch of keys ``jax.random.split(key, n)`` runs *n*
    trajectories in parallel::

        sim = TrajectorySimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)

        # Single trajectory
        key = jax.random.key(0)
        psi, outcomes = sim.compute(params, key)

        # Batched trajectories
        keys = jax.random.split(jax.random.key(0), 100)
        psi_batch, outcomes_batch = sim.compute(params, keys)

    The ``sample`` method is a convenience wrapper that runs trajectories
    in batches and discards state vectors, returning only measurement
    outcomes.
    """

    __slots__ = ("_kraus_truncation_threshold",)

    def __init__(
        self,
        program: Program,
        qubits: List[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 0,
        kraus_truncation_threshold: float = 1e-6,
    ) -> None:
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._kraus_truncation_threshold = kraus_truncation_threshold

    def adapt(self, compressed: List[ResolvedOp]) -> List[TrajectoryOp]:
        """Convert compressed ops to trajectory-compatible types."""
        return adapt_for_trajectory(compressed, self._kraus_truncation_threshold)

    def compute(
        self,
        params: Array,
        key: Array,
    ) -> Tuple[qx.StateVector, Array]:
        """Run trajectory simulation.

        :param params: Flat parameter vector from :meth:`linearize`.
        :param key: JAX PRNG key.  Scalar key → single trajectory.
            Batch of keys (from ``jax.random.split``) → batched trajectories.
        :return: Tuple of ``(state_vector, measurement_outcomes)``.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        operations = self.adapt(compressed)

        if key.ndim == 0:
            psi = qx.zero_state_vector(dims=self.dims)
        else:
            n_traj = key.shape[0]
            psi = qx.zero_state_vector(dims=self.dims, ensemble_size=(n_traj,))

        return _apply_trajectory_operations(operations, psi, key)

    def __call__(self, params: Array, key: Array) -> Tuple[qx.StateVector, Array]:
        return self.compute(params, key)

    def sample(
        self,
        params: Array,
        num_trajectories: int = 1000,
        batch_size: int = 250,
        random_seed: int = 0,
    ) -> Array:
        """Run trajectory simulation in batches, returning only measurement outcomes.

        State vectors are discarded after each batch, making this scalable
        to arbitrarily many trajectories.

        :param params: Flat parameter vector from :meth:`linearize`.
        :param num_trajectories: Total number of trajectories to simulate.
        :param batch_size: Maximum number of trajectories per batch.
        :param random_seed: Seed for the JAX PRNG.
        :return: Measurement outcomes with shape ``(num_trajectories, n_measurements)``.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        operations = self.adapt(compressed)

        _, all_outcomes = _run_batched_trajectories(
            operations, self.n_qubits, num_trajectories, batch_size, random_seed,
            keep_states=False, dims=self.dims,
        )

        if len(all_outcomes) == 1:
            return all_outcomes[0]
        return jnp.concatenate(all_outcomes, axis=0)


# ══════════════════════════════════════════════════════════
# Trajectory simulation internals
# ══════════════════════════════════════════════════════════


def _apply_trajectory_operations(
    operations: List[TrajectoryOp],
    psi: qx.StateVector,
    key: Array,
) -> Tuple[qx.StateVector, Array]:
    """Apply trajectory operations to a (batched) state vector.

    Dispatches each operation by type:

    - ``qx.Unitary``: deterministic gate application
    - ``qx.KrausMap``: probabilistic Kraus operator sampling
    - ``qx.QuantumInstrument``: measurement with outcome recording

    :param operations: Ordered list of (operator, subsystem) pairs.
    :param psi: Initial state vector, optionally batched via ensemble dimension.
    :param key: JAX PRNG key (scalar typed key). Will be split internally to
        produce per-trajectory, per-operation sub-keys.
    :return: Tuple of ``(final_state_vector, measurement_outcomes)`` where
        measurement_outcomes has shape ``(*ensemble, n_measurements)`` with
        dtype int32.
    """
    measurement_outcomes: List[Array] = []

    n_stochastic = sum(
        1 for op, _ in operations
        if isinstance(op, (qx.KrausMap, qx.QuantumInstrument))
    )

    ensemble_size = psi.ensemble_size

    if n_stochastic > 0:
        if ensemble_size:
            n_traj = ensemble_size[0]
            all_keys = jax.random.split(key, n_stochastic * n_traj)
            all_keys = all_keys.reshape(n_stochastic, n_traj)
        else:
            all_keys = jax.random.split(key, n_stochastic)

    key_idx = 0

    for op, subsystem in operations:
        match op:
            case qx.Unitary():
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
            case qx.KrausMap():
                op_keys = all_keys[key_idx]
                psi = qx.targeted_apply_kraus_map_trajectory(op, psi, op_keys, subsystem)
                key_idx += 1
            case qx.QuantumInstrument():
                op_keys = all_keys[key_idx]
                psi, outcome = qx.targeted_apply_instrument_to_state_vector(op, psi, op_keys, subsystem)
                measurement_outcomes.append(outcome)
                key_idx += 1
            case _:
                raise TypeError(f"Unsupported operator type: {type(op)}")

    if measurement_outcomes:
        outcomes = jnp.stack(measurement_outcomes, axis=-1)
    else:
        outcomes = jnp.empty((*psi.ensemble_size, 0), dtype=jnp.int32)

    return psi, outcomes


def _run_batched_trajectories(
    operations: List[TrajectoryOp],
    n_qubits: int,
    num_trajectories: int,
    batch_size: int,
    random_seed: int,
    keep_states: bool = True,
    dims: Tuple[int, ...] | None = None,
) -> Tuple[List[qx.StateVector] | None, List[Array]]:
    """Run trajectory simulation in batches."""
    if dims is None:
        dims = (2,) * n_qubits

    key = jax.random.key(random_seed)
    all_psis: List[qx.StateVector] = [] if keep_states else []
    all_outcomes: List[Array] = []

    remaining = num_trajectories
    batch_idx = 0
    t_total = 0.0
    while remaining > 0:
        this_batch = min(remaining, batch_size)
        key, batch_key = jax.random.split(key)

        if this_batch == 1:
            psi = qx.zero_state_vector(dims=dims)
        else:
            psi = qx.zero_state_vector(dims=dims, ensemble_size=(this_batch,))

        t0 = time.perf_counter()
        psi_out, outcomes = _apply_trajectory_operations(operations, psi, batch_key)
        psi_out.matrix.block_until_ready()
        t1 = time.perf_counter()
        t_total += t1 - t0

        if this_batch == 1:
            psi_out = qx.StateVector.from_matrix(
                psi_out.matrix[jnp.newaxis], psi_out.dims,
            )
            outcomes = outcomes[jnp.newaxis]

        logger.debug(
            "Batch %d: %d trajectories, %d qubits, %.3f s",
            batch_idx, this_batch, n_qubits, t1 - t0,
        )

        if keep_states:
            all_psis.append(psi_out)
        all_outcomes.append(outcomes)
        remaining -= this_batch
        batch_idx += 1

    logger.info(
        "Trajectories complete: %d total, %d batches (size=%d), "
        "n_qubits=%d, %.3f s total, %.1f traj/s",
        num_trajectories, batch_idx, batch_size, n_qubits,
        t_total, num_trajectories / t_total if t_total > 0 else float("inf"),
    )

    return (all_psis if keep_states else None), all_outcomes
