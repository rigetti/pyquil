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

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from pyquil.api import MemoryMap
from pyquil.noise._noise_model import NoiseModelLike
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Reset, ResetQubit
from pyquil.simulation._resolver import (
    ResolvedOp,
    TrajectoryOp,
    adapt_for_density_matrix,
    adapt_for_trajectory,
    build_dag,
    compressor_from_dag,
    expand_program,
    remap_qubits,
)

# ══════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════


class ProgramSimulator:
    """Base class for program simulators.

    Handles all shared preprocessing: circuit expansion, qubit ordering,
    building the linearizer, resolver, and compressor closures.

    Subclasses override :meth:`_validate` and :meth:`compute`.

    Instances are immutable after construction.
    """

    __slots__ = ("n_qubits", "qubits", "dims", "_linearize_fn", "_resolve_fn", "_compress_fn")

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
    ) -> None:
        self._validate(program)

        if qubits is None:
            qubits = sorted(program.get_qubit_indices())
        self.qubits = qubits
        self.n_qubits = len(qubits)

        # Build resolver from the expanded program.
        expanded_ops, phys_qubits, param_refs = expand_program(program, noise_model)
        qubit_indices = {q: i for i, q in enumerate(qubits)}
        mapped_qubits = remap_qubits(phys_qubits, qubit_indices)
        dag = build_dag(mapped_qubits)

        frozen_ops = list(zip(expanded_ops, mapped_qubits, strict=False))

        def resolve(params: Array) -> list[ResolvedOp]:
            return [
                (cast(Callable[[Array], qx.Unitary], item)(params) if callable(item) else item, subsystem)
                for item, subsystem in frozen_ops
            ]

        # Build linearizer from parameter references discovered during expansion.
        def linearize(memory_map: MemoryMap) -> Array:
            if not param_refs:
                return jnp.array([], dtype=float)
            values = [float(memory_map[name][offset]) for name, offset in param_refs]
            return jnp.array(values, dtype=float)

        self.dims = (2,) * self.n_qubits
        self._linearize_fn = linearize
        self._resolve_fn = resolve

        # Derive barrier nodes: measurements (QuantumInstrument) should not
        # be merged by the compressor.
        barrier_nodes = {i for i, op in enumerate(expanded_ops) if isinstance(op, qx.QuantumInstrument)}

        self._compress_fn = compressor_from_dag(
            dag,
            max_subsystem_size,
            dims=self.dims,
            barrier_nodes=barrier_nodes,
        )

    # -- hook for subclass validation ---------------------

    def _validate(self, program: Program) -> None:
        """Override to reject unsupported instructions."""

    # -- public pipeline methods --------------------------

    def linearize(self, memory_map: MemoryMap) -> Array:
        """Convert a memory map to a flat JAX parameter vector."""
        return self._linearize_fn(memory_map)

    def resolve(self, params: Array) -> list[ResolvedOp]:
        """Resolve parameters into one operator per DAG node."""
        return self._resolve_fn(params)

    def compress(self, resolved: list[ResolvedOp]) -> list[ResolvedOp]:
        """Merge operators via greedy edge contraction."""
        return self._compress_fn(resolved)

    def compute(self, params: Array, **kwargs: Any) -> Any:
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
        U = jax.jit(sim.unitary)(params)
    """

    __slots__ = ("_psi0",)

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        max_subsystem_size: int = 2,
    ) -> None:
        super().__init__(program, qubits, noise_model=None, max_subsystem_size=max_subsystem_size)
        self._psi0 = qx.zero_state_vector(dims=self.dims)

    def _validate(self, program: Program) -> None:
        for inst in program.instructions:
            if isinstance(inst, Measurement):
                raise ValueError(f"PureStateVectorSimulator does not support measurements.  Found: {inst}")
            if isinstance(inst, (Reset, ResetQubit)):
                raise ValueError(f"PureStateVectorSimulator does not support resets.  Found: {inst}")

    def compute(self, params: Array) -> qx.StateVector:  # type: ignore[override]
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
            return qx.Unitary.from_matrix(jnp.eye(d, dtype=complex), (self.dims, self.dims))

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
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
    ) -> None:
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._rho0 = qx.zero_state_matrix(dims=self.dims)

    def compute(self, params: Array) -> qx.DensityMatrix:  # type: ignore[override]
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

    __slots__ = ("_kraus_truncation_threshold", "_devices")

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        kraus_truncation_threshold: float = 1e-6,
        devices: list[jax.Device] | None = None,
    ) -> None:
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._kraus_truncation_threshold = kraus_truncation_threshold
        self._devices = devices if devices is not None else jax.devices()

    def adapt(self, compressed: list[ResolvedOp]) -> list[TrajectoryOp]:
        """Convert compressed ops to trajectory-compatible types."""
        return adapt_for_trajectory(compressed, self._kraus_truncation_threshold)

    def compute(  # type: ignore[override]
        self,
        params: Array,
        key: Array,
    ) -> tuple[qx.StateVector, Array]:
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

    def __call__(self, params: Array, key: Array) -> tuple[qx.StateVector, Array]:
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
        to arbitrarily many trajectories.  When multiple devices are
        available, each batch is sharded across them so that every device
        processes ``batch_size // n_devices`` trajectories concurrently.

        :param params: Flat parameter vector from :meth:`linearize`.
        :param num_trajectories: Total number of trajectories to simulate.
        :param batch_size: Maximum number of trajectories per batch
            (total across all devices).
        :param random_seed: Seed for the JAX PRNG.
        :return: Measurement outcomes with shape ``(num_trajectories, n_measurements)``.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        operations = self.adapt(compressed)

        _, all_outcomes = _run_batched_trajectories(
            operations,
            self.n_qubits,
            num_trajectories,
            batch_size,
            random_seed,
            keep_states=False,
            dims=self.dims,
            devices=self._devices,
        )

        if len(all_outcomes) == 1:
            return all_outcomes[0]
        return jnp.concatenate(all_outcomes, axis=0)


# ══════════════════════════════════════════════════════════
# Trajectory simulation internals
# ══════════════════════════════════════════════════════════


def _apply_trajectory_operations(
    operations: list[TrajectoryOp],
    psi: qx.StateVector,
    key: Array,
) -> tuple[qx.StateVector, Array]:
    """Apply trajectory operations to a (batched) state vector.

    Dispatches each operation by type:

    - ``qx.Unitary``: deterministic gate application
    - ``qx.KrausMap``: probabilistic Kraus operator sampling
    - ``qx.QuantumInstrument``: measurement with outcome recording

    Key generation is sharding-friendly: per-operation keys are derived
    lazily via ``jax.random.fold_in`` so that the key array is never
    materialised in full on a single device.

    :param operations: Ordered list of (operator, subsystem) pairs.
    :param psi: Initial state vector, optionally batched via ensemble dimension.
    :param key: JAX PRNG key (scalar typed key). Will be split internally to
        produce per-trajectory, per-operation sub-keys.
    :return: Tuple of ``(final_state_vector, measurement_outcomes)`` where
        measurement_outcomes has shape ``(*ensemble, n_measurements)`` with
        dtype int32.
    """
    measurement_outcomes: list[Array] = []

    ensemble_size = psi.ensemble_size

    # Derive per-trajectory base keys once.  When the state is sharded
    # across devices the resulting key array inherits the same sharding,
    # so each device only materialises its own slice.
    if ensemble_size:
        if key.ndim > 0:
            # Already per-trajectory keys (e.g. from multi-device sharding
            # or batched ``compute()``).
            per_traj_keys = key
        else:
            per_traj_keys = jax.random.split(key, ensemble_size[0])
    else:
        per_traj_keys = None

    stochastic_idx = 0

    for op, subsystem in operations:
        match op:
            case qx.Unitary():
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
            case qx.KrausMap():
                if per_traj_keys is not None:
                    op_keys = jax.vmap(lambda k, s=stochastic_idx: jax.random.fold_in(k, s))(per_traj_keys)
                else:
                    op_keys = jax.random.fold_in(key, stochastic_idx)
                psi = qx.targeted_apply_kraus_map_trajectory(op, psi, op_keys, subsystem)
                stochastic_idx += 1
            case qx.QuantumInstrument():
                if per_traj_keys is not None:
                    op_keys = jax.vmap(lambda k, s=stochastic_idx: jax.random.fold_in(k, s))(per_traj_keys)
                else:
                    op_keys = jax.random.fold_in(key, stochastic_idx)
                psi, outcome = qx.targeted_apply_instrument_to_state_vector(op, psi, op_keys, subsystem)
                measurement_outcomes.append(outcome)
                stochastic_idx += 1
            case _:
                raise TypeError(f"Unsupported operator type: {type(op)}")

    if measurement_outcomes:
        outcomes = jnp.stack(measurement_outcomes, axis=-1)
    else:
        outcomes = jnp.empty((*psi.ensemble_size, 0), dtype=jnp.int32)

    return psi, outcomes


def _make_mesh(devices: list[jax.Device] | None) -> Mesh | None:
    """Build a 1-D ``Mesh`` over *devices*, or ``None`` for single-device."""
    if devices is None:
        devices = jax.devices()
    if len(devices) <= 1:
        return None
    return Mesh(np.array(devices), axis_names=("traj",))


def _round_up_to(n: int, divisor: int) -> int:
    """Round *n* up to the nearest multiple of *divisor*."""
    return ((n + divisor - 1) // divisor) * divisor


def _run_batched_trajectories(
    operations: list[TrajectoryOp],
    n_qubits: int,
    num_trajectories: int,
    batch_size: int,
    random_seed: int,
    keep_states: bool = True,
    dims: tuple[int, ...] | None = None,
    devices: list[jax.Device] | None = None,
) -> tuple[list[qx.StateVector] | None, list[Array]]:
    """Run trajectory simulation in batches, optionally sharded across devices.

    When *devices* contains more than one device a :class:`jax.sharding.Mesh`
    is constructed and both the initial state vector and PRNG keys are sharded
    along the trajectory (ensemble) axis.  XLA's SPMD partitioner then
    distributes the work so that each device processes its own slice.
    """
    if dims is None:
        dims = (2,) * n_qubits

    mesh = _make_mesh(devices)
    n_devices = len(mesh.devices.flat) if mesh is not None else 1

    key = jax.random.key(random_seed)
    all_psis: list[qx.StateVector] = [] if keep_states else []
    all_outcomes: list[Array] = []

    remaining = num_trajectories
    while remaining > 0:
        this_batch = min(remaining, batch_size)

        # Pad to a multiple of n_devices so the shard split is even.
        padded_batch = _round_up_to(this_batch, n_devices) if n_devices > 1 else this_batch
        n_pad = padded_batch - this_batch

        key, batch_key = jax.random.split(key)

        if padded_batch == 1:
            psi = qx.zero_state_vector(dims=dims)
        else:
            psi = qx.zero_state_vector(dims=dims, ensemble_size=(padded_batch,))

        # Shard state and key across devices when a mesh is available.
        if mesh is not None:
            sharding = NamedSharding(mesh, PartitionSpec("traj"))  # type: ignore[no-untyped-call]
            psi = qx.StateVector.from_matrix(
                jax.device_put(psi.matrix, sharding),
                psi.dims,
            )
            # Split a per-trajectory key vector and shard it.
            batch_keys = jax.random.split(batch_key, padded_batch)
            batch_keys = jax.device_put(batch_keys, sharding)
        else:
            batch_keys = batch_key

        psi_out, outcomes = _apply_trajectory_operations(operations, psi, batch_keys)
        psi_out.matrix.block_until_ready()

        # Strip padding rows.
        if n_pad > 0:
            psi_out = qx.StateVector.from_matrix(
                psi_out.matrix[:this_batch],
                psi_out.dims,
            )
            outcomes = outcomes[:this_batch]

        if this_batch == 1 and padded_batch == 1:
            psi_out = qx.StateVector.from_matrix(
                psi_out.matrix[jnp.newaxis],
                psi_out.dims,
            )
            outcomes = outcomes[jnp.newaxis]

        if keep_states:
            all_psis.append(psi_out)
        all_outcomes.append(outcomes)
        remaining -= this_batch

    return (all_psis if keep_states else None), all_outcomes
