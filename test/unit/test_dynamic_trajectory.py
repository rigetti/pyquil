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

"""Unit tests for the dynamic-shape trajectory simulator."""

import jax
import jax.numpy as jnp
import numpy as np
import quax as qx

from pyquil.gates import CNOT, MEASURE, RX, H, X
from pyquil.noise._channels import Channel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare
from pyquil.simulation._simulator import (
    DensityMatrixSimulator,
    DynamicTrajectorySimulator,
    PureStateVectorSimulator,
    _dyn_apply,
)

_EMPTY = jnp.array([], dtype=float)


def _overlap(a: qx.StateVector, b: qx.StateVector) -> float:
    return float(jnp.abs(jnp.vdot(a.matrix, b.matrix)) ** 2)


# --------------------------------------------------------------------------
# Deterministic correctness against the exact simulators
# --------------------------------------------------------------------------


def test_matches_pure_state_vector_and_stays_qubit():
    """A noiseless gate-only program reproduces the exact state and never grows."""
    program = Program(X(0), CNOT(0, 1), RX(0.7, 2))
    reference = PureStateVectorSimulator(program).compute(_EMPTY)

    sim = DynamicTrajectorySimulator(program)
    psi, outcomes = sim.compute(_EMPTY, jax.random.key(0))

    assert psi.dims == (2, 2, 2)  # no leakage -> no growth
    assert outcomes.shape == (0,)
    assert _overlap(psi, reference) > 1 - 1e-6


def test_bell_measurement_is_correlated():
    """Bell-state measurements are perfectly correlated and roughly balanced."""
    program = Program(
        Declare("ro", "BIT", 2),
        H(0),
        CNOT(0, 1),
        MEASURE(0, MemoryReference("ro", 0)),
        MEASURE(1, MemoryReference("ro", 1)),
    )

    shots = DynamicTrajectorySimulator(program).sample(_EMPTY, num_trajectories=200, random_seed=1)
    assert shots.shape == (200, 2)
    # the two qubits always agree
    assert np.all(np.asarray(shots)[:, 0] == np.asarray(shots)[:, 1])
    # and both outcomes occur
    assert set(np.asarray(shots)[:, 0].tolist()) == {0, 1}


# --------------------------------------------------------------------------
# Dynamic growth / squeeze of the live state
# --------------------------------------------------------------------------


def test_dyn_apply_grows_then_squeezes():
    """A 1<->2 raising gate grows the state to dim 3; returning empties and squeezes it."""
    psi = qx.StateVector.from_matrix(jnp.array([0.0, 1.0], dtype=complex), (2,))  # |1>
    swap_12 = qx.Unitary.from_matrix(jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex), ((3,), (3,)))

    grown, outcome = _dyn_apply(swap_12, psi, (0,), jax.random.key(0), 1e-9)
    assert outcome is None
    assert grown.dims == (3,)  # state was grown to host |2>
    assert float(jnp.abs(grown.matrix[2]) ** 2) > 0.99
    assert qx.squeeze(grown, 1e-9).dims == (3,)  # population in |2> is retained

    # Apply the raising gate again: population returns to |1>, so squeeze shrinks back.
    returned, _ = _dyn_apply(swap_12, grown, (0,), jax.random.key(1), 1e-9)
    squeezed = qx.squeeze(returned, 1e-9)
    assert squeezed.dims == (2,)
    assert float(jnp.abs(squeezed.matrix[1]) ** 2) > 0.99


# --------------------------------------------------------------------------
# Noise: trajectory average reproduces the density matrix
# --------------------------------------------------------------------------


def test_depolarizing_trajectory_average_matches_density_matrix():
    """Averaging |psi><psi| over trajectories reproduces the depolarized density matrix."""
    gate = X(0)
    noise = NoiseModel.from_channels([Channel.from_depolarizing_constant(gate, depolarizing_constant=0.85)])
    program = Program(gate)

    rho_exact = np.asarray(DensityMatrixSimulator(program, noise_model=noise).compute(_EMPTY).matrix)

    sim = DynamicTrajectorySimulator(program, noise_model=noise)
    key = jax.random.key(0)
    n = 1500
    rho_est = np.zeros((2, 2), dtype=complex)
    for t in range(n):
        psi, _ = sim.compute(_EMPTY, jax.random.fold_in(key, t))
        v = np.asarray(psi.matrix)
        rho_est += np.outer(v, v.conj())
    rho_est /= n

    assert np.allclose(rho_est, rho_exact, atol=0.05)


def test_sample_shape_no_measurements():
    """sample returns an empty per-trajectory outcome row when there are no measurements."""
    shots = DynamicTrajectorySimulator(Program(X(0))).sample(_EMPTY, num_trajectories=5)
    assert shots.shape == (5, 0)
