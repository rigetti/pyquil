"""Correctness tests for the compressor used by ``TrajectorySimulator``.

The simulators in :mod:`pyquil.simulation._simulator` share a *compressor*
(see :func:`pyquil.simulation._resolver.compressor_from_dag` and
``_merge_ops``) which merges adjacent operations into multi-qudit groups when
``max_subsystem_size >= 2``.  Compression must be a no-op on the *physics*: the
state produced by a compressed simulation has to match the state produced by an
uncompressed one (and an independent oracle).

These tests pin that invariant down for the :class:`TrajectorySimulator` across
the full matrix of cases requested:

* qubit (d=2) and qutrit (d=3) registers,
* gate-only and gate+measurement circuits,
* noiseless and noisy circuits,
* small (2-register) and larger (5-register) circuits.

All tests run at ``max_subsystem_size=2`` and use **non-sequential** qubit
indices so that the physical-to-logical remapping is exercised.  Several
circuits deliberately use multi-qudit gates whose qubit arguments are *not*
sorted (e.g. ``CNOT(2, 5)``, ``TSWAP(2, 5)``) to exercise the embedding of an
operator at non-trivial positions inside a merged subsystem — the path most
likely to be dimension-sensitive.

Verification strategy
---------------------
The ground-truth ("oracle") is always produced **without** compression
(``max_subsystem_size=0``), which the user has confirmed to be correct:

* gate-only oracles use :class:`PureStateVectorSimulator` (pure state) and
  :class:`DensityMatrixSimulator` (density matrix);
* noisy / measurement oracles use :class:`DensityMatrixSimulator`.

For the compressed :class:`TrajectorySimulator` we reconstruct the simulated
density matrix as the Monte-Carlo average of the pure trajectory projectors,

    rho_est = (1 / N) * sum_i |psi_i><psi_i|,

and compare it against the oracle density matrix via fidelity.  Noiseless
circuits need only a handful of (identical) trajectories; noisy circuits use
many trajectories and keep the noise light so the resulting state stays close
to low-rank and is therefore reconstructable with a tractable sample count.
Measurement circuits additionally compare empirical outcome distributions to
the oracle populations.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, CZ, H, RX, RZ, X
from pyquil.noise._channels import Channel, MeasurementChannel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import Gate, Measurement
from pyquil.simulation._simulator import (
    DensityMatrixSimulator,
    TrajectorySimulator,
    _op_to_kraus_matrix,
)

# Every test runs the compressed simulator at this subsystem size.
MAX_SUBSYSTEM_SIZE = 2

# Non-sequential register layouts shared across the 2- and 5-register cases.
QUBITS_2 = [5, 2]
QUBITS_5 = [7, 2, 9, 4, 0]


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════


def _density_from_states(matrix, dims):
    """Reconstruct a density matrix from an ensemble of pure state vectors.

    :param matrix: Array of shape ``(N, d)`` whose rows are trajectory kets.
    :param dims: Per-qudit dimensions of the system.
    :return: ``qx.DensityMatrix`` equal to ``mean_i |psi_i><psi_i|``.
    """
    m = jnp.asarray(matrix)
    n = m.shape[0]
    rho = jnp.einsum("ni,nj->ij", m, jnp.conj(m)) / n
    return qx.DensityMatrix.from_matrix(rho, dims)


def _oracle_density(program, qubits, noise_model=None):
    """Ground-truth density matrix via the uncompressed density-matrix simulator."""
    sim = DensityMatrixSimulator(
        program, qubits=qubits, noise_model=noise_model, max_subsystem_size=0
    )
    return sim.compute(sim.linearize({}))


def _assert_real_compression(program, qubits, noise_model=None):
    """Assert the compressor actually merges ops at ``MAX_SUBSYSTEM_SIZE``.

    A test of compression correctness is meaningless if no merging happens, so
    every test first confirms the compressed op count is strictly smaller than
    the resolved op count.
    """
    sim = TrajectorySimulator(
        program, qubits=qubits, noise_model=noise_model,
        max_subsystem_size=MAX_SUBSYSTEM_SIZE,
    )
    params = sim.linearize({})
    resolved = sim.resolve(params)
    compressed = sim.compress(resolved)
    assert len(compressed) < len(resolved), (
        f"compression did not merge any ops "
        f"({len(resolved)} resolved -> {len(compressed)} compressed)"
    )


def _trajectory_density(program, qubits, dims, n_traj, *, noise_model=None,
                        seed=0, max_subsystem_size=MAX_SUBSYSTEM_SIZE):
    """Run ``n_traj`` trajectories and reconstruct the simulated density matrix.

    The compressor preserves program order for measurement nodes, so the columns
    of ``outcomes`` correspond to the ``MEASURE`` instructions in program order
    regardless of ``max_subsystem_size``.  No re-ordering is applied here — the
    raw column order is part of what these tests verify (see
    :func:`test_measurement_outcome_column_order_under_compression`).

    :return: ``(rho_est, outcomes)`` where ``rho_est`` is the Monte-Carlo
        density matrix and ``outcomes`` has shape ``(n_traj, n_measurements)``.
    """
    sim = TrajectorySimulator(
        program, qubits=qubits, noise_model=noise_model,
        max_subsystem_size=max_subsystem_size,
    )
    params = sim.linearize({})
    keys = jax.random.split(jax.random.key(seed), n_traj)
    psi, outcomes = sim.compute(params, keys)
    return _density_from_states(psi.matrix, dims), np.asarray(outcomes)


def _outcome_distribution(outcomes, dims):
    """Empirical joint distribution of measurement outcomes as a length-``d`` vector.

    Each trajectory's per-measurement outcomes (in program / qubit-slot order)
    are interpreted as digits in a row-major mixed-radix number with the given
    ``dims`` and histogrammed.  The resulting index ordering matches the
    canonical basis ordering of the density matrix, so the distribution can be
    compared directly against the oracle diagonal.
    """
    out = np.asarray(outcomes)
    n_traj, n_meas = out.shape
    radices = np.asarray(dims[:n_meas])
    # Mixed-radix encoding: index = sum_k outcome_k * prod(radices[k+1:]).
    weights = np.ones(n_meas, dtype=np.int64)
    for k in range(n_meas - 2, -1, -1):
        weights[k] = weights[k + 1] * radices[k + 1]
    indices = (out * weights).sum(axis=1)
    counts = np.bincount(indices, minlength=int(np.prod(radices)))
    return counts / n_traj


def _total_variation(p, q):
    """Total-variation distance between two probability vectors."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def _sampler_total_channel(program, qubits, *, noise_model=None, max_subsystem_size):
    """Full superoperator the trajectory *sampler* implements, as a dense matrix.

    This composes the per-operation channels exactly as the trajectory sampler
    sees them: each operation is converted to its padded Kraus matrices via
    :func:`pyquil.simulation._simulator._op_to_kraus_matrix` (the very matrices
    fed into the Monte-Carlo sampling loop), promoted to a superoperator, and
    embedded into the full register before being composed.

    Building the channel from ``pyquil.simulation._simulator._op_to_kraus_matrix`` (rather than from the
    high-level operator objects) means this exercises the padding, outcome/Kraus
    axis flattening, and dimension handling that only the trajectory path uses.
    The result is deterministic and independent of any sampling, so it is a
    far more sensitive probe of compression correctness than a Monte-Carlo
    fidelity: a *subtly* wrong error rate at ``max_subsystem_size=2`` shows up
    here as a non-zero channel difference even though it would hide under the
    statistical noise of a fidelity estimate.

    :return: A dense ``(D**2, D**2)`` superoperator matrix.
    """
    sim = TrajectorySimulator(
        program, qubits=qubits, noise_model=noise_model,
        max_subsystem_size=max_subsystem_size,
    )
    operations = sim.adapt(sim.compress(sim.resolve(sim.linearize({}))))
    dims = tuple(sim.dims)
    dimension = int(np.prod(dims))

    channel = np.eye(dimension * dimension, dtype=complex)
    for op, subsystem in operations:
        matrix, _divisor, _is_measure = _op_to_kraus_matrix(op)
        kraus = qx.KrausMap.from_matrix(
            np.asarray(matrix), (tuple(dims[i] for i in subsystem),) * 2
        )
        embedded = qx.embed(
            qx.to_superop(kraus), target_dims=dims, positions=tuple(subsystem)
        )
        channel = np.asarray(embedded.matrix) @ channel
    return channel



# ══════════════════════════════════════════════════════════
# Circuit builders
#
# Each builder returns ``(program, qubits, dims)`` and (optionally) a matching
# noise model.  Single-qudit gates flanking the entanglers are absorbable by
# the compressor, which is what forces real 2-qudit merges to occur.
# ══════════════════════════════════════════════════════════


def _qubit_circuit_2():
    """2-qubit gate-only circuit on non-sequential qubits [5, 2].

    ``CNOT(2, 5)`` is intentionally control>target so the two-qubit gate is
    embedded at a non-sorted position inside the merged subsystem.
    """
    p = Program()
    p += H(5)
    p += CNOT(2, 5)        # control on the high index -> non-sorted subsystem
    p += X(5)
    p += H(2)
    return p, QUBITS_2, (2, 2)


def _qubit_circuit_5():
    """5-qubit gate-only circuit on non-sequential qubits [7, 2, 9, 4, 0]."""
    p = Program()
    for q in QUBITS_5:
        p += H(q)
    p += CZ(7, 2)
    p += CZ(9, 4)
    for q, a in zip(QUBITS_5, (0.3, 0.7, 1.1, 0.5, 0.9)):
        p += RX(a, q)
    p += CNOT(2, 9)        # non-sorted relative to qubit ordering
    p += CNOT(4, 0)
    for q, a in zip(QUBITS_5, (0.2, 0.6, 0.4, 0.8, 1.0)):
        p += RZ(a, q)
    p += CNOT(0, 7)        # spans the two ends -> forces another merge
    return p, QUBITS_5, (2, 2, 2, 2, 2)


def _qutrit_circuit_2():
    """2-qutrit gate-only circuit on non-sequential qubits [5, 2].

    ``TSWAP(2, 5)`` is non-sorted, which is the case the user reports as broken
    under compression.
    """
    p = Program()
    p += Gate("TH", [], [5])
    p += Gate("TSWAP", [], [2, 5])     # non-sorted two-qutrit gate
    p += Gate("TRX01", [0.9], [2])
    p += Gate("TX", [], [5])
    return p, QUBITS_2, (3, 3)


def _qutrit_circuit_5():
    """5-qutrit gate-only circuit on non-sequential qubits [7, 2, 9, 4, 0]."""
    p = Program()
    for q in QUBITS_5:
        p += Gate("TH", [], [q])
    p += Gate("TSWAP", [], [2, 7])      # non-sorted
    p += Gate("TSWAP", [], [9, 4])
    for q, a in zip(QUBITS_5, (0.3, 0.7, 1.1, 0.5, 0.9)):
        p += Gate("TRX01", [a], [q])
    p += Gate("TSWAP", [], [4, 0])
    p += Gate("TX", [], [7])
    p += Gate("TX", [], [9])
    return p, QUBITS_5, (3, 3, 3, 3, 3)


def _depolarizing_model(insts, fidelity):
    """Build a depolarizing noise model for the given instructions."""
    return NoiseModel.from_channels(
        [Channel.from_gate_fidelity(inst=inst, fidelity=fidelity) for inst in insts]
    )


# ══════════════════════════════════════════════════════════
# 1-2.  Gate-only, no noise
# ══════════════════════════════════════════════════════════


def test_compression_two_qubit_gates_no_noise():
    """2-qubit gate-only circuit: compressed trajectory == uncompressed oracle."""
    program, qubits, dims = _qubit_circuit_2()
    _assert_real_compression(program, qubits)

    oracle = _oracle_density(program, qubits)
    rho_est, _ = _trajectory_density(program, qubits, dims, n_traj=8)
    assert float(qx.fidelity(rho_est, oracle)) > 0.9999

    # Compressed and uncompressed trajectories must agree exactly (noiseless).
    rho_uncompressed, _ = _trajectory_density(
        program, qubits, dims, n_traj=8, max_subsystem_size=0
    )
    assert float(qx.fidelity(rho_est, rho_uncompressed)) > 0.9999


def test_compression_five_qubit_gates_no_noise():
    """5-qubit gate-only circuit: compressed trajectory == uncompressed oracle."""
    program, qubits, dims = _qubit_circuit_5()
    _assert_real_compression(program, qubits)

    oracle = _oracle_density(program, qubits)
    rho_est, _ = _trajectory_density(program, qubits, dims, n_traj=8)
    assert float(qx.fidelity(rho_est, oracle)) > 0.9999

    rho_uncompressed, _ = _trajectory_density(
        program, qubits, dims, n_traj=8, max_subsystem_size=0
    )
    assert float(qx.fidelity(rho_est, rho_uncompressed)) > 0.9999


# ══════════════════════════════════════════════════════════
# 3-4.  Gate-only, with noise
# ══════════════════════════════════════════════════════════


def test_compression_two_qubit_gates_with_noise():
    """2-qubit noisy circuit: reconstructed density matrix matches oracle."""
    program, qubits, dims = _qubit_circuit_2()
    noise_model = _depolarizing_model(
        [CNOT(2, 5), X(5), H(2), H(5)], fidelity=0.97
    )
    _assert_real_compression(program, qubits, noise_model)

    oracle = _oracle_density(program, qubits, noise_model)
    rho_est, _ = _trajectory_density(
        program, qubits, dims, n_traj=8000, noise_model=noise_model, seed=1
    )
    assert float(qx.fidelity(rho_est, oracle)) > 0.99


def test_compression_five_qubit_gates_with_noise():
    """5-qubit noisy circuit: reconstructed density matrix matches oracle."""
    program, qubits, dims = _qubit_circuit_5()
    noise_model = _depolarizing_model(
        [CZ(7, 2), CZ(9, 4), CNOT(2, 9), CNOT(4, 0), CNOT(0, 7)], fidelity=0.99
    )
    _assert_real_compression(program, qubits, noise_model)

    oracle = _oracle_density(program, qubits, noise_model)
    rho_est, _ = _trajectory_density(
        program, qubits, dims, n_traj=12000, noise_model=noise_model, seed=2
    )
    assert float(qx.fidelity(rho_est, oracle)) > 0.98


# ══════════════════════════════════════════════════════════
# 5-6.  Qutrit gate-only, no noise
# ══════════════════════════════════════════════════════════


def test_compression_two_qutrit_gates_no_noise():
    """2-qutrit gate-only circuit: the qutrit compression case under test."""
    program, qubits, dims = _qutrit_circuit_2()
    _assert_real_compression(program, qubits)

    oracle = _oracle_density(program, qubits)
    rho_est, _ = _trajectory_density(program, qubits, dims, n_traj=8)
    assert float(qx.fidelity(rho_est, oracle)) > 0.9999

    rho_uncompressed, _ = _trajectory_density(
        program, qubits, dims, n_traj=8, max_subsystem_size=0
    )
    assert float(qx.fidelity(rho_est, rho_uncompressed)) > 0.9999


def test_compression_five_qutrit_gates_no_noise():
    """5-qutrit gate-only circuit: compressed trajectory == uncompressed oracle."""
    program, qubits, dims = _qutrit_circuit_5()
    _assert_real_compression(program, qubits)

    oracle = _oracle_density(program, qubits)
    rho_est, _ = _trajectory_density(program, qubits, dims, n_traj=8)
    assert float(qx.fidelity(rho_est, oracle)) > 0.9999

    rho_uncompressed, _ = _trajectory_density(
        program, qubits, dims, n_traj=8, max_subsystem_size=0
    )
    assert float(qx.fidelity(rho_est, rho_uncompressed)) > 0.9999


# ══════════════════════════════════════════════════════════
# 7-8.  Qutrit gate-only, with noise
# ══════════════════════════════════════════════════════════


def test_compression_two_qutrit_gates_with_noise():
    """2-qutrit noisy circuit: reconstructed density matrix matches oracle."""
    program, qubits, dims = _qutrit_circuit_2()
    noise_model = _depolarizing_model(
        [Gate("TSWAP", [], [2, 5]), Gate("TX", [], [5]), Gate("TH", [], [5])],
        fidelity=0.98,
    )
    _assert_real_compression(program, qubits, noise_model)

    oracle = _oracle_density(program, qubits, noise_model)
    rho_est, _ = _trajectory_density(
        program, qubits, dims, n_traj=8000, noise_model=noise_model, seed=3
    )
    assert float(qx.fidelity(rho_est, oracle)) > 0.99


def test_compression_five_qutrit_gates_with_noise():
    """5-qutrit noisy circuit: reconstructed density matrix matches oracle.

    Noise is kept light (per-gate fidelity 0.99 on two gates only) so the
    output state stays close to pure and is reconstructable from a tractable
    number of trajectories despite the d=243 Hilbert space.
    """
    program, qubits, dims = _qutrit_circuit_5()
    noise_model = _depolarizing_model(
        [Gate("TSWAP", [], [2, 7]), Gate("TSWAP", [], [9, 4])], fidelity=0.99
    )
    _assert_real_compression(program, qubits, noise_model)

    oracle = _oracle_density(program, qubits, noise_model)
    rho_est, _ = _trajectory_density(
        program, qubits, dims, n_traj=8000, noise_model=noise_model, seed=4
    )
    assert float(qx.fidelity(rho_est, oracle)) > 0.97


# ══════════════════════════════════════════════════════════
# 9-10.  Gates + measurements, no noise
# ══════════════════════════════════════════════════════════


def test_compression_two_qubit_measurements_no_noise():
    """2-qubit Bell circuit with measurements: outcomes are perfectly correlated."""
    program = Program()
    program += H(5)
    program += CNOT(5, 2)
    program += Measurement(qubit=Qubit(5), classical_reg=None)
    program += Measurement(qubit=Qubit(2), classical_reg=None)
    qubits, dims = QUBITS_2, (2, 2)

    _assert_real_compression(program, qubits)

    oracle = _oracle_density(program, qubits)
    rho_est, outcomes = _trajectory_density(program, qubits, dims, n_traj=6000, seed=5)
    # Trajectory-averaged (post-measurement) state reproduces the oracle mixture.
    assert float(qx.fidelity(rho_est, oracle)) > 0.99

    # Bell state -> only 00 and 11 ever occur, each ~50%.
    dist = _outcome_distribution(outcomes, dims)
    oracle_diag = np.real(np.diag(np.asarray(oracle.matrix)))
    assert _total_variation(dist, oracle_diag) < 0.05
    assert dist[1] < 1e-9 and dist[2] < 1e-9  # |01> and |10> never measured


def test_compression_five_qubit_measurements_no_noise():
    """5-qubit circuit with terminal measurements on all qubits."""
    program, qubits, dims = _qubit_circuit_5()
    for q in qubits:
        program += Measurement(qubit=Qubit(q), classical_reg=None)

    _assert_real_compression(program, qubits)

    oracle = _oracle_density(program, qubits)
    rho_est, outcomes = _trajectory_density(program, qubits, dims, n_traj=8000, seed=6)
    assert float(qx.fidelity(rho_est, oracle)) > 0.98

    dist = _outcome_distribution(outcomes, dims)
    oracle_diag = np.real(np.diag(np.asarray(oracle.matrix)))
    assert _total_variation(dist, oracle_diag) < 0.06


# ══════════════════════════════════════════════════════════
# 11-12.  Gates + measurements, with noise
# ══════════════════════════════════════════════════════════


def test_compression_two_qubit_measurements_with_noise():
    """2-qubit Bell circuit with gate noise and readout error."""
    program = Program()
    program += H(5)
    program += CNOT(5, 2)
    meas5 = Measurement(qubit=Qubit(5), classical_reg=None)
    meas2 = Measurement(qubit=Qubit(2), classical_reg=None)
    program += meas5
    program += meas2
    qubits, dims = QUBITS_2, (2, 2)

    noise_model = NoiseModel.from_channels(
        [
            Channel.from_gate_fidelity(inst=CNOT(5, 2), fidelity=0.97),
            Channel.from_gate_fidelity(inst=H(5), fidelity=0.98),
            MeasurementChannel.from_readout_fidelity(inst=meas5, fidelity=0.95),
            MeasurementChannel.from_readout_fidelity(inst=meas2, fidelity=0.95),
        ]
    )
    _assert_real_compression(program, qubits, noise_model)

    oracle = _oracle_density(program, qubits, noise_model)
    rho_est, outcomes = _trajectory_density(
        program, qubits, dims, n_traj=8000, noise_model=noise_model, seed=7
    )
    assert float(qx.fidelity(rho_est, oracle)) > 0.99

    # Readout confusion error perturbs the reported outcomes (not the QND
    # post-measurement state), so the outcome distribution is compared against
    # the uncompressed run rather than the density-matrix diagonal.
    dist = _outcome_distribution(outcomes, dims)
    _, outcomes_uncompressed = _trajectory_density(
        program, qubits, dims, n_traj=8000, noise_model=noise_model, seed=7,
        max_subsystem_size=0,
    )
    dist_uncompressed = _outcome_distribution(outcomes_uncompressed, dims)
    assert _total_variation(dist, dist_uncompressed) < 0.04


def test_compression_five_qubit_measurements_with_noise():
    """5-qubit circuit with gate noise, readout error, and terminal measurements."""
    program, qubits, dims = _qubit_circuit_5()
    measurements = []
    for q in qubits:
        m = Measurement(qubit=Qubit(q), classical_reg=None)
        measurements.append(m)
        program += m

    noise_model = NoiseModel.from_channels(
        [
            Channel.from_gate_fidelity(inst=CZ(7, 2), fidelity=0.99),
            Channel.from_gate_fidelity(inst=CZ(9, 4), fidelity=0.99),
            Channel.from_gate_fidelity(inst=CNOT(2, 9), fidelity=0.99),
            Channel.from_gate_fidelity(inst=CNOT(4, 0), fidelity=0.99),
            Channel.from_gate_fidelity(inst=CNOT(0, 7), fidelity=0.99),
        ]
        + [MeasurementChannel.from_readout_fidelity(inst=m, fidelity=0.96) for m in measurements]
    )
    _assert_real_compression(program, qubits, noise_model)

    oracle = _oracle_density(program, qubits, noise_model)
    rho_est, outcomes = _trajectory_density(
        program, qubits, dims, n_traj=12000, noise_model=noise_model, seed=8
    )
    assert float(qx.fidelity(rho_est, oracle)) > 0.97

    # Readout confusion error perturbs the reported outcomes (not the QND
    # post-measurement state), so the outcome distribution is compared against
    # the uncompressed run rather than the density-matrix diagonal.
    dist = _outcome_distribution(outcomes, dims)
    _, outcomes_uncompressed = _trajectory_density(
        program, qubits, dims, n_traj=12000, noise_model=noise_model, seed=8,
        max_subsystem_size=0,
    )
    dist_uncompressed = _outcome_distribution(outcomes_uncompressed, dims)
    assert _total_variation(dist, dist_uncompressed) < 0.05


# ══════════════════════════════════════════════════════════
# Regression: measurement-outcome column order under compression
# ══════════════════════════════════════════════════════════


def test_measurement_outcome_column_order_under_compression():
    """Outcome columns follow ``MEASURE`` program order even when gates merge.

    The compressor merges gates into multi-qudit groups, which reorders the
    topological emission of operations.  Measurement nodes must nonetheless be
    emitted in program order so that ``outcomes[:, i]`` corresponds to the
    *i*-th ``MEASURE`` instruction.  This pins down that invariant directly by
    preparing a distinct, deterministic basis state on every qubit and checking
    each column independently — at ``max_subsystem_size=2`` (compressed) against
    both the analytic expectation and the uncompressed (``max=0``) run.
    """
    qubits = QUBITS_5  # [7, 2, 9, 4, 0]
    dims = (2, 2, 2, 2, 2)

    # X on qubits 7, 9, 0 -> per-slot states [1, 0, 1, 0, 1] for slots 0..4.
    program = Program()
    program += X(7)
    program += X(9)
    program += X(0)
    # Single-qubit gates flanking an entangler force real merges around the
    # measurement barriers without changing the deterministic outcomes.
    program += H(2)
    program += H(2)        # H*H = I on slot 1, but creates a mergeable group
    program += CNOT(7, 9)  # both already |1>: CNOT leaves 7->1, 9 flips 1->0
    program += CNOT(7, 9)  # flip back: 9 -> 1, restoring [1,0,1,0,1]
    # Measure in program order: qubits 7, 2, 9, 4, 0 == slots 0, 1, 2, 3, 4.
    for q in qubits:
        program += Measurement(qubit=Qubit(q), classical_reg=None)

    _assert_real_compression(program, qubits)

    expected = np.array([1, 0, 1, 0, 1])

    _, outcomes_compressed = _trajectory_density(
        program, qubits, dims, n_traj=16, seed=0, max_subsystem_size=2
    )
    _, outcomes_uncompressed = _trajectory_density(
        program, qubits, dims, n_traj=16, seed=0, max_subsystem_size=0
    )

    # Every trajectory is deterministic; every column must match program order.
    assert outcomes_compressed.shape == (16, 5)
    assert np.all(outcomes_compressed == expected)
    assert np.all(outcomes_uncompressed == expected)
    # Compressed and uncompressed must agree column-for-column.
    assert np.array_equal(outcomes_compressed, outcomes_uncompressed)


# ══════════════════════════════════════════════════════════
# Tightened: deterministic total-channel equality across sizes
# ══════════════════════════════════════════════════════════


def _noisy_channel_cases():
    """Yield ``(id, program, qubits, noise_model)`` for every noisy circuit.

    These reuse the noisy circuits exercised by the Monte-Carlo tests above for
    a deterministic, sampling-free channel comparison.  The 5-qutrit circuit is
    deliberately omitted: its dense superoperator is ``243**2 x 243**2`` (~52
    GiB), which is intractable.  That case stays covered by the Monte-Carlo
    fidelity test; the cases below already span qubit and qutrit registers,
    non-sorted multi-qudit gates, and 2- and 5-register sizes.
    """
    p2, q2, _ = _qubit_circuit_2()
    yield (
        "qubit2",
        p2,
        q2,
        _depolarizing_model([CNOT(2, 5), X(5), H(2), H(5)], fidelity=0.97),
    )

    p5, q5, _ = _qubit_circuit_5()
    yield (
        "qubit5",
        p5,
        q5,
        _depolarizing_model(
            [CZ(7, 2), CZ(9, 4), CNOT(2, 9), CNOT(4, 0), CNOT(0, 7)], fidelity=0.99
        ),
    )

    pq2, qq2, _ = _qutrit_circuit_2()
    yield (
        "qutrit2",
        pq2,
        qq2,
        _depolarizing_model(
            [Gate("TSWAP", [], [2, 5]), Gate("TX", [], [5]), Gate("TH", [], [5])],
            fidelity=0.98,
        ),
    )


@pytest.mark.parametrize("case", tuple(_noisy_channel_cases()), ids=lambda c: c[0])
def test_compression_preserves_total_channel_exactly(case):
    """Compression must not change the *channel* the sampler implements.

    The Monte-Carlo fidelity tests above can only resolve an error rate to
    within their statistical noise (~1e-2).  A *subtly* wrong error rate at
    ``max_subsystem_size=2`` — exactly the symptom under investigation — would
    slip beneath that floor.  This test removes the sampling entirely: it builds
    the exact dense superoperator the trajectory sampler implements at each
    ``max_subsystem_size`` and asserts the compressed channel equals the
    uncompressed one to machine precision.

    Because the channel is reconstructed from
    :func:`pyquil.simulation._simulator._op_to_kraus_matrix`, this directly
    exercises the merged-Kraus padding/embedding path that only compression
    triggers, for both qubit (d=2) and qutrit (d=3) registers including
    non-sorted multi-qudit gates.
    """
    _id, program, qubits, noise_model = case

    reference = _sampler_total_channel(
        program, qubits, noise_model=noise_model, max_subsystem_size=0
    )
    for size in (1, 2):
        channel = _sampler_total_channel(
            program, qubits, noise_model=noise_model, max_subsystem_size=size
        )
        max_abs_diff = float(np.abs(channel - reference).max())
        assert max_abs_diff < 1e-10, (
            f"compression at max_subsystem_size={size} changed the channel "
            f"(max |ΔS| = {max_abs_diff:.2e})"
        )


# ══════════════════════════════════════════════════════════
# Tightened: per-qubit error rate is mapped to the correct column
# ══════════════════════════════════════════════════════════


def test_asymmetric_readout_error_rate_column_mapping_under_compression():
    """Each qubit's readout error rate stays on its own outcome column.

    A column-permutation bug under compression would swap which qubit a measured
    error rate is attributed to.  A *symmetric* readout model cannot detect such
    a swap, so this test gives the two measured qubits **very different** error
    rates (one ~30%, one ~1%) and interleaves a mergeable gate block between the
    two ``MEASURE`` instructions.  At ``max_subsystem_size=2`` the gate block
    collapses to a single merged operator sitting between the measurement
    barriers — the configuration most likely to disturb measurement emission
    order — yet each column must still report its own qubit's error rate.
    """
    qubits = QUBITS_2  # [5, 2]

    program = Program()
    program += X(2)  # slot 1 (qubit 2) -> |1>, slot 0 (qubit 5) -> |0>
    meas2 = Measurement(qubit=Qubit(2), classical_reg=None)  # program-order col 0
    meas5 = Measurement(qubit=Qubit(5), classical_reg=None)  # program-order col 1
    program += meas2
    # Mergeable gate block between the two measurements (identity on the state:
    # H(5) H(5) = I, CNOT(2,5) CNOT(2,5) = I) but it forces a real 2-qubit merge.
    program += H(5)
    program += CNOT(2, 5)
    program += CNOT(2, 5)
    program += H(5)
    program += meas5

    # qubit 2 measured |1> with 30% flip (fidelity 0.70); qubit 5 measured |0>
    # with 1% flip (fidelity 0.99).  The two error rates are deliberately far
    # apart so any column swap is unmistakable.
    noise_model = NoiseModel.from_channels(
        [
            MeasurementChannel.from_readout_fidelity(inst=meas2, fidelity=0.70),
            MeasurementChannel.from_readout_fidelity(inst=meas5, fidelity=0.99),
        ]
    )
    _assert_real_compression(program, qubits, noise_model)

    n_traj = 20000
    dims = (2, 2)
    for size in (0, 1, 2):
        _, outcomes = _trajectory_density(
            program, qubits, dims, n_traj, noise_model=noise_model,
            seed=0, max_subsystem_size=size,
        )
        # Column 0 == qubit 2 (|1>, 30% flip to 0) -> P(=1) ~ 0.70.
        # Column 1 == qubit 5 (|0>, 1% flip to 1)  -> P(=1) ~ 0.01.
        p_col0_one = outcomes[:, 0].mean()
        p_col1_one = outcomes[:, 1].mean()
        assert abs(p_col0_one - 0.70) < 0.02, (
            f"size {size}: qubit-2 error rate landed on the wrong column "
            f"(col0 P(=1)={p_col0_one:.3f}, expected ~0.70)"
        )
        assert abs(p_col1_one - 0.01) < 0.01, (
            f"size {size}: qubit-5 error rate landed on the wrong column "
            f"(col1 P(=1)={p_col1_one:.3f}, expected ~0.01)"
        )


# ══════════════════════════════════════════════════════════
# Regression: compression must not reorder mid-circuit measurements
# ══════════════════════════════════════════════════════════


def test_compression_does_not_merge_gates_across_mid_circuit_measurement():
    """Gates straddling a mid-circuit measurement must not be fused.

    This pins down the bug that produced a *different logical error rate* under
    compression in repetition-code experiments.  Two gates that act on the same
    qubits but sit on opposite sides of a mid-circuit ``MEASURE`` form a DAG
    edge (they share a qubit), so a size-only merge check happily fuses them —
    which silently moves the measurement to *after* both gates.  In a circuit
    with repeated syndrome extraction this corrupts every round and inflates the
    logical error rate.

    The compressor must keep the merged group *convex*: it may only fuse two
    operations if no barrier (or any other operation) lies on a dependency path
    between them.  Here a non-trivial unitary surrounds a mid-circuit
    measurement, so the merged ``emit_order`` at ``max_subsystem_size=2`` must
    still place the measurement between the two gate groups, and the sampled
    outcome distribution must match the uncompressed run.
    """
    qubits = QUBITS_2  # [5, 2]
    dims = (2, 2)

    program = Program()
    program += H(5)
    program += CNOT(5, 2)  # entangle 5 and 2
    mid = Measurement(qubit=Qubit(2), classical_reg=None)  # collapse slot 1
    program += mid
    program += H(2)
    program += CNOT(5, 2)  # acts on the same pair again, *after* the measurement
    program += Measurement(qubit=Qubit(5), classical_reg=None)
    program += Measurement(qubit=Qubit(2), classical_reg=None)

    _assert_real_compression(program, qubits)

    # The mid-circuit measurement must remain sandwiched between the two
    # two-qubit gate groups; the two CNOTs must NOT be fused into one operator.
    sim = TrajectorySimulator(
        program, qubits=qubits, max_subsystem_size=MAX_SUBSYSTEM_SIZE
    )
    emitted = sim.compress(sim.resolve(sim.linearize({})))
    op_types = [type(op).__name__ for op, _ in emitted]
    # Exactly three QuantumInstruments (one mid-circuit + two terminal) and the
    # first must appear before the second gate group — i.e. there is a gate
    # operation emitted *after* the first measurement.
    first_measure = op_types.index("QuantumInstrument")
    assert any(
        name != "QuantumInstrument" for name in op_types[first_measure + 1 :]
    ), f"a gate group must follow the mid-circuit measurement, got {op_types}"

    # The sampled outcome statistics must be independent of compression.  There
    # are three measurements (mid-circuit on slot 1, then terminal on slots 0
    # and 1), so the joint distribution is histogrammed over 2**3 = 8 patterns.
    def _joint(max_subsystem_size):
        _, outcomes = _trajectory_density(
            program, qubits, dims, 40000, seed=0,
            max_subsystem_size=max_subsystem_size,
        )
        codes = outcomes[:, 0] * 4 + outcomes[:, 1] * 2 + outcomes[:, 2]
        return np.bincount(codes, minlength=8) / len(codes)

    assert _total_variation(_joint(0), _joint(2)) < 0.02


