"""Benchmarks for the quax-backed state vector and trajectory simulators."""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, RX, RZ
from pyquil.noise._channels import Channel, CycleChannel, MeasurementChannel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilbase import DefCircuit, ResetQubit
from pyquil.quilbase import Gate as QuilGate
from pyquil.quilbase import Measurement as QuilMeasurement
from pyquil.quilbase import Reset as QuilReset
from pyquil.simulation._simulator import (
    DensityMatrixSimulator,
    PureStateVectorSimulator,
    TrajectorySimulator,
)
from pyquil.simulation._simulator import (
    _apply_trajectory_operations as apply_trajectory_operations,
)

_EMPTY_PARAMS = np.array([], dtype=float)
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_SURFACE17_FIXTURE = _FIXTURES_DIR / "surface_17_depth_5_no_reset.quil"
_SURFACE17_QUBITS = (65, 66, 74, 75, 76, 77, 82, 83, 84, 85, 86, 91, 92, 93, 94, 102, 103)
_SURFACE17_CYCLES = {
    "SZ_INIT",
    "SX_INIT",
    "CZ_0",
    "SZ_DATA",
    "SX_DATA",
    "CZ_1",
    "CZ_2",
    "CZ_3",
    "SZ_ANCILLA",
    "SX_ANCILLA_ECHO",
    "MEASURE_ANCILLA",
    "MEASURE_ALL",
}
_DEFAULT_NUM_QUBITS = 15
_DEFAULT_NUM_LAYERS = 10
_DEFAULT_NUM_TRAJECTORIES = 128
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_SUBSYSTEM_SIZE = 1


def _build_noisy_program_and_model(num_qubits, num_layers, seed=4867):
    """Build a layered noisy circuit and matching noise model for shared scaling benchmarks."""
    edges_0 = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    edges_1 = [(i, i + 1) for i in range(1, num_qubits - 1, 2)]
    rng = np.random.default_rng(seed)

    t1s, t2s = {}, {}
    for q in range(num_qubits):
        t1 = np.clip(rng.normal(30, 10), 10, 50)
        t2 = np.clip(rng.normal(30, 20), 5, 2 * t1)
        t1s[q], t2s[q] = t1, t2

    channels = [
        Channel.from_coherence_times(
            CNOT(*edge), gate_duration=0.1, t1s=[t1s[q] for q in edge], t2s=[t2s[q] for q in edge]
        )
        for edge in edges_0 + edges_1
    ] + [
        Channel.from_coherence_times(RX(np.pi / 2, q), gate_duration=0.04, t1s=[t1s[q]], t2s=[t2s[q]])
        for q in range(num_qubits)
    ]
    noise_model = NoiseModel(channels=channels)

    program = Program()
    for _ in range(num_layers):
        for edges in [edges_0, edges_1]:
            program += [RZ(rng.uniform(-np.pi, np.pi), idx) for idx in range(num_qubits)]
            program += [RX(np.pi / 2, idx) for idx in range(num_qubits)]
            program += [RZ(rng.uniform(-np.pi, np.pi), idx) for idx in range(num_qubits)]
            program += [RX(np.pi / 2, idx) for idx in range(num_qubits)]
            program += [RZ(rng.uniform(-np.pi, np.pi), idx) for idx in range(num_qubits)]
            program += [CNOT(*edge) for edge in edges]

    return program, noise_model


def _surface17_defcircuits(program):
    return {inst.name: inst for inst in program.instructions if isinstance(inst, DefCircuit)}


def _surface17_program_variant(variant="full"):
    program = Program(_SURFACE17_FIXTURE.read_text())
    if variant == "full":
        return program

    if variant == "first_measurement":
        prefix_program = Program()
        for inst in program.instructions:
            prefix_program += inst
            if isinstance(inst, QuilGate) and inst.name == "MEASURE_ANCILLA":
                break
        return prefix_program

    if variant == "no_measurements":
        keep_gates, keep_measurements = True, False
    elif variant == "measurements_only":
        keep_gates, keep_measurements = False, True
    else:
        raise ValueError(f"Unknown surface-17 variant: {variant}")

    original_defcircuits = _surface17_defcircuits(program)
    filtered_defcircuits = {}
    filtered_program = Program()

    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            instructions = [
                cycle_inst
                for cycle_inst in inst.instructions
                if (keep_gates and isinstance(cycle_inst, QuilGate))
                or (keep_measurements and isinstance(cycle_inst, QuilMeasurement))
            ]
            if instructions:
                defcircuit = DefCircuit(inst.name, inst.parameters, inst.qubit_variables, instructions)
                filtered_defcircuits[inst.name] = defcircuit
                filtered_program += defcircuit
        elif isinstance(inst, QuilGate) and inst.name in original_defcircuits:
            if inst.name in filtered_defcircuits:
                filtered_program += inst
        else:
            filtered_program += inst

    return filtered_program


def _build_surface17_cycle_noise_model(program, depolarizing_constant=0.99, readout_fidelity=1.0):
    defcircuits = _surface17_defcircuits(program)
    cycle_channels = []

    for inst in program.instructions:
        if not isinstance(inst, QuilGate) or inst.name not in defcircuits:
            continue

        defcircuit = defcircuits[inst.name]
        qubit_map = dict(zip(defcircuit.qubit_variables, inst.qubits))
        channels = []

        for cycle_inst in defcircuit.instructions:
            if isinstance(cycle_inst, QuilGate):
                concrete_gate = QuilGate(
                    cycle_inst.name,
                    list(cycle_inst.params),
                    [qubit_map[qubit] for qubit in cycle_inst.qubits],
                )
                channels.append(Channel.from_depolarizing_constant(concrete_gate, depolarizing_constant))
            elif isinstance(cycle_inst, QuilMeasurement):
                concrete_measurement = QuilMeasurement(qubit=qubit_map[cycle_inst.qubit], classical_reg=None)
                channels.append(
                    MeasurementChannel.from_readout_fidelity(concrete_measurement, fidelity=readout_fidelity)
                )

        cycle_channels.append(CycleChannel(inst=inst, defcircuit=defcircuit, channels=tuple(channels)))

    return NoiseModel(channels=cycle_channels)


def _prepare_trajectory_operations(program, noise_model, max_subsystem_size=0):
    sim = TrajectorySimulator(program, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
    params = sim.linearize({})
    resolved = sim.resolve(params)
    compressed = sim.compress(resolved)
    operations = sim.adapt(compressed)
    return sim, resolved, compressed, operations


def _operation_counts(operations):
    return {
        "unitary_ops": sum(1 for op, _ in operations if isinstance(op, qx.Unitary)),
        "kraus_ops": sum(1 for op, _ in operations if isinstance(op, qx.KrausMap)),
        "instrument_ops": sum(1 for op, _ in operations if isinstance(op, qx.QuantumInstrument)),
    }


def _record_counts(benchmark, resolved, compressed, operations):
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(
            {
                "resolved_ops": len(resolved),
                "compressed_ops": len(compressed),
                "trajectory_ops": len(operations),
                **_operation_counts(operations),
            }
        )


def _block_until_ready(matrix, outcomes):
    matrix.block_until_ready()
    outcomes.block_until_ready()


def _benchmark_trajectory_operations(
    benchmark,
    sim,
    operations,
    *,
    num_trajectories,
    batch_size,
    random_seed=0,
    use_jit=False,
):
    if use_jit:

        def apply_matrix(matrix, key):
            psi = qx.StateVector.from_matrix(matrix, sim.dims)
            psi_out, outcomes = apply_trajectory_operations(operations, psi, key)
            return psi_out.matrix, outcomes

        apply_batch = jax.jit(apply_matrix)
    else:

        def apply_batch(matrix, key):
            psi = qx.StateVector.from_matrix(matrix, sim.dims)
            psi_out, outcomes = apply_trajectory_operations(operations, psi, key)
            return psi_out.matrix, outcomes

    warmup_psi = qx.zero_state_vector(dims=sim.dims, ensemble_size=(batch_size,))
    _block_until_ready(*apply_batch(warmup_psi.matrix, jax.random.key(random_seed)))

    def thunk():
        key = jax.random.key(random_seed)
        remaining = num_trajectories
        while remaining > 0:
            this_batch = min(remaining, batch_size)
            key, batch_key = jax.random.split(key)
            psi = qx.zero_state_vector(dims=sim.dims, ensemble_size=(this_batch,))
            _block_until_ready(*apply_batch(psi.matrix, batch_key))
            remaining -= this_batch

    benchmark.pedantic(thunk, iterations=1, rounds=1)


def _run_perf_benchmark(
    benchmark,
    num_qubits=_DEFAULT_NUM_QUBITS,
    num_layers=_DEFAULT_NUM_LAYERS,
    num_trajectories=_DEFAULT_NUM_TRAJECTORIES,
    batch_size=_DEFAULT_BATCH_SIZE,
    max_subsystem_size=_DEFAULT_MAX_SUBSYSTEM_SIZE,
):
    program, noise_model = _build_noisy_program_and_model(num_qubits, num_layers)
    sim, resolved, compressed, operations = _prepare_trajectory_operations(program, noise_model, max_subsystem_size)
    _record_counts(benchmark, resolved, compressed, operations)
    _benchmark_trajectory_operations(
        benchmark,
        sim,
        operations,
        num_trajectories=num_trajectories,
        batch_size=batch_size,
    )


def _run_surface17_benchmark(
    benchmark,
    *,
    variant="full",
    num_trajectories=128,
    batch_size=16,
    max_subsystem_size=2,
    depolarizing_constant=0.99,
    readout_fidelity=1.0,
    use_jit=False,
):
    program = _surface17_program_variant(variant)
    noise_model = _build_surface17_cycle_noise_model(
        program,
        depolarizing_constant=depolarizing_constant,
        readout_fidelity=readout_fidelity,
    )
    sim, resolved, compressed, operations = _prepare_trajectory_operations(program, noise_model, max_subsystem_size)
    _record_counts(benchmark, resolved, compressed, operations)
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(
            {
                "variant": variant,
                "num_trajectories": num_trajectories,
                "batch_size": batch_size,
                "max_subsystem_size": max_subsystem_size,
                "use_jit": use_jit,
            }
        )
    _benchmark_trajectory_operations(
        benchmark,
        sim,
        operations,
        num_trajectories=num_trajectories,
        batch_size=batch_size,
        use_jit=use_jit,
    )


def test_surface17_fixture_structure():
    program = Program(_SURFACE17_FIXTURE.read_text())
    defcircuits = {inst.name: inst for inst in program.instructions if isinstance(inst, DefCircuit)}
    invocations = [inst for inst in program.instructions if isinstance(inst, QuilGate)]
    invocation_names = [inst.name for inst in invocations]

    assert _SURFACE17_FIXTURE.exists()
    assert set(defcircuits) == _SURFACE17_CYCLES
    assert set(program.get_qubit_indices()) == set(_SURFACE17_QUBITS)
    assert not any(isinstance(inst, (QuilReset, ResetQubit)) for inst in program.instructions)
    assert invocation_names.count("MEASURE_ANCILLA") == 4
    assert invocation_names[-1] == "MEASURE_ALL"


def test_surface17_cycle_noise_model_preserves_measurements():
    program = Program(_SURFACE17_FIXTURE.read_text())
    noise_model = _build_surface17_cycle_noise_model(program, depolarizing_constant=1.0)
    sim, _, _, _ = _prepare_trajectory_operations(program, noise_model, max_subsystem_size=0)
    resolved = sim.resolve(_EMPTY_PARAMS)

    assert sum(1 for op, _ in resolved if isinstance(op, qx.QuantumInstrument)) == 49


@pytest.mark.parametrize("variant", ["full", "no_measurements", "measurements_only"])
def test_surface17_cycle_noise_compression_preserves_instruments(variant):
    program = _surface17_program_variant(variant)
    noise_model = _build_surface17_cycle_noise_model(program)
    counts = []

    for max_subsystem_size in (0, 1, 2):
        _, resolved, compressed, operations = _prepare_trajectory_operations(program, noise_model, max_subsystem_size)
        counts.append((len(resolved), len(compressed), len(operations), _operation_counts(operations)))

    assert counts[0][0] == counts[1][0] == counts[2][0]
    if variant == "measurements_only":
        assert counts[0] == counts[1] == counts[2]
    else:
        assert counts[2][1] < counts[1][1] < counts[0][1]
    assert counts[0][3]["instrument_ops"] == counts[1][3]["instrument_ops"] == counts[2][3]["instrument_ops"]


class TestPerformance:
    """Trajectory simulator performance benchmarks."""

    @pytest.mark.parametrize(
        "num_qubits",
        [
            pytest.param(3, id="3q"),
            pytest.param(6, id="6q"),
            pytest.param(9, id="9q"),
            pytest.param(12, id="12q"),
            pytest.param(15, id="15q"),
        ],
    )
    def test_scaling_qubits(self, benchmark, num_qubits):
        _run_perf_benchmark(benchmark, num_qubits=num_qubits)

    @pytest.mark.parametrize(
        "num_layers",
        [
            pytest.param(1, id="1L"),
            pytest.param(3, id="3L"),
            pytest.param(10, id="10L"),
            pytest.param(20, id="20L"),
        ],
    )
    def test_scaling_depth(self, benchmark, num_layers):
        _run_perf_benchmark(benchmark, num_layers=num_layers)

    @pytest.mark.parametrize(
        "batch_size",
        [
            pytest.param(8, id="b8"),
            pytest.param(16, id="b16"),
            pytest.param(64, id="b64"),
        ],
    )
    def test_scaling_batch_size(self, benchmark, batch_size):
        _run_perf_benchmark(benchmark, batch_size=batch_size)

    @pytest.mark.parametrize(
        "max_subsystem_size",
        [
            pytest.param(0, id="s0"),
            pytest.param(1, id="s1"),
        ],
    )
    def test_scaling_subsystem_size(self, benchmark, max_subsystem_size):
        _run_perf_benchmark(benchmark, max_subsystem_size=max_subsystem_size)

    @pytest.mark.parametrize(
        "batch_size",
        [
            pytest.param(8, id="b8"),
            pytest.param(16, id="b16"),
            pytest.param(64, id="b64"),
        ],
    )
    def test_17q_batch_size(self, benchmark, batch_size):
        _run_perf_benchmark(benchmark, num_qubits=17, batch_size=batch_size)

    def test_surface17_depth5_cycle_noise(self, benchmark):
        _run_surface17_benchmark(benchmark)

    def test_surface17_depth5_cycle_noise_low_trajectory(self, benchmark):
        _run_surface17_benchmark(benchmark, num_trajectories=4, batch_size=4)

    def test_surface17_depth5_cycle_noise_micro(self, benchmark):
        _run_surface17_benchmark(benchmark, variant="first_measurement", num_trajectories=1, batch_size=1)

    def test_surface17_depth5_cycle_noise_micro_jit(self, benchmark):
        _run_surface17_benchmark(benchmark, variant="first_measurement", num_trajectories=1, batch_size=1, use_jit=True)

    def test_surface17_depth5_cycle_noise_no_measurements_micro(self, benchmark):
        _run_surface17_benchmark(benchmark, variant="no_measurements", num_trajectories=4, batch_size=4)

    def test_surface17_depth5_cycle_noise_measurements_only_micro(self, benchmark):
        _run_surface17_benchmark(benchmark, variant="measurements_only", num_trajectories=4, batch_size=4)


def _build_gate_program(num_qubits, num_layers, seed=4867):
    """Build a layered gate-only (noise-free) brickwork circuit."""
    rng = np.random.default_rng(seed)
    program = Program()
    for _ in range(num_layers):
        for q in range(num_qubits):
            program += RX(rng.uniform(-np.pi, np.pi), q)
            program += RZ(rng.uniform(-np.pi, np.pi), q)
        for q in range(0, num_qubits - 1, 2):
            program += CNOT(q, q + 1)
        for q in range(1, num_qubits - 1, 2):
            program += CNOT(q, q + 1)
    return program


def _benchmark_compile_time(benchmark, sim, params, extra=None):
    """Benchmark the JIT compile time of ``sim.compute``.

    A fresh ``jax.jit`` wrapper is created on every round so the XLA compilation
    cache is bypassed and the full lower+compile cost is measured each time.
    """
    if hasattr(benchmark, "extra_info") and extra:
        benchmark.extra_info.update(extra)

    def thunk():
        return jax.jit(lambda p: sim.compute(p)).lower(params).compile()

    benchmark.pedantic(thunk, iterations=1, rounds=1)


class TestJitCompileTime:
    """JIT compile-time benchmarks for the lax-loop simulators.

    These measure how compilation time scales with program depth.  The lax-loop
    ``compute`` traces a single loop body plus one switch branch per distinct
    base subsystem, so the compiled graph size is bounded by the number of
    distinct subsystems rather than the number of operations.
    """

    @pytest.mark.parametrize(
        "num_layers",
        [
            pytest.param(10, id="10L"),
            pytest.param(40, id="40L"),
            pytest.param(80, id="80L"),
        ],
    )
    def test_state_vector_compile_depth(self, benchmark, num_layers):
        num_qubits = 10
        program = _build_gate_program(num_qubits, num_layers)
        sim = PureStateVectorSimulator(program, qubits=list(range(num_qubits)))
        params = sim.linearize({})
        _benchmark_compile_time(
            benchmark,
            sim,
            params,
            extra={
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "num_ops": len(program.instructions),
                "num_bases": len(sim.bases),
            },
        )

    @pytest.mark.parametrize(
        "num_layers",
        [
            pytest.param(5, id="5L"),
            pytest.param(20, id="20L"),
            pytest.param(40, id="40L"),
        ],
    )
    def test_density_matrix_compile_depth(self, benchmark, num_layers):
        num_qubits = 6
        program = _build_gate_program(num_qubits, num_layers)
        sim = DensityMatrixSimulator(program, qubits=list(range(num_qubits)))
        params = sim.linearize({})
        _benchmark_compile_time(
            benchmark,
            sim,
            params,
            extra={
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "num_ops": len(program.instructions),
                "num_bases": len(sim.bases),
            },
        )
