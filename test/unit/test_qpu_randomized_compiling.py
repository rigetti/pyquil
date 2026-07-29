"""Unit tests for `pyquil._qpu.randomized_compiling`.

Note, we test several underlying internal features here where much of the complexity lies. These are indeed
implementation details but they provide a robust scaffolding for testing the overall correctness of the
randomized compiling implementation.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from qcs_sdk.qpu.experimental.random import PrngSeedValue, choose_random_real_sub_region_indices
from quil import instructions as inst
from syrupy.assertion import SnapshotAssertion

from pyquil import Program, gates
from pyquil._qpu import randomized_compiling as rc
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Call, Declare
from pyquil.simulation import matrices


@pytest.mark.parametrize(
    "seed, sequence_length, expected_value",
    (
        (231018089914722, 7864, 72130689498700),
        (48337553834185, 348, 222126916911797),
        (238228272237383, 9224, 68566226079270),
        (99359104223091, 191, 102235786504924),
        (152395841476258, 7155, 150130174624373),
        (7208143265828, 8430, 172575543901836),
        (150372856527354, 5036, 32355578502344),
        (11890978449387, 8599, 50433886160564),
        (32925875164545, 9860, 119857684629909),
        (1518961100800, 5757, 131378683707906),
    ),
)
def test_lfsr_v1_next(seed: int, sequence_length: int, expected_value: int):
    """Test that `rc._lfsr_v1_next` produces the expected value after a given number of iterations for a given seed.

    The test cases here were randomly generated and the expected value was generated from the QCS SDK.

    Note that `qcs_sdk.qpu.experimental.random.lfsr_v1_next` is _the_ source of truth for the control system PRNG. However, because the current
    version of the QCS SDK does not expose the inner value of `PrngSeedValue` over the Python API, we have to re-implement the functionality
    here as a stopgap until we can update the QCS SDK. In the meantime, this test simply tests the Python results against results
    generated from an updated version of the QCS SDK.
    """
    for _ in range(sequence_length):
        seed = rc._lfsr_v1_next(seed)
    assert (
        seed == expected_value
    ), f"Expected {expected_value} but found {seed} after {sequence_length} iterations of _lfsr_v1_next starting from {seed}"


def test_pauli_conjugates_map():
    """Test that the `rc.PAULI_CONJUGATES_MAPS` defines accurate conjugations.

    Specifically, this checks that multiplying the conjugages, the 2Q gate, and the previous paulis produces the
    2Q gate unitary, up to a global phase.
    """
    for gate_name, pauli_conjugates_map in rc.PAULI_CONJUGATES_MAPS.items():
        gate_matrix = getattr(matrices, gate_name)
        for previous_paulis, next_paulis in pauli_conjugates_map.items():
            previous = np.kron(previous_paulis[0].matrix, previous_paulis[1].matrix)
            next_ = np.kron(next_paulis[0].matrix, next_paulis[1].matrix)
            result = next_ @ gate_matrix @ previous
            assert rc._unitary_equal(
                result, gate_matrix
            ), f"Failed for gate {gate_name} with previous {previous_paulis} and next {next_paulis}"


def test_pauli_literal_multiplication():
    """Test that multiplying two `rc.PauliLiteral`s produces the expected coefficient and resulting `rc.PauliLiteral`."""
    for pauli_left, pauli_right in product(rc.PauliLiteral.all(), repeat=2):
        coefficient, result = pauli_left * pauli_right
        assert np.allclose(
            coefficient * result.matrix, pauli_left.matrix @ pauli_right.matrix
        ), f"Failed for {pauli_left} * {pauli_right}"


def _get_expected_random_pauli(
    seeds: list[int], layer_index: int, layer_count: int, sequence_index: int, paulis_per_value: int, accumulate: bool
) -> tuple[Optional[int], rc.PauliLiteral]:
    if layer_index == layer_count - 1:
        return None, rc.PauliLiteral.I
    seed_index = layer_index // paulis_per_value
    pauli_index = layer_index % paulis_per_value
    seed = seeds[seed_index]
    if accumulate:
        accumulated_pauli_value = None
        for value in rc._generate_lfsr_v1_sequence(seed, 1, sequence_index + 1):
            pauli_value = rc.PauliLiteral((value >> (2 * pauli_index)) & 0b11)
            if accumulated_pauli_value is None:
                accumulated_pauli_value = pauli_value
            else:
                _, accumulated_pauli_value = accumulated_pauli_value * pauli_value
            seed = value
        return seed, cast(rc.PauliLiteral, accumulated_pauli_value)
    else:
        seed = rc._generate_lfsr_v1_sequence(seed, sequence_index + 1, 1)[0]
        return seed, rc.PauliLiteral((seed >> (2 * pauli_index)) & 0b11)

_SIMPLE_TEST_CYCLE = ((0, 1),)
_ALTERNATING_BASE_CYCLES = (((0, 1), (2, 3), (4, 5)), (0, (1, 2), (3, 4), 5))
_NON_TWIRLED_QUBITS = (((0, 1), 2), (0, (1, 2)), (0, 2))


_PAULI_FRAME_TRACKING_TEST_CASES = [
    # Most basic test case.
    rc.RandomizedCompilingConfiguration(
        base_cycles=(_SIMPLE_TEST_CYCLE,),
        base_cycle_repetitions=1,
    ),
    # Note that qubits {0, 5} idle in the second 2Q cycle. Additionally, note that the 13 base cycle reptitions require
    # more than 48 bits, so we test the Pauli frame tracking over multiple seed values.
    rc.RandomizedCompilingConfiguration(
        base_cycles=_ALTERNATING_BASE_CYCLES,
        base_cycle_repetitions=13,
    ),
    # Same as previous case but without Pauli inversion.
    rc.RandomizedCompilingConfiguration(
        base_cycles=_ALTERNATING_BASE_CYCLES,
        base_cycle_repetitions=13,
        invert_random_paulis=False,
    ),
    # Includes non-twirled qubits.
    rc.RandomizedCompilingConfiguration(
        base_cycles=_NON_TWIRLED_QUBITS,
        base_cycle_repetitions=13,
    ),
    # Custom twirled unitaries variable.
    rc.RandomizedCompilingConfiguration(
        base_cycles=_ALTERNATING_BASE_CYCLES,
        base_cycle_repetitions=13,
        variables=rc.RandomizedCompilingVariables(
            twirled_unitaries_prefix="twirled_unitaries"
        )
    ),
]

@pytest.mark.parametrize("configuration", _PAULI_FRAME_TRACKING_TEST_CASES)
def test_pauli_frame_tracking(configuration: rc.RandomizedCompilingConfiguration):
    """Test that `rc.RandomizedCompilingConfiguration.track_pauli_frames` produces the expected tracked Paulis for a given configuration and random seed.

    This test checks the following:

    * Randomly generated Paulis match expectations for the given random seed (see `_get_expected_pauli_pair`).
    * If `configuration.invert_random_paulis` is `True`, then the tracked Paulis conjugate through the 2Q gates and 1Q identities as expected.
    * If `configuration.invert_random_paulis` is `False`, then all previous Paulis are the identity.

    Conjugation assertions assume that the `rc.PAULI_CONJUGATES_MAPS` are correct.
    """
    rng = np.random.default_rng(seed=156_548_857)
    random_seeds = configuration.generate_seed_values(rng)
    pauli_conjugates_map = rc.PAULI_CONJUGATES_MAPS["CZ"]
    all_pauli_pairs = list(configuration.track_pauli_frames(10, pauli_conjugates_map, random_seeds))
    assert len(all_pauli_pairs) == 10
    all_cycles = configuration._base_twirled_cycles * configuration.base_cycle_repetitions
    assert all(
        len(pauli_pairs) == len(configuration.qubits_sorted) * (len(all_cycles) + 1) for pauli_pairs in all_pauli_pairs
    )
    seeds = {
        q: random_seeds[qubit_index].tolist()
        for qubit_index, q in enumerate(configuration.qubits_sorted)
    }
    for sequence_index, pauli_pairs in enumerate(all_pauli_pairs):
        # Check that the randomly generated Paulis match expectations.
        for qubit in configuration.qubits_sorted:
            for layer_index in range(len(all_cycles) + 1):
                key = rc.PauliPairKey(qubit=qubit, layer_index=layer_index)
                base_cycle = all_cycles[layer_index] if layer_index < len(all_cycles) else None
                seed, pauli_pair = pauli_pairs[key]
                if base_cycle is None or qubit not in base_cycle:
                    expected_next_pauli = rc.PauliLiteral.I
                else:
                    expected_seed, expected_next_pauli = _get_expected_random_pauli(
                        seeds[qubit], layer_index, len(all_cycles) + 1, sequence_index, configuration._paulis_per_value, configuration.variables.twirled_overwrites_source_unitaries
                    )
                    assert (
                        seed == expected_seed
                    ), f"Seed mismatch for qubit {qubit} at layer {layer_index}: expected {expected_seed}, got {seed}"
                assert (
                    pauli_pair[1] == expected_next_pauli
                ), f"Pauli mismatch for qubit {qubit} at layer {layer_index}: expected {expected_next_pauli}, got {pauli_pair[1]}"

        if configuration.invert_random_paulis:
            # Check that the Paulis conjugate as expected.
            for layer_index, cycle in enumerate(all_cycles):
                # Check conjugation through 2Q gates.
                for edge in cycle.two_qubit_gates.values():
                    before_pauli_left = pauli_pairs[rc.PauliPairKey(qubit=edge[0], layer_index=layer_index)][1][1]
                    before_pauli_right = pauli_pairs[rc.PauliPairKey(qubit=edge[1], layer_index=layer_index)][1][1]
                    before_paulis = np.kron(before_pauli_left.matrix, before_pauli_right.matrix)
                    after_pauli_left = pauli_pairs[rc.PauliPairKey(qubit=edge[0], layer_index=layer_index + 1)][1][0]
                    after_pauli_right = pauli_pairs[rc.PauliPairKey(qubit=edge[1], layer_index=layer_index + 1)][1][0]
                    after_paulis = np.kron(after_pauli_left.matrix, after_pauli_right.matrix)
                    result = after_paulis @ matrices.CZ @ before_paulis
                    assert rc._unitary_equal(
                        result, matrices.CZ
                    ), f"Failed at sequence {sequence_index} for cycle {cycle} at layer {layer_index}: found {result}"

                # Check conjugation over 1Q identity.
                for qubit in cycle.idle_qubits:
                    before_pauli = pauli_pairs[rc.PauliPairKey(qubit=qubit, layer_index=layer_index)][1][1]
                    after_pauli = pauli_pairs[rc.PauliPairKey(qubit=qubit, layer_index=layer_index + 1)][1][0]
                    result = after_pauli.matrix @ before_pauli.matrix
                    assert rc._unitary_equal(
                        result, matrices.I
                    ), f"Failed at sequence {sequence_index} for non-cycle qubit {qubit} at layer {layer_index}: found {result}"

                # Check that Paulis not configured in the cycle are untwirled.
                for qubit in configuration.qubits_sorted:
                    if qubit not in cycle:
                        before_pauli = pauli_pairs[rc.PauliPairKey(qubit=qubit, layer_index=layer_index)][1][1]
                        assert before_pauli == rc.PauliLiteral.I, f"Expected identity for previous Pauli but found {before_pauli} for qubit {qubit} at layer {layer_index}"
                        after_pauli = pauli_pairs[rc.PauliPairKey(qubit=qubit, layer_index=layer_index + 1)][1][0]
                        assert after_pauli == rc.PauliLiteral.I, f"Expected identity for next Pauli but found {after_pauli} for qubit {qubit} at layer {layer_index + 1}"
        else:
            # If Paulis are not inverted then we can simply asssert that all previous Paulis are the identity.
            for qubit in configuration.qubits_sorted:
                for layer_index in range(len(all_cycles) + 1):
                    key = rc.PauliPairKey(qubit=qubit, layer_index=layer_index)
                    _, pauli_pair = pauli_pairs[key]
                    assert (
                        pauli_pair[0] == rc.PauliLiteral.I
                    ), f"Expected identity for previous Pauli but found {pauli_pair[0]} for qubit {qubit} at layer {layer_index}"


@pytest.mark.parametrize("configuration", [configuration for configuration in _PAULI_FRAME_TRACKING_TEST_CASES if configuration.variables.twirled_overwrites_source_unitaries])
def test_pauli_cache_accumulation(configuration: rc.RandomizedCompilingConfiguration):
    """Test `rc._PauliSeedAndPairCache.accumulate`.

    The result should match the last element produced by `configuration.track_pauli_frames` after the same number of steps,
    assuming that the configuration overwrites source unitaries with twirled unitaries. This test assumes the following
    features are correctly implemented:

    * Pauli multiplication (see `test_pauli_literal_multiplication` above).
    * Pauli frame tracking (see `test_pauli_frame_tracking` above).
    * `rc.PAULI_CONJUGATES_MAPS` (see `test_pauli_conjugates_map` above).
    """
    assert configuration.variables.twirled_overwrites_source_unitaries
    rng = np.random.default_rng(seed=685_522_415)
    random_seeds = configuration.generate_seed_values(rng)
    pauli_conjugates_map = rc.PAULI_CONJUGATES_MAPS["CZ"]

    pauli_cache = rc._PauliSeedAndPairCache(
        original_seeds={
            q: random_seeds[qubit_index].tolist()
            for qubit_index, q in enumerate(configuration.qubits_sorted)
        },
        pauli_conjugates_map=pauli_conjugates_map,
        cycles=configuration._base_twirled_cycles * configuration.base_cycle_repetitions,
        qubits_sorted=configuration.qubits_sorted,
        prng_sequence_steps=1,
        invert_random_paulis=configuration.invert_random_paulis,
        paulis_per_value=configuration._paulis_per_value,
    )

    accumulation_steps = 10
    tracked_paulis = list(configuration.track_pauli_frames(accumulation_steps, pauli_conjugates_map, random_seeds))[-1]
    accumulated_paulis = pauli_cache.accumulate(accumulation_steps)
    all_cycles = configuration.base_cycles * configuration.base_cycle_repetitions
    assert len(accumulated_paulis) == len(configuration.qubits_sorted) * (len(all_cycles) + 1)
    for qubit in configuration.qubits_sorted:
        for layer_index in range(len(all_cycles) + 1):
            key = rc.PauliPairKey(qubit=qubit, layer_index=layer_index)
            seed, pauli_pair = accumulated_paulis[key]
            expected_seed, expected_pauli_pair = tracked_paulis[key]
            assert (
                seed == expected_seed
            ), f"Seed mismatch for qubit {qubit} at layer {layer_index}: expected {expected_seed}, got {seed}"
            assert (
                pauli_pair == expected_pauli_pair
            ), f"Pauli pair mismatch for qubit {qubit} at layer {layer_index}: expected {expected_pauli_pair}, got {pauli_pair}"


_FIXTURE_DIRECTORY = Path(__file__).parent / "__fixtures__"
_FIXTURE_DIRECTORY.mkdir(exist_ok=True)
_TETRAHEDRAL_ANGLES = np.array([[ 0.0,  0.5,  0.5],
       [-1./4,  0.0, -1./4],
       [ 0.0,  0.0,  0.5],
       [ 1./4,  0.5, -1./4],
       [ 0.0,  1./4, -1./4],
       [ 0.5,  1./4, -1./4],
       [ 0.5,  1./4,  1./4],
       [ 0.0,  1./4,  1./4],
       [ -1./4,  1./4,  0.0],
       [ 1./4,  1./4,  0.5],
       [-1./4,  1./4,  0.5],
       [ 1./4,  1./4,  0.0]], dtype=np.float64)

@dataclass(frozen=True)
class ReadoutRandomization:
    qubits_sorted: tuple[int, ...]
    readout_source_angles: NDArray[np.float64] = field(default_factory=lambda: _TETRAHEDRAL_ANGLES)
    """Shape is (unitary_count, 3)."""

    def build_quil_program(self) -> Program:
        program = Program()
        program += Declare("readout_source_angles", "REAL", len(self.readout_source_angles) * rc._ANGLES_PER_UNITARY)
        for qubit in self.qubits_sorted:
            program += Declare(f"readout_seed_q{qubit}", "INTEGER", 1)
            program += Declare(f"readout_randomization_q{qubit}", "REAL", rc._ANGLES_PER_UNITARY)

        for qubit in self.qubits_sorted:
            program += Call(
                "choose_random_real_sub_regions",
                [
                    inst.CallArgument.from_identifier(f"readout_randomization_q{qubit}"),
                    inst.CallArgument.from_identifier("readout_source_angles"),
                    inst.CallArgument.from_immediate(complex(rc._ANGLES_PER_UNITARY)),
                    inst.CallArgument.from_memory_reference(inst.MemoryReference(f"readout_seed_q{qubit}", 0)),
                ]
            )
        return program

    def generate_seeds(self, rng: np.random.Generator) -> dict[int, int]:
        return {qubit: rng.integers(0, rc._MAX_SEQUENCER_VALUE + 1, dtype=np.int64).item() for qubit in self.qubits_sorted}

    def build_memory_map(self, seeds: Mapping[int, int]) -> dict[str, list[float]]:
        memory_map = {}
        memory_map["readout_source_angles"] = self.readout_source_angles.flatten().tolist()
        for qubit in self.qubits_sorted:
            memory_map[f"readout_seed_q{qubit}"] = [seeds[qubit]]
            memory_map[f"readout_randomization_q{qubit}"] = [0.0] * rc._ANGLES_PER_UNITARY
        return memory_map

    def verify_final_memory(
        self,
        final_memory: dict[str, Union[list[int], list[float]]],
        seeds: Mapping[int, int],
        shot_count: int,
        final_layer_index: int,
        pauli_pairs: Mapping[rc.PauliPairKey, tuple[Optional[int], tuple[rc.PauliLiteral, rc.PauliLiteral]]]
    ) -> None:
        for qubit in self.qubits_sorted:
            final_random_unitary_index = choose_random_real_sub_region_indices(
                PrngSeedValue(seeds[qubit]), shot_count - 1, 1, len(self.readout_source_angles)
            )[0]
            final_random_unitary_angles = tuple(self.readout_source_angles[final_random_unitary_index].tolist())
            final_random_unitary = rc._compute_unitary_from_zxzxz_angles(
                final_random_unitary_angles
            )
            _, pauli_pair = pauli_pairs[rc.PauliPairKey(qubit=qubit, layer_index=final_layer_index)]
            assert pauli_pair[1] == rc.PauliLiteral.I, f"Expected identity for final Pauli but found {pauli_pair[1]} for qubit {qubit} at layer {final_layer_index}"
            expected_unitary = final_random_unitary @ pauli_pair[0].matrix
            found_unitary_angles = tuple(final_memory[f"readout_randomization_q{qubit}"])
            found_unitary = rc._compute_unitary_from_zxzxz_angles(found_unitary_angles)
            assert rc._unitary_equal(
                found_unitary, expected_unitary
            ), f"Final unitary mismatch for qubit {qubit}: expected {final_random_unitary_angles} @ {pauli_pair}, found {found_unitary_angles}"


@dataclass(frozen=True)
class ConfigurationTestCase:
    configuration: rc.RandomizedCompilingConfiguration
    seed_loop_length: int = 0
    seed_loop_inner_length: int = 0
    base_cycle_loop_length: int = 0
    readout_randomization: Optional[ReadoutRandomization] = None

    def build_quil_program(self) -> Program:
        program = Program()
        program += self.configuration.build_quil_program()
        if self.readout_randomization is not None:
            program += self.readout_randomization.build_quil_program()
            for qubit in self.readout_randomization.qubits_sorted:
                call = self.configuration.apply_pauli_pair(
                    qubit,
                    self.configuration._cycle_count,
                    source_unitaries=f"readout_randomization_q{qubit}",
                    target_unitaries=f"readout_randomization_q{qubit}",
                    unitary_offset=0
                )
                if call is not None:
                    program += call
        return program

    def generate_seeds_and_memory_map(self, rng: np.random.Generator) -> tuple[dict[str, Union[list[int], list[float]]], NDArray[np.int64], Optional[dict[int, int]]]:
        memory_map: dict[str, Union[list[int], list[float]]] = {}
        rc_seeds = self.configuration.generate_seed_values(rng)
        memory_map.update(self.configuration.build_memory_map(rc_seeds, rc.build_memory_values_for_paulis_conjugates_map(rc.PAULI_CONJUGATES_MAPS["CZ"])))
        if self.readout_randomization is not None:
            readout_seeds = self.readout_randomization.generate_seeds(rng)
            memory_map.update(self.readout_randomization.build_memory_map(readout_seeds))
        else:
            readout_seeds = None
        memory_map.update(generate_source_unitaries(self.configuration, rng))
        return memory_map, rc_seeds, readout_seeds


CONFIGURATION_TEST_SEED = 156_548_857
TEST_SHOT_COUNT = 2_500


def _sx(qubit: int) -> gates.Gate:
    return gates.RX(np.pi / 2, qubit)


def _zxzxz(configuration: rc.RandomizedCompilingConfiguration, layer_index: int, readout_randomization: Optional[ReadoutRandomization] = None) -> Program:
    program = Program()
    for qubit in configuration.qubits_sorted:
        if readout_randomization is None:
            ref = configuration.variables.twirled_unitaries_ref(qubit, layer_index, 0)
        else:
            ref = MemoryReference(f"readout_randomization_q{qubit}", 0)
        program += gates.RZ(2 * np.pi * ref, qubit)
        program += _sx(qubit)
        program += gates.FENCE(qubit)

        if readout_randomization is None:
            ref = configuration.variables.twirled_unitaries_ref(qubit, layer_index, 1)
        else:
            ref = MemoryReference(f"readout_randomization_q{qubit}", 1)
        program += gates.RZ(2 * np.pi * ref, qubit)
        program += _sx(qubit)
        program += gates.FENCE(qubit)

        if readout_randomization is None:
            ref = configuration.variables.twirled_unitaries_ref(qubit, layer_index, 2)
        else:
            ref = MemoryReference(f"readout_randomization_q{qubit}", 2)
        program += gates.RZ(2 * np.pi * ref, qubit)
    return program


def build_cycle_program(configuration: rc.RandomizedCompilingConfiguration, readout_randomization: Optional[ReadoutRandomization]) -> Program:
    program = Program()
    cycle_count = configuration.base_cycle_repetitions * len(configuration.base_cycles) + 1
    if not configuration.variables.twirled_overwrites_source_unitaries:
        for qubit in configuration.qubits_sorted:
            memory_region_name = configuration.variables.source_unitaries(qubit)
            program += Declare(memory_region_name, "REAL", cycle_count * rc._ANGLES_PER_UNITARY)
    for rep_index in range(configuration.base_cycle_repetitions):
        for base_index, cycle in enumerate(configuration._base_twirled_cycles):
            layer_index = rep_index * len(configuration.base_cycles) + base_index
            program += _zxzxz(configuration, layer_index)
            for edge in cycle.two_qubit_gates.values():
                program += gates.CZ(edge[0], edge[1])
                program += gates.FENCE(edge[0], edge[1])

    program += _zxzxz(configuration, configuration.base_cycle_repetitions * len(configuration.base_cycles), readout_randomization)

    return program


CONFIGURATION_TEST_CASES: list[ConfigurationTestCase] = [
    # 0) simple base case; no loops required (single base cycle)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,),
            base_cycle_repetitions=1,
        ),
    ),
    # 1) base cycle loop only (single base cycle)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,),
            base_cycle_repetitions=2,
        ),
        base_cycle_loop_length=1
    ),
    # 2) base cycle loop with maximum iterations (single base cycle)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,),
            base_cycle_repetitions=24,
        ),
        base_cycle_loop_length=23
    ),
    # 3) seed loop required (single base cycle)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,),
            base_cycle_repetitions=25,
        ),
        seed_loop_length=1,
        seed_loop_inner_length=23,
    ),
    # 4) base cycle loop only (two base cycles)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,) * 2,
            base_cycle_repetitions=12,
        ),
        base_cycle_loop_length=11
    ),
    # 5) seed loop required + base cycle loop (two base cycles)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,) * 2,
            base_cycle_repetitions=14,
        ),
        seed_loop_length=1,
        seed_loop_inner_length=11,
        base_cycle_loop_length=1
    ),
    # 6) base cycle loop only (four base cycles)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,) * 4,
            base_cycle_repetitions=5,
        ),
        base_cycle_loop_length=4
    ),
    # 7) seed loop required + base cycle loop (four base cycles)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,) * 4,
            base_cycle_repetitions=7,
        ),
        seed_loop_length=1,
        seed_loop_inner_length=5,
    ),
    # 8) base cycle length >= max Paulis per value (i.e. seed transition within base cycle)
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=(_SIMPLE_TEST_CYCLE,) * rc._MAX_PAULIS_PER_VALUE,
            base_cycle_repetitions=2,
        ),
        base_cycle_loop_length=1,
    ),
    # 9) 4 looped base cycles + final base cycle.
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=5,
        ),
        base_cycle_loop_length=4
    ),
    # 10) 2 seed loop iterations + final base cycle.
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=25,
        ),
        seed_loop_length=2,
        seed_loop_inner_length=11,
    ),
    # 11) 2 seed loop iterations + 2 base cycle iterations + final base cycle.
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=27,
        ),
        seed_loop_length=2,
        seed_loop_inner_length=11,
        base_cycle_loop_length=2,
    ),
    # 12)seed loop with shots per randomization.
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=13,
            shots_per_randomization=rc.ShotsPerRandomization(
                shots_per_randomization=50,
            )
        ),
        seed_loop_length=1,
        seed_loop_inner_length=11,
    ),
    # 13) seed loop with readout randomization
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=13,
            skip_final_layer=True,
        ),
        seed_loop_length=1,
        seed_loop_inner_length=11,
        readout_randomization=ReadoutRandomization(
            qubits_sorted=tuple(range(6)),
        )
    ),
    # 14) seed loop with shots per randomization and readout randomization
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=13,
            shots_per_randomization=rc.ShotsPerRandomization(
                shots_per_randomization=50,
            ),
            skip_final_layer=True
        ),
        seed_loop_length=1,
        seed_loop_inner_length=11,
        readout_randomization=ReadoutRandomization(
            qubits_sorted=tuple(range(6)),
        )
    ),
    # 15) seed loop on cycles with untwirled qubits.
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_NON_TWIRLED_QUBITS,
            base_cycle_repetitions=13,
        ),
        seed_loop_length=1,
        seed_loop_inner_length=7,
        base_cycle_loop_length=4,
    ),
    # 16) 4 looped base cycles;
    ConfigurationTestCase(
        configuration=rc.RandomizedCompilingConfiguration(
            base_cycles=_ALTERNATING_BASE_CYCLES,
            base_cycle_repetitions=5,
            variables=rc.RandomizedCompilingVariables(
                twirled_unitaries_prefix="twirled_unitaries"
            )
        ),
        base_cycle_loop_length=4
    ),
]


def generate_source_unitaries(configuration: rc.RandomizedCompilingConfiguration, rng: np.random.Generator) -> dict[str, list[float]]:
    source_unitaries = {}
    cycle_count = configuration.base_cycle_repetitions * len(configuration.base_cycles) + 1
    for qubit in configuration.qubits_sorted:
        source_unitaries[configuration.variables.source_unitaries(qubit)] = rng.uniform(-0.5, 0.5, size=rc._ANGLES_PER_UNITARY * cycle_count).tolist()
    return source_unitaries


@pytest.mark.parametrize(
    "test_case",
    CONFIGURATION_TEST_CASES,
    ids=[f"configuration{i}" for i in range(len(CONFIGURATION_TEST_CASES))]
)
def test_randomized_compiling_configuration(
    test_case: ConfigurationTestCase,
    snapshot: SnapshotAssertion,
    request: pytest.FixtureRequest,
):
    """Test that the provided configuration for loop parameters, program structure, and final memory validation.

    The final memory fixtures are produced from real programs on the QPU. Should these require update, you can
    run `test/e2e/test_qpu_randomized_compiling.py` to read the final memory and then manually update the files.
    """
    assert test_case.configuration._seed_loop_length == test_case.seed_loop_length
    assert test_case.configuration._seed_loop_inner_length == test_case.seed_loop_inner_length
    assert test_case.configuration._base_cycle_loop_length == test_case.base_cycle_loop_length
    expected_total_u2_cycles = test_case.configuration.base_cycle_repetitions * len(test_case.configuration.base_cycles) + 1
    looped_base_cycles = test_case.seed_loop_length * (test_case.seed_loop_inner_length + 1) + test_case.base_cycle_loop_length
    # after the seed and base loop cycles, we complete a final base cycle.
    completed_base_cycles = looped_base_cycles + 1
    # we add one for the initial cycle (i.e. where there were no previous random Paulis to invert).
    completed_u2_cycles = completed_base_cycles * len(test_case.configuration.base_cycles) + 1
    assert expected_total_u2_cycles == completed_u2_cycles

    program = test_case.build_quil_program()
    assert program.out() == snapshot(name="quil")
    program += build_cycle_program(test_case.configuration, test_case.readout_randomization)

    rng = np.random.default_rng(seed=CONFIGURATION_TEST_SEED)
    memory_map, rc_seeds,readout_seeds = test_case.generate_seeds_and_memory_map(rng)

    with open(_FIXTURE_DIRECTORY / f"{request.node.name}.json") as f:
        final_memory = json.load(f)

    pauli_conjugates_map = rc.PAULI_CONJUGATES_MAPS["CZ"]
    test_case.configuration.verify_final_memory(
        final_memory,
        memory_map,
        TEST_SHOT_COUNT,
        pauli_conjugates_map,
    )

    if test_case.readout_randomization is not None:
        if readout_seeds is None:
            raise ValueError("Readout seeds should not be None when readout randomization is provided.")
        pauli_pairs = test_case.configuration.get_final_pauli_pairs(TEST_SHOT_COUNT, pauli_conjugates_map, rc_seeds, accumulate=False)
        test_case.readout_randomization.verify_final_memory(
            final_memory,
            readout_seeds,
            TEST_SHOT_COUNT,
            test_case.configuration._cycle_count,
            pauli_pairs,
        )
