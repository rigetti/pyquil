"""A utility for building programs and memory maps for randomized compiling on the QPU.

The utilities here are built specifically for random compilation with the
ZXZXZ unitary decomposition using the "merge_zxzxz_unitary_with_paulis" suite
of extern functions.

The main entrypoint is the `RandomizedCompilingConfiguration` dataclass which can:

* build the classical instructions to randomly compile 2Q gate cycles by modifying
    the phase angles of the ZXZXZ decomposition (see
    `RandomizedCompilingConfiguration.build_quil_program`).
* generate random seeds for drawing random Paulis on the QPU (see
    `RandomizedCompilingConfiguration.generate_random_seeds`).
* build a memory map for QPU execution (see `RandomizedCompilingConfiguration.build_memory_map`).
* track Pauli frames on a per shot basis (see `RandomizedCompilingConfiguration.track_pauli_frames`).
* verify the final memory after execution to check that the correct Pauli frames were applied (see
    `RandomizedCompilingConfiguration.verify_final_memory`).

Note, these utilities do not build the cycle program itself nor the source unitaries for that cycle
program.
"""

from contextlib import contextmanager
from pyquil.gates import FENCE
import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from quil import instructions as inst

from pyquil.quil import InstructionDesignator, Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import (
    Call,
    ClassicalAdd,
    ClassicalGreaterEqual,
    ClassicalLoad,
    ClassicalMove,
    ClassicalShiftRight,
    Declare,
    Delay,
    Expression,
    Fence,
    Jump,
    JumpTarget,
    JumpUnless,
    JumpWhen,
    Label,
    MemoryReference,
)
from pyquil.simulation import matrices

from ._classical_computations import build_extern_function_signatures, delay_and_fence_classical_preamble

_BITS_PER_VALUE = 48
_BITS_PER_PAULI = 2
_MAX_PAULIS_PER_VALUE = _BITS_PER_VALUE // _BITS_PER_PAULI
_ANGLES_PER_UNITARY = 3
_NUMBER_PAULI_PAIRS = 16


_TEdge = tuple[int, int]


@dataclass(frozen=True)
class _TwirledCycle:
    """A representation of all two and single qubit gates that require twirling within a given cycle.

    Single qubit gates are presumed to be the identity and, therefore, will be sandwiched by the same
    Pauli. Any qubit not in neither `two_qubit_gates` nor `idle_qubits` may otherwise be assumed
    untwirled.

    Any given qubit may be present only either in `two_qubit_gates` or `idle_qubits`.
    """

    two_qubit_gates: Mapping[int, _TEdge]
    """Two qubit gate which must be twirled in the cycle."""

    idle_qubits: frozenset[int]
    """Qubits on which the single qubit identity gate is played during the cycle."""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for qubit in self.all_qubits:
            if qubit in self.two_qubit_gates and qubit in self.idle_qubits:
                raise ValueError(f"qubit {qubit} is configured as both an edge and idle qubit")

    def __getitem__(self, node: int) -> Union[_TEdge, int]:
        if node in self.idle_qubits:
            return node
        elif node in self.two_qubit_gates:
            return self.two_qubit_gates[node]
        else:
            raise KeyError(f"node {node} not found in cycle")

    def __contains__(self, node: int) -> bool:
        return node in self.two_qubit_gates or node in self.idle_qubits

    @cached_property
    def all_qubits(self) -> set[int]:
        return set(self.two_qubit_gates.keys()) | self.idle_qubits

    @classmethod
    def from_base_cycle(cls, cycle: Sequence[Union[_TEdge, int]]) -> "_TwirledCycle":
        two_qubit_gates = dict()
        idle_qubits = set()
        for edge_or_idle in cycle:
            if isinstance(edge_or_idle, tuple):
                if any(node in two_qubit_gates or node in idle_qubits for node in edge_or_idle):
                    raise ValueError(f"edge {edge_or_idle} overlaps with existing edges or idles in cycle")
                two_qubit_gates[edge_or_idle[0]] = edge_or_idle
                two_qubit_gates[edge_or_idle[1]] = edge_or_idle
            else:
                if edge_or_idle in two_qubit_gates or edge_or_idle in idle_qubits:
                    raise ValueError(f"idle qubit {edge_or_idle} overlaps with existing edges or idles in cycle")
                idle_qubits.add(edge_or_idle)
        return cls(two_qubit_gates=two_qubit_gates, idle_qubits=frozenset(idle_qubits))


def _generate_lfsr_v1_sequence(seed_value: int, start_index: int, count: int) -> list[int]:
    """Generate a sequence of values from the LFSR v1 PRNG given an initial seed value, a start index, and a count."""
    sequence = []
    current_value = seed_value
    for i in range(start_index + count):
        if i >= start_index:
            sequence.append(current_value)
        current_value = _lfsr_v1_next(current_value)
    return sequence


@dataclass(frozen=True)
class PauliPairKey:
    """A key for looking up the random Pauli pair for a specific qubit and layer index.

    This key is used within the context of a single shot of the cycle program. See
    `RandomizedCompilingConfiguration.track_pauli_frames` for more details.
    """

    qubit: int
    layer_index: int


def _accumulate_pauli_pairs(
    existing_pairs: dict[PauliPairKey, tuple[Optional[int], tuple["PauliLiteral", "PauliLiteral"]]],
    new_pairs: dict[PauliPairKey, tuple[Optional[int], tuple["PauliLiteral", "PauliLiteral"]]],
) -> dict[PauliPairKey, tuple[Optional[int], tuple["PauliLiteral", "PauliLiteral"]]]:
    """Accumulate new Pauli pairs into the existing accumulator.

    This is used to track the final Pauli pairs for each qubit and layer across multiple sequences.
    """
    accumulator = {}
    for key, (seed, new_pair) in new_pairs.items():
        if key not in existing_pairs:
            raise ValueError(f"Key {key} not found in accumulator.")
        _, existing_pair = existing_pairs[key]
        _, accumulated_next = existing_pair[1] * new_pair[1]
        _, accumulated_previous = new_pair[0] * existing_pair[0]
        accumulator[key] = (seed, (accumulated_previous, accumulated_next))

    return accumulator


@dataclass(frozen=True)
class _PauliSeedAndPairCache:
    """Cache for final Pauli seeds and pairs per qubit and layer.

    Each layer beyond the first of the circuit requires knowledge of the previous
    layer's Pauli pair in order to determine the conjugate.
    """

    prng_sequence_steps: int
    original_seeds: Mapping[int, Sequence[int]]
    pauli_conjugates_map: Mapping[tuple["PauliLiteral", "PauliLiteral"], tuple["PauliLiteral", "PauliLiteral"]]
    cycles: tuple[_TwirledCycle, ...]
    qubits_sorted: tuple[int, ...]
    invert_random_paulis: bool
    pauli_pairs: dict[PauliPairKey, tuple["PauliLiteral", "PauliLiteral"]] = field(default_factory=dict, init=False)
    paulis_per_value: int

    def accumulate(
        self,
        sequence_count: int,
    ) -> "dict[PauliPairKey, tuple[Optional[int], tuple[PauliLiteral, PauliLiteral]]]":
        """Iterate over the requested `sequence_count` and accumulate the final Pauli pair for each qubit and layer index.

        "Accumulation" in this context means applying random Pauli pair successively over the sequence count. This
        is useful in the context where the twirled unitary angles are overwritten each shot and, therefore, the Paulis
        accumulate over the shot sequence rather than act independently.
        """
        current = self
        accumulated_pauli_pairs = None
        for sequence_index in range(sequence_count):
            pauli_pairs = {}
            for qubit in self.qubits_sorted:
                for layer_index in range(len(self.cycles) + 1):
                    key = PauliPairKey(qubit=qubit, layer_index=layer_index)
                    pauli_pairs[key] = current[key]
            if accumulated_pauli_pairs is None:
                accumulated_pauli_pairs = pauli_pairs
            else:
                accumulated_pauli_pairs = _accumulate_pauli_pairs(accumulated_pauli_pairs, pauli_pairs)
            if sequence_index < sequence_count - 1:
                current = next(current)
        if accumulated_pauli_pairs is None:
            raise ValueError("must specify sequence_count > 0 to accumulate Pauli pairs")
        return accumulated_pauli_pairs

    def __next__(self) -> "_PauliSeedAndPairCache":
        """Return the "next" cache in the sequence.

        This returns a fresh cache with the original seeds set to the final seed values from this cache.
        The "next" cache thus implies that the sequence advances by "prng_sequence_steps".
        """
        return _PauliSeedAndPairCache(
            prng_sequence_steps=self.prng_sequence_steps,
            original_seeds=self._final_seeds,
            pauli_conjugates_map=self.pauli_conjugates_map,
            cycles=self.cycles,
            qubits_sorted=self.qubits_sorted,
            invert_random_paulis=self.invert_random_paulis,
            paulis_per_value=self.paulis_per_value,
        )

    @cached_property
    def _final_seeds(self) -> dict[int, tuple[int, ...]]:
        final_seeds = {}
        for qubit in self.qubits_sorted:
            seeds = self.original_seeds[qubit]
            final_seeds[qubit] = tuple(
                _generate_lfsr_v1_sequence(seed, start_index=self.prng_sequence_steps, count=1)[0] for seed in seeds
            )
        return final_seeds

    def _get_next_pauli(self, seed_value: int, key: PauliPairKey) -> "PauliLiteral":
        base_cycle = self.cycles[key.layer_index] if key.layer_index < len(self.cycles) else None
        if base_cycle is not None and key.qubit in base_cycle:
            pauli_index = key.layer_index % self.paulis_per_value
            return PauliLiteral(seed_value >> (2 * pauli_index) & 0b11)
        else:
            return PauliLiteral.I

    def _get_previous_random_pauli(self, key: PauliPairKey) -> "PauliLiteral":
        if not self.invert_random_paulis or key.layer_index == 0:
            return PauliLiteral.I
        previous_layer_index = key.layer_index - 1
        previous_cycle = self.cycles[previous_layer_index]
        if key.qubit in previous_cycle.two_qubit_gates:
            previous_edge = previous_cycle.two_qubit_gates[key.qubit]
            previous_left_key = PauliPairKey(qubit=previous_edge[0], layer_index=previous_layer_index)
            previous_right_key = PauliPairKey(qubit=previous_edge[1], layer_index=previous_layer_index)
            _, previous_pauli_pair_left = self[previous_left_key]
            _, previous_pauli_pair_right = self[previous_right_key]
            conjugate = self.pauli_conjugates_map[(previous_pauli_pair_left[1], previous_pauli_pair_right[1])]
            is_pauli_left = key.qubit == previous_edge[0]
            previous_conjugate = conjugate[0] if is_pauli_left else conjugate[1]
        elif key.qubit in previous_cycle.idle_qubits:
            previous_key = PauliPairKey(qubit=key.qubit, layer_index=previous_layer_index)
            _, previous_pauli_pair = self[previous_key]
            previous_conjugate = previous_pauli_pair[1]
        else:
            previous_conjugate = PauliLiteral.I

        return previous_conjugate

    def __getitem__(self, key: PauliPairKey) -> "tuple[Optional[int], tuple[PauliLiteral, PauliLiteral]]":
        q, layer_index = key.qubit, key.layer_index
        if layer_index == len(self.cycles):
            # there is no random Pauli to apply!
            if key not in self.pauli_pairs:
                next_pauli = PauliLiteral.I
                previous_conjugate = self._get_previous_random_pauli(key)
                self.pauli_pairs[key] = (previous_conjugate, next_pauli)
            return None, self.pauli_pairs[key]
        seed_index = layer_index // self.paulis_per_value
        seed_value = self._final_seeds[q][seed_index]
        if key not in self.pauli_pairs:
            next_pauli = self._get_next_pauli(seed_value, key)
            previous_conjugate = self._get_previous_random_pauli(key)
            self.pauli_pairs[key] = (previous_conjugate, next_pauli)
        pauli_pair = self.pauli_pairs[key]
        return seed_value, pauli_pair


def _radians_to_cycles(region_name: str, index: int) -> Expression:
    return MemoryReference(region_name, index) * 2 * math.pi


_MAX_SEQUENCER_VALUE = (1 << _BITS_PER_VALUE) - 1
_TAPS = (47, 46, 20, 19)


def _lfsr_v1_next(seed: int) -> int:
    """Return the next value in the PRNG sequence available on the QPU.

    This implementation is necessary as `qcs_sdk.qpu.experimental.random` does not currently have a way to get
    expose the inner value of the `PrngSeedValue` to Python; we can drop this in favor of the QCS SDK version
    once pyQuil updates to a version of the QCS SDK that has this capability.
    """
    feedback_value = 0
    for tap in _TAPS:
        base = 1 << tap
        bit = int((seed & base) != 0)
        feedback_value ^= bit
    return ((seed << 1) & _MAX_SEQUENCER_VALUE) | feedback_value


def _pauli_conjugates_map_str_to_literals(
    pauli_conjugates_map_str: Mapping[str, str],
) -> dict[tuple["PauliLiteral", "PauliLiteral"], tuple["PauliLiteral", "PauliLiteral"]]:
    pauli_conjugates_map = {}
    for previous_str, next_str in pauli_conjugates_map_str.items():
        previous_paulis = tuple(PauliLiteral.from_name(c) for c in previous_str)
        next_paulis = tuple(PauliLiteral.from_name(c) for c in next_str)
        if len(previous_paulis) != 2 or len(next_paulis) != 2:
            raise ValueError(f"invalid pauli pair strings: {previous_str}, {next_str}")
        pauli_conjugates_map[previous_paulis] = next_paulis
    return pauli_conjugates_map


def _pauli_pair_to_int(pauli_pair: tuple["PauliLiteral", "PauliLiteral"]) -> int:
    return (pauli_pair[0].value << _BITS_PER_PAULI) + pauli_pair[1].value


def _unitary_equal(A: NDArray[np.complex128], B: NDArray[np.complex128]) -> bool:
    """Check if two matrices are unitarily equal."""
    if A.shape != B.shape:
        return False
    dim = A.shape[0]
    return cast(bool, np.isclose(np.abs(np.trace(A.T.conjugate() @ B) / dim), 1.0))


@dataclass(frozen=True)
class ShotsPerRandomizationVariables:
    pulse_program_label: str = "pulse_program"
    randomization_label: str = "rc_main"
    modulo_counter: str = "modulo_counter"
    is_mod_zero: str = "is_mod_zero"


@dataclass(frozen=True)
class ShotsPerRandomization:
    """Configuration for randomizing angles every N shots.

    This configuration may be useful in the context of active reset so as to avoid the
    overhead of randomizing every shot.
    """

    shots_per_randomization: int
    non_randomization_delay_seconds: Optional[float] = 2e-4
    variables: ShotsPerRandomizationVariables = field(default_factory=ShotsPerRandomizationVariables)

    @property
    def pulse_program_label(self) -> InstructionDesignator:
        return JumpTarget(Label(self.variables.pulse_program_label))

    def generate_mod_shot_count_block(self, qubits_sorted: Sequence[int]) -> tuple[InstructionDesignator, ...]:
        instructions: list[InstructionDesignator] = [
            ClassicalAdd(
                MemoryReference(self.variables.modulo_counter, 0),
                1,
            ),
            ClassicalGreaterEqual(
                MemoryReference(self.variables.is_mod_zero, 0),
                MemoryReference(self.variables.modulo_counter, 0),
                self.shots_per_randomization,
            ),
            Call(
                "if_then_else_integer",
                [
                    # destination
                    inst.CallArgument.from_memory_reference(inst.MemoryReference(self.variables.modulo_counter, 0)),
                    # condition
                    inst.CallArgument.from_memory_reference(inst.MemoryReference(self.variables.is_mod_zero, 0)),
                    # true value
                    inst.CallArgument.from_immediate(complex(0, 0)),
                    # false value
                    inst.CallArgument.from_memory_reference(inst.MemoryReference(self.variables.modulo_counter, 0)),
                ],
            ),
            JumpWhen(
                Label(self.variables.randomization_label),
                MemoryReference(self.variables.is_mod_zero, 0),
            ),
        ]
        if self.non_randomization_delay_seconds is not None:
            for qubit in qubits_sorted:
                instructions.append(Delay([], [qubit], self.non_randomization_delay_seconds))
        instructions.append(Jump(Label(self.variables.pulse_program_label)))
        instructions.append(JumpTarget(Label(self.variables.randomization_label)))
        return tuple(instructions)


class _ToQuilCallArguments(ABC):
    @abstractmethod
    def to_call_arguments(self) -> tuple[inst.CallArgument, ...]: ...


class PauliLiteral(Enum):
    """A literal Pauli known at program construction time."""

    I = 0  # noqa
    X = 1
    Y = 2
    Z = 3

    @property
    def matrix(self) -> NDArray[np.complex128]:
        if self == PauliLiteral.I:
            return matrices.I
        elif self == PauliLiteral.X:
            return matrices.X
        elif self == PauliLiteral.Y:
            return matrices.Y
        elif self == PauliLiteral.Z:
            return matrices.Z
        else:
            raise ValueError(f"{self} cannot be cast to matrix")

    @classmethod
    def from_name(cls, name: str) -> "PauliLiteral":
        if name == "I":
            return cls.I
        elif name == "X":
            return cls.X
        elif name == "Y":
            return cls.Y
        elif name == "Z":
            return cls.Z
        else:
            raise ValueError(f"invalid pauli name: {name}")

    def to_call_arguments(self) -> tuple[inst.CallArgument, ...]:
        return (inst.CallArgument.from_immediate(complex(self.value, 0)),)

    def __mul__(self, rhs: "PauliLiteral") -> "tuple[complex, PauliLiteral]":
        if self == PauliLiteral.I:
            return 1, rhs
        elif rhs == PauliLiteral.I:
            return 1, self
        elif (self, rhs) == (PauliLiteral.X, PauliLiteral.X):
            return 1, PauliLiteral.I
        elif (self, rhs) == (PauliLiteral.Y, PauliLiteral.Y):
            return 1, PauliLiteral.I
        elif (self, rhs) == (PauliLiteral.Z, PauliLiteral.Z):
            return 1, PauliLiteral.I
        elif (self, rhs) == (PauliLiteral.X, PauliLiteral.Y):
            return 1j, PauliLiteral.Z
        elif (self, rhs) == (PauliLiteral.Y, PauliLiteral.Z):
            return 1j, PauliLiteral.X
        elif (self, rhs) == (PauliLiteral.Z, PauliLiteral.X):
            return 1j, PauliLiteral.Y
        elif (self, rhs) == (PauliLiteral.Y, PauliLiteral.X):
            return -1j, PauliLiteral.Z
        elif (self, rhs) == (PauliLiteral.Z, PauliLiteral.Y):
            return -1j, PauliLiteral.X
        elif (self, rhs) == (PauliLiteral.X, PauliLiteral.Z):
            return -1j, PauliLiteral.Y
        else:
            raise ValueError(f"invalid pauli multiplication: {self} * {rhs}")

    @classmethod
    def all(cls) -> tuple["PauliLiteral", ...]:
        return (cls.I, cls.X, cls.Y, cls.Z)


PAULI_CONJUGATES_MAPS = {
    "CZ": _pauli_conjugates_map_str_to_literals(
        {
            "II": "II",
            "IX": "ZX",
            "IY": "ZY",
            "IZ": "IZ",
            "XI": "XZ",
            "XX": "YY",
            "XY": "YX",
            "XZ": "XI",
            "YI": "YZ",
            "YX": "XY",
            "YY": "XX",
            "YZ": "YI",
            "ZI": "ZI",
            "ZX": "IX",
            "ZY": "IY",
            "ZZ": "ZZ",
        }
    ),
    "ISWAP": _pauli_conjugates_map_str_to_literals(
        {
            "II": "II",
            "IX": "YZ",
            "IY": "XZ",
            "IZ": "ZI",
            "XI": "ZY",
            "XX": "XX",
            "XY": "YX",
            "XZ": "IY",
            "YI": "ZX",
            "YX": "XY",
            "YY": "YY",
            "YZ": "IX",
            "ZI": "IZ",
            "ZX": "YI",
            "ZY": "XI",
            "ZZ": "ZZ",
        }
    ),
}
"""Maps from each Pauli pair to its conjugate under the specified two-qubit gate."""


def build_memory_values_for_paulis_conjugates_map(
    pauli_conjugates_map: Mapping[tuple["PauliLiteral", "PauliLiteral"], tuple["PauliLiteral", "PauliLiteral"]],
) -> Union[list[int], list[float]]:
    """Convert a Pauli conjugates map to a list of integers representing the next Pauli pair for each previous Pauli pair.

    The result may be supplied as the memory values for the `pauli_conjugates_map` memory region on the QPU (see
    `RandomizedCompilingVariables.pauli_conjugates_map`).
    """
    memory_values: list[Optional[int]] = [None] * _NUMBER_PAULI_PAIRS
    for previous_paulis, next_paulis in pauli_conjugates_map.items():
        previous_pauli_index = _pauli_pair_to_int(previous_paulis)
        next_pauli_index = _pauli_pair_to_int(next_paulis)
        memory_values[previous_pauli_index] = next_pauli_index
    return cast(Union[list[int], list[float]], memory_values)


@dataclass(frozen=True)
class _PauliReference(_ToQuilCallArguments):
    """A Pauli specified by reference to shared memory on the QPU.

    We fit a given number of Paulis in a single word of shared memory, so the precise
    Pauli within this word is specified by `pauli_index` (the control system will
    shift and mask bits to get the two bit Pauli representation).
    """

    memory_reference: inst.MemoryReference
    pauli_index: int

    def to_call_arguments(self) -> tuple[inst.CallArgument, inst.CallArgument]:
        return (
            inst.CallArgument.from_memory_reference(self.memory_reference),
            inst.CallArgument.from_immediate(complex(self.pauli_index, 0)),
        )


@dataclass(frozen=True)
class _PauliConjugate(_ToQuilCallArguments):
    """A Pauli specified by conjugation of two other Paulis.

    This is used when a previous cycle applied two random Paulis before the two qubit
    gate. We look these random Paulis up by reference and then index into
    `pauli_conjugates_map` to get the two qubit conjugation; we then select one of
    these Paulis based on `is_left_conjugate`.
    """

    pauli_left: _PauliReference
    pauli_right: _PauliReference
    is_left_conjugate: bool
    pauli_conjugates_map: str = "pauli_conjugates_map"

    def to_call_arguments(self) -> tuple[inst.CallArgument, ...]:
        arguments: list[inst.CallArgument] = []
        arguments.extend(self.pauli_left.to_call_arguments())
        arguments.extend(self.pauli_right.to_call_arguments())

        arguments.append(inst.CallArgument.from_immediate(complex(1 if self.is_left_conjugate else 0, 0)))
        arguments.append(inst.CallArgument.from_identifier(self.pauli_conjugates_map))
        return tuple(arguments)


@dataclass(frozen=True)
class _PauliPair:
    """A pair of Paulis applied on a specific qubit at a specific layer.

    The pair is read on the control system and used to apply mutations to the
    unitary angles for this (qubit, layer_index).
    """

    previous: Union[_PauliReference, PauliLiteral, _PauliConjugate]
    next: Union[_PauliReference, PauliLiteral]

    def build_quil_call_instruction(
        self,
        destination: inst.CallArgument,
        source: inst.CallArgument,
        unitary_angle_offset: inst.CallArgument,
    ) -> Union[Call, None]:
        """Build a Quil Call instruction based on the Pauli pair.

        Each underlying union variant will correspond to a different extern function signature.
        """
        arguments = [destination, source, unitary_angle_offset]
        arguments.extend(self.next.to_call_arguments())
        arguments.extend(self.previous.to_call_arguments())
        if isinstance(self.previous, PauliLiteral) and isinstance(self.next, PauliLiteral):
            if self.previous == PauliLiteral.I and self.next == PauliLiteral.I:
                # no need to call an extern function if the Paulis are both identity since this would be a no-op
                return None
            return Call(
                "merge_zxzxz_unitary_with_paulis_literal_literal",
                arguments,
            )
        elif isinstance(self.previous, _PauliReference) and isinstance(self.next, PauliLiteral):
            return Call(
                "merge_zxzxz_unitary_with_paulis_literal_reference",
                arguments,
            )
        elif isinstance(self.previous, _PauliConjugate) and isinstance(self.next, PauliLiteral):
            return Call(
                "merge_zxzxz_unitary_with_paulis_literal_conjugate",
                arguments,
            )
        elif isinstance(self.previous, _PauliReference) and isinstance(self.next, _PauliReference):
            return Call(
                "merge_zxzxz_unitary_with_paulis_reference_reference",
                arguments,
            )
        elif isinstance(self.previous, _PauliConjugate) and isinstance(self.next, _PauliReference):
            return Call(
                "merge_zxzxz_unitary_with_paulis_reference_conjugate",
                arguments,
            )
        elif isinstance(self.previous, PauliLiteral) and isinstance(self.next, _PauliReference):
            return Call(
                "merge_zxzxz_unitary_with_paulis_reference_literal",
                arguments,
            )
        else:
            raise ValueError(f"invalid pauli pair: {self.previous}, {self.next}")


def _compute_unitary_from_zxzxz_angles(unitary: Sequence[float]) -> NDArray[np.complex128]:
    """Compute the unitary matrix from ZXZXZ angles."""
    sx = matrices.RX(np.pi / 2)
    return cast(
        NDArray[np.complex128],
        matrices.RZ(unitary[2] * 2 * np.pi)
        @ sx
        @ matrices.RZ(unitary[1] * 2 * np.pi)
        @ sx
        @ matrices.RZ(unitary[0] * 2 * np.pi),
    )


@dataclass(frozen=True)
class RandomizedCompilingVariables:
    """Memory variable names for randomized compiling."""

    seed_loop_label: str = "rc_seed_loop"
    seed_index: str = "rc_seed_index"
    seed_loop_inner_label: str = "rc_seed_loop_inner"
    base_cycle_loop_label: str = "rc_base_cycle_loop"
    base_cycle_loop_index: str = "rc_base_cycle_loop_index"
    unitary_angle_offset: str = "unitary_angle_offset"
    loop_break: str = "break"
    current_seeds_prefix: str = "current_seeds"
    pauli_conjugates_map: str = "pauli_conjugates_map"
    unitaries_prefix: str = "unitaries"
    twirled_unitaries_prefix: str = "unitaries"
    pauli_seed_prefix: str = "pauli_seed"

    @property
    def twirled_overwrites_source_unitaries(self) -> bool:
        return self.twirled_unitaries_prefix == self.unitaries_prefix

    def current_seeds(self, qubit: int) -> str:
        return f"{self.current_seeds_prefix}_q{qubit}"

    def source_unitaries(self, qubit: int) -> str:
        return f"{self.unitaries_prefix}_q{qubit}"

    def source_unitaries_ref(self, qubit: int, layer_index: int, angle_index: int) -> MemoryReference:
        return MemoryReference(self.source_unitaries(qubit), layer_index * _ANGLES_PER_UNITARY + angle_index)

    def twirled_unitaries(self, qubit: int) -> str:
        return f"{self.twirled_unitaries_prefix}_q{qubit}"

    def twirled_unitaries_ref(self, qubit: int, layer_index: int, angle_index: int) -> MemoryReference:
        return MemoryReference(self.twirled_unitaries(qubit), layer_index * _ANGLES_PER_UNITARY + angle_index)

    def pauli_seed(self, qubit: int) -> str:
        return f"{self.pauli_seed_prefix}_q{qubit}"


class _PauliCursor(Enum):
    """Tracks the memory location of the previous and next Paulis.

    In order to effectively loop over random compilation seeds on the QPU to support deeper
    circuits, throughout the Quil program, we point to the `RandomizedCompilingVariables.current_seeds`
    at specific offsets representing different `_PauliReference`s.

    More specifically, the previous Pauli is generally at `_PauliReference` pointing to the first
    Pauli at `current_seeds[0]` and Pauli index 0, while the next Pauli is at `current_seeds[1]`
    and Pauli index 0.

    The exception is after we transition from one seed value to the next, in which case the
    previous Pauli is at `current_seeds[1]` Pauli index 1 and the next Pauli is at `current_seeds[0]`
    Pauli index 0.
    """

    DEFAULT_POSITION = 0
    AFTER_SEED_TRANSITION = 1

    def next_ref(self, current_seed_name: str) -> _PauliReference:
        if self == _PauliCursor.DEFAULT_POSITION:
            next_pauli_seed_index = 1
        elif self == _PauliCursor.AFTER_SEED_TRANSITION:
            next_pauli_seed_index = 0
        else:
            raise ValueError(f"invalid Pauli cursor: {self}")
        return _PauliReference(
            memory_reference=inst.MemoryReference(current_seed_name, 0), pauli_index=next_pauli_seed_index
        )

    def previous_ref(self, current_seed_name: str) -> _PauliReference:
        if self == _PauliCursor.DEFAULT_POSITION:
            previous_seed_index = 0
            previous_seed_pauli_index = 0
        elif self == _PauliCursor.AFTER_SEED_TRANSITION:
            previous_seed_index = 1
            previous_seed_pauli_index = 0
        else:
            raise ValueError(f"invalid Pauli cursor: {self}")
        return _PauliReference(
            memory_reference=inst.MemoryReference(current_seed_name, previous_seed_index),
            pauli_index=previous_seed_pauli_index,
        )


def _requires_seed_transition(
    cycle_index: int,
    is_final_cycle: bool,
    transition_to_next_seed_on_last_cycle: bool,
    paulis_per_value: int,
) -> bool:
    requires_seed_transition = is_final_cycle and transition_to_next_seed_on_last_cycle
    requires_seed_transition |= (cycle_index + 1) % paulis_per_value == 0
    return requires_seed_transition


@dataclass(frozen=True)
class RandomizedCompilingConfiguration:
    """A utility for configuring randomized compiling on a Rigetti QPU.

    This class supports the following functionality:

    * Building a Quil program that applies random Pauli gates in sequence according to specified
        base cycles and twirls the angles of the specified ZXZXZ unitaries accordingly (see
        `build_quil_program`).
    * Generating the random seeds for the random Paulis on the QPU given a numpy random generator and
        the number of qubits and layers (see `generate_random_seeds`).
    * Building a memory map for QPU execution (see `build_memory_map`).
    * Tracking the Paulis played on each qubit at each layer over a sequence of shots (see
        `track_pauli_frames`).
    * Verifying that the final memory read off the QPU is consistent with the expected random Paulis calculated
        on the client (see `verify_final_paulis`)that the final memory read off the QPU is consistent with the expected random Paulis calculated
        on the client (see `verify_final_paulis`) and, more generally, verifying

    This class does not:

    * build the gate program.
    * generate source unitaries for the gate program.
    """

    base_cycles: tuple[tuple[Union[_TEdge, int], ...], ...]
    """
    A list of cycles (which itself is a list of edges) representing the base cycles to apply in sequence
    for randomized compiling.

    The length must be either a multiple of _MAX_PAULIS_PER_VALUE or less than or equal to
    _MAX_PAULIS_PER_VALUE (24).
    """

    base_cycle_repetitions: int
    """
    The number of times to repeat the full set of base cycles.

    Note maximum execution efficiency is achieved by configuring the (base cycle length * repetitions) to be
    equal to `_base_cycle_length` plus some multiple of `_paulis_per_value / _base_cycle_length`. For instance,
    given a base cycle length 4, `_paulis_per_value` is 24. Choosing 16 repetitions, `4 * 16 = 64` and
    `4 + (24 / 4) * 10 = 64`.
    """

    variables: RandomizedCompilingVariables = field(default_factory=RandomizedCompilingVariables)
    """Configuration for variable naming conventions in the generated Quil program."""

    leading_delay_seconds: float | None = 2e-4
    """
    The delay to insert before starting the gate program. If None, no leading delay or final fence will be
    included in `open_classical_preamble` or `build_quil_program`.
    """

    shots_per_randomization: Optional[ShotsPerRandomization] = None
    """Configuration for randomizing only a subset of shots."""

    invert_random_paulis: bool = True
    """
    Whether to invert the random Paulis from the previous layer. Setting this to False may be useful
    in conjuction with `track_pauli_frames`.
    """

    skip_first_layer: bool = False

    skip_final_layer: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self._base_cycle_length > _MAX_PAULIS_PER_VALUE and self._base_cycle_length % _MAX_PAULIS_PER_VALUE != 0:
            raise ValueError(
                f"Base cycle length must be a multiple of {_MAX_PAULIS_PER_VALUE} if it exceeds {_MAX_PAULIS_PER_VALUE}, but got {self._base_cycle_length}."
            )
        if self.base_cycle_repetitions <= 0:
            raise ValueError(f"Base cycle repetitions must be greater than 0, but got {self.base_cycle_repetitions}.")

    @property
    def _paulis_per_value(self) -> int:
        """The number of Paulis reprensented in a single INTEGER memory value.

        If the base cycle length is less than `_MAX_PAULIS_PER_VALUE` and it is not a factor of `_MAX_PAULIS_PER_VALUE`,
        this will be the largest multiple of the base cycle length that is less than `_MAX_PAULIS_PER_VALUE`. This ensures
        transitioning from one seed value to the next within a loop can consistently occur at a given index. For instance,
        if the base cycle length is 5, 4 base cycles will fit within a single seed value and `_paulis_per_word` is therefore
        5 * 4 = 20. After the fourth base cycle, we drop the remaining 4 bits (2 Paulis) and transition to the next seed value;
        otherwise, we would transition after the second base cycle in the next iteration, which adds substantial complexity
        to the program at runtime.

        This assumes that when the base cycle length exceeds `_MAX_PAULIS_PER_VALUE`, the base cycle length is a multiple of
        `_MAX_PAULIS_PER_VALUE`, which is validated in `__post_init__`.
        """
        if self._base_cycle_length > _MAX_PAULIS_PER_VALUE:
            return _MAX_PAULIS_PER_VALUE
        elif _MAX_PAULIS_PER_VALUE % self._base_cycle_length == 0:
            return _MAX_PAULIS_PER_VALUE
        else:
            return self._base_cycle_length * math.floor(_MAX_PAULIS_PER_VALUE / self._base_cycle_length)

    @property
    def _base_cycle_length(self) -> int:
        return len(self.base_cycles)

    @property
    def _cycle_count(self) -> int:
        return self._base_cycle_length * self.base_cycle_repetitions

    @property
    def _seed_length(self) -> int:
        return math.ceil(self._cycle_count / self._paulis_per_value)

    @property
    def _seed_loop_length(self) -> int:
        if self._base_cycle_length < self._paulis_per_value:
            return self._seed_length - 1
        return 0

    @property
    def _seed_loop_inner_length(self) -> int:
        if self._seed_loop_length >= 1:
            return self._paulis_per_value // self._base_cycle_length - 1
        else:
            return 0

    @property
    def _base_cycle_loop_length(self) -> int:
        base_cycles_per_seed_value = self._paulis_per_value // self._base_cycle_length
        completed_base_cycles = self._seed_loop_length * base_cycles_per_seed_value
        if self._base_cycle_length == 1:
            total_required_u2_cycles = self.base_cycle_repetitions + 1
            # we subtract 2 for the initial and final cycles.
            return total_required_u2_cycles - completed_base_cycles - 2
        # in the base case, we complete a base cycle (sans the first cycle within the base cycle)
        # after the base cycle loop.
        completed_base_cycles += 1
        return self.base_cycle_repetitions - completed_base_cycles

    @cached_property
    def _base_twirled_cycles(self) -> tuple[_TwirledCycle, ...]:
        return tuple(_TwirledCycle.from_base_cycle(cycle) for cycle in self.base_cycles)

    @cached_property
    def qubits_sorted(self) -> tuple[int, ...]:
        return tuple(sorted({qubit for cycle in self._base_twirled_cycles for qubit in cycle.all_qubits}))

    def _generate_declarations(self) -> tuple[Declare, ...]:
        declarations: list[Declare] = []
        declarations.append(Declare(self.variables.pauli_conjugates_map, "INTEGER", _NUMBER_PAULI_PAIRS))

        for q in self.qubits_sorted:
            declarations.append(Declare(self.variables.pauli_seed(q), "INTEGER", self._seed_length))
        for q in self.qubits_sorted:
            declarations.append(
                Declare(
                    self.variables.twirled_unitaries(q),
                    "REAL",
                    (self._cycle_count + 1) * _ANGLES_PER_UNITARY,
                )
            )

        if self.shots_per_randomization is not None:
            declarations.extend(
                (
                    Declare(self.shots_per_randomization.variables.modulo_counter, "INTEGER", 1),
                    Declare(self.shots_per_randomization.variables.is_mod_zero, "BIT", 1),
                )
            )

        declarations.extend(
            (
                Declare(self.variables.unitary_angle_offset, "INTEGER", 1),
                Declare(self.variables.loop_break, "BIT", 1),
            )
        )
        if self._seed_loop_length > 0 or self._base_cycle_length >= self._paulis_per_value:
            declarations.append(
                Declare(self.variables.seed_index, "INTEGER", 1),
            )
        if self._base_cycle_loop_length > 0 or self._seed_loop_inner_length > 0:
            declarations.append(
                Declare(self.variables.base_cycle_loop_index, "INTEGER", 1),
            )
        current_seed_length = (
            2 if self._seed_loop_length > 0 or self._base_cycle_length >= self._paulis_per_value else 1
        )
        for q in self.qubits_sorted:
            declarations.append(Declare(self.variables.current_seeds(q), "INTEGER", current_seed_length))
        return tuple(declarations)

    def generate_seed_values(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Generate random seed values for the random Paulis on the QPU."""
        size = (len(self.qubits_sorted), self._seed_length)
        return rng.integers(0, _MAX_SEQUENCER_VALUE + 1, size=size, dtype=np.int64)

    def build_memory_map(
        self,
        random_seeds: NDArray[np.int64],
        pauli_conjugates_map: Union[list[int], list[float]],
    ) -> dict[str, Union[list[int], list[float]]]:
        """Build the memory map for executing the randomized compiling program on the QPU.

        This does not include the source unitary angles, which must separately be supplied by the user.
        """
        memory_map: dict[str, Union[list[int], list[float]]] = {
            self.variables.unitary_angle_offset: [_ANGLES_PER_UNITARY],
            self.variables.loop_break: [0],
        }
        if self._seed_loop_length > 0 or self._base_cycle_length >= self._paulis_per_value:
            memory_map[self.variables.seed_index] = [0]
        if self._base_cycle_loop_length > 0 or self._seed_loop_inner_length > 0:
            memory_map[self.variables.base_cycle_loop_index] = [0]

        memory_map[self.variables.pauli_conjugates_map] = pauli_conjugates_map
        for qubit_index, q in enumerate(self.qubits_sorted):
            memory_map[self.variables.pauli_seed(q)] = random_seeds[qubit_index].tolist()
            memory_map[self.variables.twirled_unitaries(q)] = np.zeros(
                ((self._cycle_count + 1) * _ANGLES_PER_UNITARY,), dtype=float
            ).tolist()

        current_seed_length = (
            2 if self._seed_loop_length > 0 or self._base_cycle_length >= self._paulis_per_value else 1
        )
        for q in self.qubits_sorted:
            memory_map[self.variables.current_seeds(q)] = [0] * current_seed_length

        if self.shots_per_randomization is not None:
            memory_map[self.shots_per_randomization.variables.modulo_counter] = [
                self.shots_per_randomization.shots_per_randomization - 1
            ]
            memory_map[self.shots_per_randomization.variables.is_mod_zero] = [0]

        return memory_map

    def _build_quil_instructions_for_base_cycle(
        self,
        /,
        transition_to_next_seed_on_last_cycle: bool = False,
        is_final_base_cycle: bool = False,
    ) -> list[InstructionDesignator]:
        instructions: list[InstructionDesignator] = []
        for cycle_index, cycle in enumerate(self._base_twirled_cycles):
            is_final_cycle = cycle_index == self._base_cycle_length - 1
            if is_final_cycle and is_final_base_cycle and self.skip_final_layer:
                break
            requires_seed_transition = _requires_seed_transition(
                cycle_index=cycle_index,
                is_final_cycle=is_final_cycle,
                transition_to_next_seed_on_last_cycle=transition_to_next_seed_on_last_cycle,
                paulis_per_value=self._paulis_per_value,
            )
            if requires_seed_transition:
                instructions.extend(
                    self._build_quil_instructions_for_seed_transition(
                        advance_next_seed=not (is_final_cycle and is_final_base_cycle)
                    )
                )
                cursor = _PauliCursor.AFTER_SEED_TRANSITION
            else:
                cursor = _PauliCursor.DEFAULT_POSITION

            for qubit in self.qubits_sorted:
                edge = cycle.two_qubit_gates[qubit] if qubit in cycle.two_qubit_gates else None
                previous: Union[_PauliConjugate, _PauliReference, PauliLiteral]
                if self.invert_random_paulis and edge is not None:
                    pauli_left = cursor.previous_ref(self.variables.current_seeds(edge[0]))
                    pauli_right = cursor.previous_ref(self.variables.current_seeds(edge[1]))
                    is_pauli_left = qubit == edge[0]
                    previous = _PauliConjugate(
                        pauli_left=pauli_left,
                        pauli_right=pauli_right,
                        is_left_conjugate=is_pauli_left,
                    )
                elif self.invert_random_paulis and qubit in cycle.idle_qubits:
                    previous = cursor.previous_ref(self.variables.current_seeds(qubit))
                else:
                    previous = PauliLiteral.I

                next_: Union[_PauliReference, PauliLiteral]
                next_cycle = (
                    self._base_twirled_cycles[cycle_index + 1] if cycle_index < self._base_cycle_length - 1 else None
                )
                if is_final_cycle and is_final_base_cycle or (next_cycle is not None and qubit not in next_cycle):
                    next_ = PauliLiteral.I
                else:
                    next_ = cursor.next_ref(self.variables.current_seeds(qubit))
                pauli_pair = _PauliPair(
                    previous=previous,
                    next=next_,
                )
                call = pauli_pair.build_quil_call_instruction(
                    inst.CallArgument.from_identifier(self.variables.twirled_unitaries(qubit)),
                    inst.CallArgument.from_identifier(self.variables.source_unitaries(qubit)),
                    inst.CallArgument.from_memory_reference(
                        inst.MemoryReference(self.variables.unitary_angle_offset, 0)
                    ),
                )
                if call is not None:
                    instructions.append(call)

            if not (is_final_cycle and is_final_base_cycle):
                instructions.append(
                    ClassicalAdd(
                        MemoryReference(self.variables.unitary_angle_offset, 0),
                        _ANGLES_PER_UNITARY,
                    ),
                )

                if not requires_seed_transition:
                    for q in self.qubits_sorted:
                        instructions.append(
                            ClassicalShiftRight(
                                MemoryReference(self.variables.current_seeds(q), 0),
                                _BITS_PER_PAULI,
                            )
                        )

        return instructions

    def _build_quil_instructions_for_seed_transition(
        self, advance_next_seed: bool = True
    ) -> list[InstructionDesignator]:
        instructions: list[InstructionDesignator] = []
        instructions.extend(
            [
                ClassicalMove(
                    MemoryReference(self.variables.current_seeds(qubit), 1),
                    MemoryReference(self.variables.current_seeds(qubit), 0),
                )
                for qubit in self.qubits_sorted
            ]
        )
        if advance_next_seed:
            instructions.extend(
                [
                    ClassicalLoad(
                        MemoryReference(self.variables.current_seeds(qubit), 0),
                        self.variables.pauli_seed(qubit),
                        MemoryReference(self.variables.seed_index, 0),
                    )
                    for qubit in self.qubits_sorted
                ]
            )
            instructions.append(
                ClassicalAdd(
                    MemoryReference(self.variables.seed_index, 0),
                    1,
                )
            )
        return instructions

    def _build_quil_loop_instructions(
        self,
        instructions: list[InstructionDesignator],
        loop_label: str,
        loop_index_variable: str,
        loop_index_end: int,
        loop_index_start: int = 0,
        loop_index_increment: Optional[int] = 1,
    ) -> list[InstructionDesignator]:
        loop_instructions: list[InstructionDesignator] = []
        loop_instructions.append(ClassicalMove(MemoryReference(loop_index_variable, 0), loop_index_start))
        loop_instructions.append(JumpTarget(Label(loop_label)))
        loop_instructions.extend(instructions)
        if loop_index_increment is not None:
            loop_instructions.append(
                ClassicalAdd(
                    MemoryReference(loop_index_variable, 0),
                    loop_index_increment,
                )
            )
        loop_instructions.append(
            ClassicalGreaterEqual(
                MemoryReference(self.variables.loop_break, 0),
                MemoryReference(loop_index_variable, 0),
                loop_index_end,
            )
        )
        loop_instructions.append(
            JumpUnless(
                Label(loop_label),
                MemoryReference(self.variables.loop_break, 0),
            )
        )
        return loop_instructions

    def _build_quil_instructions_for_randomized_compiling(self) -> list[InstructionDesignator]:
        instructions: list[InstructionDesignator] = []
        for q in self.qubits_sorted:
            for i in range(self._seed_length):
                instructions.append(
                    Call(
                        "prng_set_seed_and_step",
                        [
                            inst.CallArgument.from_memory_reference(
                                inst.MemoryReference(self.variables.pauli_seed(q), i)
                            ),
                            inst.CallArgument.from_memory_reference(
                                inst.MemoryReference(self.variables.pauli_seed(q), i)
                            ),
                        ],
                    )
                )
        # first cycle.
        if not self.skip_first_layer:
            for q in self.qubits_sorted:
                pauli_pair = _PauliPair(
                    previous=PauliLiteral.I,
                    next=_PauliReference(
                        memory_reference=inst.MemoryReference(self.variables.pauli_seed(q), 0), pauli_index=0
                    ),
                )
                call = pauli_pair.build_quil_call_instruction(
                    inst.CallArgument.from_identifier(self.variables.twirled_unitaries(q)),
                    inst.CallArgument.from_identifier(self.variables.source_unitaries(q)),
                    inst.CallArgument.from_immediate(complex(0, 0)),
                )
                if call is not None:
                    instructions.append(call)
        instructions.append(
            ClassicalMove(
                MemoryReference(self.variables.unitary_angle_offset, 0),
                _ANGLES_PER_UNITARY,
            ),
        )
        for qubit in self.qubits_sorted:
            instructions.append(
                ClassicalMove(
                    MemoryReference(self.variables.current_seeds(qubit), 0),
                    MemoryReference(self.variables.pauli_seed(qubit), 0),
                )
            )

        if self._seed_loop_length >= 1:
            seed_loop_instructions = []
            if self._seed_loop_inner_length >= 1:
                inner_loop = self._build_quil_loop_instructions(
                    self._build_quil_instructions_for_base_cycle(),
                    loop_label=self.variables.seed_loop_inner_label,
                    loop_index_variable=self.variables.base_cycle_loop_index,
                    loop_index_end=self._seed_loop_inner_length,
                )
                seed_loop_instructions.extend(inner_loop)
            seed_loop_instructions.extend(
                self._build_quil_instructions_for_base_cycle(transition_to_next_seed_on_last_cycle=True)
            )
            seed_loop = self._build_quil_loop_instructions(
                seed_loop_instructions,
                loop_label=self.variables.seed_loop_label,
                loop_index_variable=self.variables.seed_index,
                loop_index_start=1,
                loop_index_end=self._seed_loop_length + 1,
                # seed index is incremented within the seed transition instructions
                loop_index_increment=None,
            )
            instructions.extend(seed_loop)
        elif self._base_cycle_length >= self._paulis_per_value:
            instructions.append(
                ClassicalMove(
                    MemoryReference(self.variables.seed_index, 0),
                    1,
                )
            )

        if self._base_cycle_loop_length >= 1:
            base_loop = self._build_quil_loop_instructions(
                self._build_quil_instructions_for_base_cycle(),
                loop_label=self.variables.base_cycle_loop_label,
                loop_index_variable=self.variables.base_cycle_loop_index,
                loop_index_end=self._base_cycle_loop_length,
            )
            instructions.extend(base_loop)

        final_base_cycle = self._build_quil_instructions_for_base_cycle(is_final_base_cycle=True)
        instructions.extend(final_base_cycle)

        return instructions

    @contextmanager
    def open_classical_preamble(self) -> Generator[Program, None, None]:
        """Generate a cycle program with randomized compilation according to the specified configuration.

        Note, this does not include the gate program instructions.

        In contrast to `build_quil_program`, this yields the program from a contextmanager before
        wrapping the final block of the classical preamble in the necessary delays and fences.
        """
        program = Program()
        program += list(build_extern_function_signatures().values())
        program += list(self._generate_declarations())

        if self.shots_per_randomization is not None:
            program += list(self.shots_per_randomization.generate_mod_shot_count_block(self.qubits_sorted))

        program += self._build_quil_instructions_for_randomized_compiling()

        if self.shots_per_randomization is not None:
            program += self.shots_per_randomization.pulse_program_label

        yield program
        if self.leading_delay_seconds is not None:
            delay_and_fence_classical_preamble(
                program, [Delay([], [qubit], self.leading_delay_seconds) for qubit in self.qubits_sorted], [FENCE()]
            )

    def build_quil_program(
        self,
    ) -> Program:
        """Generate a cycle program with randomized compilation according to the specified configuration.

        Note, this does not include the gate program instructions.
        """
        with self.open_classical_preamble() as program:
            pass
        return program

    def apply_pauli_pair(
        self,
        qubit: int,
        layer_index: int,
        source_unitaries: Optional[str] = None,
        target_unitaries: Optional[str] = None,
        unitary_offset: Optional[Union[MemoryReference, int, float]] = None,
    ) -> Union[Call, None]:
        """Apply the twirl to the source unitary for a given qubit and layer index.

        This function is for applying an existing twirl to a source unitary for a specific qubit and layer index. Note,
        this does not step the PRNG. This is useful when `ShotsPerRandomization` is configured and we want to apply the existing
        twirl a freshly drawn unitary (typically by invoking `choose_random_real_sub_regions`).
        """
        previous_pauli: Union[_PauliReference, _PauliConjugate, PauliLiteral]
        if layer_index == 0 or not self.invert_random_paulis:
            previous_pauli = PauliLiteral.I
        else:
            previous_twirled_cycle_index = (layer_index - 1) % self._base_cycle_length
            previous_twirled_cycle = self._base_twirled_cycles[previous_twirled_cycle_index]
            if qubit in previous_twirled_cycle.two_qubit_gates:
                edge = previous_twirled_cycle.two_qubit_gates[qubit]
                is_pauli_left = qubit == edge[0]
                previous_layer_index = layer_index - 1
                seed_index = previous_layer_index // self._paulis_per_value
                pauli_index = previous_layer_index % self._paulis_per_value
                previous_pauli = _PauliConjugate(
                    pauli_left=_PauliReference(
                        memory_reference=inst.MemoryReference(self.variables.pauli_seed(edge[0]), seed_index),
                        pauli_index=pauli_index,
                    ),
                    pauli_right=_PauliReference(
                        memory_reference=inst.MemoryReference(self.variables.pauli_seed(edge[1]), seed_index),
                        pauli_index=pauli_index,
                    ),
                    is_left_conjugate=is_pauli_left,
                )
            elif qubit in previous_twirled_cycle.idle_qubits:
                previous_layer_index = layer_index - 1
                seed_index = previous_layer_index // self._paulis_per_value
                pauli_index = previous_layer_index % self._paulis_per_value
                previous_pauli = _PauliReference(
                    memory_reference=inst.MemoryReference(self.variables.pauli_seed(qubit), seed_index),
                    pauli_index=pauli_index,
                )
            else:
                previous_pauli = PauliLiteral.I

        next_pauli: Union[_PauliReference, PauliLiteral]
        if layer_index == self._cycle_count:
            next_pauli = PauliLiteral.I
        else:
            next_twirled_cycle_index = layer_index % self._base_cycle_length
            next_twirled_cycle = self._base_twirled_cycles[next_twirled_cycle_index]
            if qubit in next_twirled_cycle:
                next_seed_index = layer_index // self._paulis_per_value
                next_pauli_index = layer_index % self._paulis_per_value
                next_pauli = _PauliReference(
                    memory_reference=inst.MemoryReference(self.variables.pauli_seed(qubit), next_seed_index),
                    pauli_index=next_pauli_index,
                )
            else:
                next_pauli = PauliLiteral.I
        pauli_pair = _PauliPair(
            previous=previous_pauli,
            next=next_pauli,
        )
        if unitary_offset is None:
            unitary_offset_argument = inst.CallArgument.from_memory_reference(
                inst.MemoryReference(self.variables.unitary_angle_offset, 0)
            )
        elif isinstance(unitary_offset, int):
            unitary_offset_argument = inst.CallArgument.from_immediate(complex(unitary_offset))
        elif isinstance(unitary_offset, MemoryReference):
            unitary_offset_argument = inst.CallArgument.from_memory_reference(
                inst.MemoryReference(unitary_offset.name, unitary_offset.offset)
            )
        else:
            unitary_offset_argument = inst.CallArgument.from_immediate(complex(unitary_offset))

        return pauli_pair.build_quil_call_instruction(
            inst.CallArgument.from_identifier(source_unitaries or self.variables.source_unitaries(qubit)),
            inst.CallArgument.from_identifier(target_unitaries or self.variables.twirled_unitaries(qubit)),
            unitary_offset_argument,
        )

    def verify_final_memory(
        self,
        final_memory: dict[str, Union[list[int], list[float]]],
        original_memory: dict[str, Union[list[int], list[float]]],
        shot_count: int,
        pauli_conjugates_map: Mapping[tuple[PauliLiteral, PauliLiteral], tuple[PauliLiteral, PauliLiteral]],
    ) -> None:
        """Verify that the final memory state matches expectations.

        Specifically, we take the Pauli seeds specified in the original memory map and
        generate the expected final random value using `_generate_lfsr_v1_sequence` and
        the shot count. We then use this final seed to infer the pair of Paulis merged
        for each qubit at every layer. We can then apply this Pauli pair to the original
        unitaries specified for the (qubit, layer) and verify that the resulting unitary
        is equal to the twirled unitaries read from the final memory for the (qubit, layer).
        """
        pauli_pairs = self.get_final_pauli_pairs(
            shot_count=shot_count,
            pauli_conjugates_map=pauli_conjugates_map,
            random_seeds=np.asarray(
                [
                    [original_memory[self.variables.pauli_seed(qubit)][i] for i in range(self._seed_length)]
                    for qubit in self.qubits_sorted
                ],
                dtype=np.int64,
            ),
        )
        cycles = self._base_twirled_cycles * self.base_cycle_repetitions
        for q in self.qubits_sorted:
            for layer_index in range(len(cycles) + 1):
                key = PauliPairKey(
                    qubit=q,
                    layer_index=layer_index,
                )
                expected_final_seed_value, expected_final_pauli_pair = pauli_pairs[key]
                if expected_final_seed_value is not None:
                    seed_index = layer_index // self._paulis_per_value
                    found_final_pauli_seed = _i48_to_u48(int(final_memory[self.variables.pauli_seed(q)][seed_index]))
                    if found_final_pauli_seed != expected_final_seed_value:
                        raise ValueError(
                            f"final seed value mismatch for q{q}, l{layer_index}: got "
                            f"{found_final_pauli_seed}, expected {expected_final_seed_value}"
                        )
                if layer_index == 0 and self.skip_first_layer:
                    continue
                if layer_index == len(cycles) and self.skip_final_layer:
                    continue
                start_angle = layer_index * _ANGLES_PER_UNITARY
                end_angle = start_angle + _ANGLES_PER_UNITARY
                found_final_unitary_angles = tuple(
                    final_memory[self.variables.twirled_unitaries(q)][start_angle:end_angle]
                )
                found_final_unitary = _compute_unitary_from_zxzxz_angles(found_final_unitary_angles)

                source_unitary_angles = original_memory[self.variables.source_unitaries(q)][start_angle:end_angle]
                source_unitary = _compute_unitary_from_zxzxz_angles(source_unitary_angles)
                expected_unitary = (
                    expected_final_pauli_pair[1].matrix @ source_unitary @ expected_final_pauli_pair[0].matrix
                )
                if not _unitary_equal(found_final_unitary, expected_unitary):
                    raise ValueError(
                        f"unitary mismatch for q{q} layer {layer_index}: got {found_final_unitary_angles} "
                        f"for source {source_unitary_angles} and final pauli pair: {expected_final_pauli_pair}"
                    )

    def get_final_pauli_pairs(
        self,
        shot_count: int,
        pauli_conjugates_map: Mapping[tuple[PauliLiteral, PauliLiteral], tuple[PauliLiteral, PauliLiteral]],
        random_seeds: NDArray[np.int64],
        accumulate: Optional[bool] = None,
    ) -> dict[PauliPairKey, tuple[Optional[int], tuple[PauliLiteral, PauliLiteral]]]:
        """Get the final Pauli frames for each qubit and layer after a sequence of shots.

        This is useful for verifying that the final memory read off the QPU is consistent with the expected random Paulis calculated
        on the client (see `verify_final_paulis`).
        """
        cycles = self._base_twirled_cycles * self.base_cycle_repetitions
        accumulate = accumulate if accumulate is not None else self.variables.twirled_overwrites_source_unitaries
        sequence_count = shot_count
        if self.shots_per_randomization is not None:
            sequence_count = math.ceil(shot_count / self.shots_per_randomization.shots_per_randomization)
        if accumulate:
            prng_sequence_steps = 1
            prng_sequence_count = sequence_count
        else:
            prng_sequence_steps = sequence_count
            prng_sequence_count = 1

        pauli_cache = _PauliSeedAndPairCache(
            original_seeds={q: random_seeds[qubit_index].tolist() for qubit_index, q in enumerate(self.qubits_sorted)},
            pauli_conjugates_map=pauli_conjugates_map,
            cycles=cycles,
            qubits_sorted=self.qubits_sorted,
            prng_sequence_steps=prng_sequence_steps,
            invert_random_paulis=self.invert_random_paulis,
            paulis_per_value=self._paulis_per_value,
        )
        return pauli_cache.accumulate(prng_sequence_count)

    def track_pauli_frames(
        self,
        shot_count: int,
        pauli_conjugates_map: Mapping[tuple[PauliLiteral, PauliLiteral], tuple[PauliLiteral, PauliLiteral]],
        random_seeds: NDArray[np.int64],
        accumulate: Optional[bool] = None,
    ) -> Generator[dict[PauliPairKey, tuple[Optional[int], tuple[PauliLiteral, PauliLiteral]]], None, None]:
        cycles = self._base_twirled_cycles * self.base_cycle_repetitions
        accumulate = accumulate if accumulate is not None else self.variables.twirled_overwrites_source_unitaries
        pauli_cache = _PauliSeedAndPairCache(
            original_seeds={q: random_seeds[qubit_index].tolist() for qubit_index, q in enumerate(self.qubits_sorted)},
            pauli_conjugates_map=pauli_conjugates_map,
            cycles=cycles,
            qubits_sorted=self.qubits_sorted,
            prng_sequence_steps=1,
            invert_random_paulis=self.invert_random_paulis,
            paulis_per_value=self._paulis_per_value,
        )
        accumulated_pauli_pairs = None
        sequence_count = shot_count
        if self.shots_per_randomization is not None:
            sequence_count = math.ceil(shot_count / self.shots_per_randomization.shots_per_randomization)
        for sequence_index in range(sequence_count):
            pauli_pairs = pauli_cache.accumulate(1)
            if accumulate:
                if accumulated_pauli_pairs is None:
                    accumulated_pauli_pairs = pauli_pairs
                else:
                    accumulated_pauli_pairs = _accumulate_pauli_pairs(accumulated_pauli_pairs, pauli_pairs)
                yield accumulated_pauli_pairs
            else:
                yield pauli_pairs
            if sequence_index < sequence_count:
                pauli_cache = next(pauli_cache)


def _i48_to_u48(value: int) -> int:
    if value < 0:
        value = (-value ^ _MAX_SEQUENCER_VALUE) + 1
    return value
