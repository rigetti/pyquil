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
Noise model container and program-level fidelity estimation.

This module defines:

- ``NoiseModelLike``: A ``typing.Protocol`` defining the interface that all noise
  models must satisfy (a single ``get_channel`` method).
- ``NoiseModel``: The primary concrete implementation — a frozen dataclass that
  collects per-instruction noise channels.
- ``DepolarizingNoiseModel``: A convenience implementation that returns a
  depolarizing channel for any gate.
- ``CompositeNoiseModel``: Chains multiple noise models, returning the first
  non-None channel.
- ``NO_NOISE``: A sentinel noise model that always returns ``None``.
- Program-level fidelity estimation utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import mul
from typing import (
    TYPE_CHECKING,
    Iterable,
    Protocol,
    Sequence,
    overload,
    runtime_checkable,
)

import quax as qx

from pyquil.external.rpcq import CompilerISA
from pyquil.quilbase import Gate, Measurement, ResetQubit

from pyquil.noise._channels import Channel, CycleChannel, MeasurementChannel, ResetChannel

if TYPE_CHECKING:
    from pyquil import Program

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────

# Channel union type returned by get_channel
ChannelType = Channel | MeasurementChannel | ResetChannel | CycleChannel


@runtime_checkable
class NoiseModelLike(Protocol):
    """Protocol defining the noise model interface.

    Any object that implements ``get_channel`` with the correct signature can be
    used wherever a noise model is expected. This enables alternative noise model
    strategies (depolarizing, dynamic, crosstalk-aware) without modifying
    consumers.

    The standard concrete implementation is :class:`NoiseModel`.
    """

    @overload
    def get_channel(self, inst: Gate) -> Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> ResetChannel | None: ...

    def get_channel(self, inst: Gate | Measurement | ResetQubit) -> ChannelType | None:
        """Retrieve the noise channel for a specific instruction.

        :param inst: A gate, measurement, or reset instruction.
        :return: The associated noise channel, or ``None`` if the instruction
            should be treated as ideal (noiseless).
        """
        ...


@dataclass(frozen=True)
class NoiseModel:
    """
    A noise model collects all the noise channels for a given quantum program.

    This includes gate channels, measurement channels, reset channels, and cycle channels.

    The constructor accepts any iterable of channels (list, tuple, set, generator, etc.)
    which is coerced to a tuple for immutable storage.
    """

    channels: tuple[Channel | MeasurementChannel | ResetChannel | CycleChannel, ...]
    """Immutable tuple of all noise channels in the model."""

    def __init__(
        self,
        channels: Iterable[Channel | MeasurementChannel | ResetChannel | CycleChannel] = (),
    ) -> None:
        # Accept any iterable, coerce to tuple for immutable storage.
        if isinstance(channels, tuple):
            object.__setattr__(self, "channels", channels)
        else:
            object.__setattr__(self, "channels", tuple(channels))

    @cached_property
    def _channel_map(
        self,
    ) -> dict[Gate | Measurement | ResetQubit, Channel | MeasurementChannel | ResetChannel | CycleChannel]:
        """Map from instruction to channel for fast lookup."""
        return {ch.inst: ch for ch in self.channels}

    @overload
    def get_channel(self, inst: Gate) -> Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> ResetChannel | None: ...

    def get_channel(
        self, inst: Gate | Measurement | ResetQubit
    ) -> Channel | MeasurementChannel | ResetChannel | CycleChannel | None:
        """
        Retrieve the noise channel associated with a specific instruction.

        :param inst: The instruction (gate, measurement, or reset) for which to retrieve the noise channel.
        :return: The noise channel associated with the instruction, or None if no channel is found.
        """
        return self._channel_map.get(inst)

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_isa(cls: type[NoiseModel], compiler_isa: "CompilerISA") -> "NoiseModel":
        """
        Create a noise model from an instruction set architecture.

        Gate fidelities are converted to depolarizing channels and measurement
        errors are symmetric. Only gates with concrete numeric parameters are
        included.

        :param compiler_isa: The compiler ISA.
        :return: A NoiseModel with channels according to the provided fidelities.
        """
        from pyquil.external.rpcq import GateInfo, MeasureInfo
        from pyquil.quilatom import Qubit as QuilQubit

        channels: dict[Gate | Measurement, Channel | MeasurementChannel | ResetChannel | CycleChannel] = {}
        seen_measure_qubits: set[int] = set()

        for qubit_label, qubit in compiler_isa.qubits.items():
            for op_info in qubit.gates:
                if isinstance(op_info, GateInfo):
                    qubits = [int(qubit_label)]
                    gate_name = op_info.operator
                    fidelity = op_info.fidelity
                    params = op_info.parameters

                    if gate_name is None:
                        continue
                    # Skip gates with non-numeric parameters
                    if not all(isinstance(p, (float, int, complex)) for p in params):
                        continue

                    numeric_params: list[float] = [float(p) for p in params if isinstance(p, (float, int, complex))]
                    inst = Gate(name=gate_name, params=numeric_params, qubits=qubits)
                    if fidelity is not None and fidelity < 1.0:
                        channels[inst] = Channel.from_gate_fidelity(inst=inst, fidelity=fidelity)

                elif isinstance(op_info, MeasureInfo):
                    if op_info.qubit is None:
                        continue
                    # Use qubit_label from the enclosing section when qubit is a wildcard
                    qubit_str = op_info.qubit if op_info.qubit != "_" else qubit_label
                    try:
                        qubit_idx = int(qubit_str)
                    except (ValueError, TypeError):
                        continue
                    fidelity = op_info.fidelity
                    if qubit_idx in seen_measure_qubits:
                        continue
                    seen_measure_qubits.add(qubit_idx)
                    if fidelity is None:
                        continue
                    m_inst = Measurement(qubit=QuilQubit(qubit_idx), classical_reg=None)
                    channels[m_inst] = MeasurementChannel.from_readout_fidelity(inst=m_inst, fidelity=fidelity)

        for edge_label, edge in compiler_isa.edges.items():
            for op_info in edge.gates:
                if isinstance(op_info, GateInfo):
                    qubits = [int(q) for q in edge_label.split("-")]
                    gate_name = op_info.operator
                    fidelity = op_info.fidelity
                    params = op_info.parameters

                    if gate_name is None:
                        continue
                    if not all(isinstance(p, (float, int, complex)) for p in params):
                        continue

                    numeric_params = [float(p) for p in params if isinstance(p, (float, int, complex))]
                    inst = Gate(name=gate_name, params=numeric_params, qubits=qubits)
                    if fidelity is not None and fidelity < 1.0:
                        channels[inst] = Channel.from_gate_fidelity(inst=inst, fidelity=fidelity)

        return cls(channels=channels.values())

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """
        Serialize NoiseModel to a JSON string.

        :return: JSON string representation.
        """
        channel_data = []
        for ch in self.channels:
            if isinstance(ch, (Channel, MeasurementChannel, ResetChannel, CycleChannel)):
                channel_data.append({"type": type(ch).__name__, "data": ch.to_json()})
            else:
                logger.warning(f"Skipping serialization of {type(ch).__name__} (not yet supported).")
        return json.dumps({"channels": channel_data})

    @classmethod
    def from_json(cls: type[NoiseModel], json_str: str) -> "NoiseModel":
        """
        Deserialize a NoiseModel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: NoiseModel instance.
        """
        data = json.loads(json_str)
        _type_map = {
            "Channel": Channel,
            "MeasurementChannel": MeasurementChannel,
            "ResetChannel": ResetChannel,
            "CycleChannel": CycleChannel,
        }
        channels: list[Channel | MeasurementChannel | ResetChannel | CycleChannel] = []
        for ch_data in data["channels"]:
            ch_cls = _type_map.get(ch_data["type"])
            if ch_cls is None:
                raise ValueError(f"Unknown channel type: {ch_data['type']}")
            channels.append(ch_cls.from_json(ch_data["data"]))
        return cls(channels=channels)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        """Check if two NoiseModels contain equivalent channel maps."""
        if not isinstance(other, NoiseModel):
            return False
        return self._channel_map == other._channel_map

    def __hash__(self) -> int:
        """Hash based on id (NoiseModel is not value-hashable due to array contents)."""
        return id(self)

    def __add__(self, other: "NoiseModel") -> "NoiseModel":
        """
        Combine two NoiseModels.

        For channels with matching instructions, compose them (``channel_A @ channel_B``).
        For non-overlapping channels, include both.
        """
        if not isinstance(other, NoiseModel):
            return NotImplemented

        my_channels = {ch.inst: ch for ch in self.channels}
        other_channels = {ch.inst: ch for ch in other.channels}

        combined: list[Channel | MeasurementChannel | ResetChannel | CycleChannel] = []
        all_insts = list(dict.fromkeys(list(my_channels) + list(other_channels)))
        for inst in all_insts:
            mine = my_channels.get(inst)
            theirs = other_channels.get(inst)
            if mine is not None and theirs is not None:
                # Both have a channel for this instruction — compose them
                # (only same-type composition is defined)
                composed = mine @ theirs  # type: ignore[operator]
                combined.append(composed)  # type: ignore[arg-type]
            elif mine is not None:
                combined.append(mine)
            elif theirs is not None:
                combined.append(theirs)

        return NoiseModel(channels=combined)


# ──────────────────────────────────────────────────────────
# Convenience NoiseModelLike implementations
# ──────────────────────────────────────────────────────────


NOISELESS: NoiseModelLike = NoiseModel()
"""Sentinel noise model that applies no noise to any instruction."""


@dataclass(frozen=True)
class DepolarizingNoiseModel:
    """A noise model that applies uniform depolarizing noise to every gate.

    For any ``Gate`` instruction, returns a :class:`Channel` with the specified
    depolarizing constant.  Measurements and resets are treated as ideal.

    :param depolarizing_constant: The depolarization constant :math:`p` where
        :math:`\\mathcal{D}_p(\\rho) = p \\, \\rho + (1-p) \\, I/d`.
        A value of 1.0 means no noise; 0.0 means full depolarization.
    """

    depolarizing_constant: float

    @overload
    def get_channel(self, inst: Gate) -> Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> ResetChannel | None: ...

    def get_channel(self, inst: Gate | Measurement | ResetQubit) -> ChannelType | None:
        """Return a depolarizing channel for gates; ``None`` for measurements/resets."""
        if isinstance(inst, Gate):
            return Channel.from_depolarizing_constant(inst, self.depolarizing_constant)
        return None


@dataclass(frozen=True)
class CompositeNoiseModel:
    """A noise model that chains multiple models, returning the first non-None channel.

    Models are queried in order. The first model that returns a non-None channel
    for a given instruction wins.

    :param models: Sequence of noise models to query in priority order.
    """

    models: tuple[NoiseModelLike, ...]

    def __init__(self, models: Sequence[NoiseModelLike]) -> None:
        object.__setattr__(self, "models", tuple(models))

    @overload
    def get_channel(self, inst: Gate) -> Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> ResetChannel | None: ...

    def get_channel(self, inst: Gate | Measurement | ResetQubit) -> ChannelType | None:
        """Query each model in order, returning the first non-None result."""
        for model in self.models:
            channel = model.get_channel(inst)
            if channel is not None:
                return channel
        return None


# ──────────────────────────────────────────────────────────
# Program-level fidelity estimation
# ──────────────────────────────────────────────────────────


def estimate_program_fidelity(program: Program, noise_model: NoiseModelLike) -> float:
    """
    Estimate the program fidelity for a given noise model.

    Works by multiplying the gate process fidelities together. Readout noise
    is not considered.

    :param program: The program of interest.
    :param noise_model: A noise model.
    :return: The estimated process fidelity.
    """
    gate_fidelities = [1.0]
    for inst in program.instructions:
        if isinstance(inst, Gate):
            channel = noise_model.get_channel(inst)
            if channel is not None and isinstance(channel, Channel):
                gate_fidelities.append(channel.pauli_fidelity)
            elif channel is not None and isinstance(channel, CycleChannel):
                gate_fidelities.append(channel.pauli_fidelity)

    return reduce(mul, gate_fidelities)


def _light_cone_program(program: Program, qubits: list[int]) -> Program:
    """Return a sub-program containing only gates in the backward light cone of *qubits*.

    Walks backward through the program's gate instructions. Any gate that
    acts on a qubit currently in the light-cone set is included, and all of
    its qubits are added to the set (because earlier gates on those qubits
    are now causally relevant).
    """
    from pyquil import Program as _Program

    gate_instructions = [inst for inst in program.instructions if isinstance(inst, Gate)]
    relevant_qubits = set(qubits)
    included: list[Gate] = []
    for inst in reversed(gate_instructions):
        inst_qubits = {q.index for q in inst.qubits}
        if inst_qubits & relevant_qubits:
            included.append(inst)
            relevant_qubits |= inst_qubits
    reduced = _Program()
    for inst in reversed(included):
        reduced += inst
    return reduced


def estimate_program_observable_fidelity(
    program: Program,
    noise_model: NoiseModelLike,
    observable: "PauliSum" | "PauliTerm",
) -> float:
    """Estimate program fidelity restricted to the backward light cone of *observable*.

    Reduces the program to only the gates causally connected to the
    observable qubits, then multiplies gate process fidelities together.
    Readout noise is not considered.

    :param program: The program of interest.
    :param noise_model: A noise model.
    :param observable: A ``PauliTerm`` or ``PauliSum`` whose qubits define
        the light cone.
    :return: The estimated process fidelity for the light-cone-reduced program.
    """
    from pyquil.paulis import PauliSum, PauliTerm

    if isinstance(observable, PauliTerm):
        observable = PauliSum(terms=[observable])

    qubits = [int(q) for term in observable.terms for q, _ in term.operations_as_set()]
    reduced_program = _light_cone_program(program, qubits)
    return estimate_program_fidelity(reduced_program, noise_model)
