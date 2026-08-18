##############################################################################
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
##############################################################################
"""Noise model container and program-level fidelity estimation.

This module defines:

- ``NoiseModelLike``: A ``typing.Protocol`` defining the interface that all noise
  models must satisfy (a single ``get_channel`` method).
- ``NoiseModel``: The primary concrete implementation — a frozen dataclass that
  collects per-instruction noise channels.
- ``DepolarizingNoiseModel``: A convenience implementation that returns a
  depolarizing channel for any gate.
- Program-level fidelity estimation utilities.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)

from deprecated import deprecated

from pyquil.external.rpcq import CompilerISA
from pyquil.noise._channels import (
    Channel,
    CycleChannel,
    MeasurementChannel,
    ResetChannel,
    SuperopChannel,
    SuperopResetChannel,
    _ChannelBase,
)
from pyquil.quilbase import Gate, Measurement, ResetQubit

if TYPE_CHECKING:
    from qcs_sdk.qpu.isa import InstructionSetArchitecture

    from pyquil import Program
    from pyquil.paulis import PauliSum, PauliTerm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────

ChannelType: TypeAlias = (
    SuperopChannel | Channel | MeasurementChannel | SuperopResetChannel | ResetChannel | CycleChannel
)
"""Any noise channel that :meth:`NoiseModelLike.get_channel` may return."""

NoiseInstruction: TypeAlias = Gate | Measurement | ResetQubit
"""Any instruction a noise channel may be attached to."""


@runtime_checkable
class NoiseModelLike(Protocol):
    """Protocol defining the noise model interface.

    Any object that implements ``get_channel`` with the correct signature can be
    used wherever a noise model is expected. This enables alternative noise model
    strategies (depolarizing, dynamic, crosstalk-aware) without modifying
    consumers.

    The standard concrete implementation is :class:`NoiseModel`.

    .. note::
        This protocol is ``runtime_checkable``, so ``isinstance`` checks only verify that a
        ``get_channel`` *attribute* exists — not that its signature matches. Do not rely on
        ``isinstance`` to validate a candidate noise model.
    """

    @overload
    def get_channel(self, inst: Gate) -> SuperopChannel | Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> SuperopResetChannel | ResetChannel | None: ...

    def get_channel(self, inst: Gate | Measurement | ResetQubit) -> ChannelType | None:
        """Retrieve the noise channel for a specific instruction.

        :param inst: A gate, measurement, or reset instruction.
        :return: The associated noise channel, or ``None`` if the instruction
            should be treated as ideal (noiseless).
        """
        ...


@dataclass(frozen=True)
class NoiseModel:
    """A noise model collects all the noise channels for a given quantum program.

    This includes gate channels, measurement channels, reset channels, and cycle channels.

    The constructor accepts a mapping from instruction to channel. Use
    :meth:`from_channels` when constructing from a list, tuple, set, generator,
    or other channel iterable.
    """

    channels: Mapping[NoiseInstruction, ChannelType] = field(default_factory=dict)
    """Immutable mapping from instruction to its noise channel.

    Wrapped in a :class:`types.MappingProxyType` by ``__post_init__``, so the mapping passed in
    cannot be mutated through the model afterwards. (``MappingProxyType`` is used in preference to
    a third-party frozen-dict type to keep pyQuil's dependency surface unchanged.)
    """

    def __post_init__(self) -> None:
        """Validate that every key matches its channel's instruction, and freeze the mapping."""
        channels = self.channels
        if not isinstance(channels, Mapping):
            raise TypeError("NoiseModel channels must be a mapping. Use NoiseModel.from_channels(...) for iterables.")

        for inst, channel in channels.items():
            if channel.inst != inst:
                raise ValueError(
                    f"NoiseModel channel key {inst!r} does not match channel instruction {channel.inst!r}."
                )
        object.__setattr__(self, "channels", MappingProxyType(dict(channels)))

    def __getstate__(self) -> dict[str, dict[NoiseInstruction, ChannelType]]:
        # ``channels`` is a MappingProxyType (unpicklable); serialize it as a plain dict.
        return {"channels": dict(self.channels)}

    def __setstate__(self, state: Mapping[str, Mapping[NoiseInstruction, ChannelType]]) -> None:
        object.__setattr__(self, "channels", MappingProxyType(dict(state["channels"])))

    @overload
    def get_channel(self, inst: Gate) -> SuperopChannel | Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> SuperopResetChannel | ResetChannel | None: ...

    def get_channel(self, inst: Gate | Measurement | ResetQubit) -> ChannelType | None:
        """Retrieve the noise channel associated with a specific instruction.

        :param inst: The instruction (gate, measurement, or reset) for which to retrieve the noise channel.
        :return: The noise channel associated with the instruction, or None if no channel is found.
        """
        return self.channels.get(inst)

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_channels(cls: type[NoiseModel], channels: Iterable[ChannelType] = ()) -> NoiseModel:
        """Create a noise model from an iterable of channels.

        :param channels: Noise channels to include in the model.
        :return: A NoiseModel keyed by each channel's instruction.
        :raises ValueError: If more than one channel targets the same instruction.
        """
        channel_map: dict[NoiseInstruction, ChannelType] = {}
        for channel in channels:
            if channel.inst in channel_map:
                raise ValueError(f"Duplicate noise channel for instruction {channel.inst!r}.")
            channel_map[channel.inst] = channel
        return cls(channels=channel_map)

    @classmethod
    def from_isa(cls: type[NoiseModel], isa: InstructionSetArchitecture) -> NoiseModel:
        """Create a noise model from a QCS ``InstructionSetArchitecture``.

        This is the preferred ISA entry point: it takes the ``qcs_sdk`` ISA returned by
        :func:`qcs_sdk.qpu.isa.get_instruction_set_architecture`, rather than the legacy
        rpcq-derived :class:`~pyquil.external.rpcq.CompilerISA`.

        :param isa: The QCS API ``InstructionSetArchitecture``.
        :return: A NoiseModel with channels according to the ISA's fidelities.

        .. seealso:: :meth:`from_compiler_isa` for the channel-construction details and caveats.
        """
        from pyquil.quantum_processor.transformers import qcs_isa_to_compiler_isa

        return cls.from_compiler_isa(qcs_isa_to_compiler_isa(isa))

    @classmethod
    @deprecated(
        version="4.17.0",
        reason="pyquil.external.rpcq (and CompilerISA) will be removed in pyQuil v5. "
        "Use NoiseModel.from_isa with a qcs_sdk InstructionSetArchitecture instead.",
    )
    def from_compiler_isa(cls: type[NoiseModel], compiler_isa: CompilerISA) -> NoiseModel:
        """Create a noise model from an rpcq-derived compiler ISA.

        Gate fidelities are converted to depolarizing channels and measurement
        errors are symmetric. Only gates with concrete numeric parameters and a
        fidelity below 1.0 are included.

        .. deprecated:: 4.17.0
            :class:`~pyquil.external.rpcq.CompilerISA` is part of the rpcq layer slated for
            removal in pyQuil v5. Use :meth:`from_isa` with a ``qcs_sdk``
            ``InstructionSetArchitecture``.

        .. note::
            Two-qubit gate channels are keyed by the ISA edge's operand order
            (e.g. ``CZ 0 1``). A program that issues the gate with the operands
            reversed (``CZ 1 0``) will not match this channel, even for symmetric
            gates. Issue gates in the ISA operand order, or build channels
            explicitly for both orderings.

        :param compiler_isa: The compiler ISA.
        :return: A NoiseModel with channels according to the provided fidelities.
        """
        from pyquil.external.rpcq import GateInfo, MeasureInfo
        from pyquil.quilatom import Qubit as QuilQubit

        channels: dict[NoiseInstruction, ChannelType] = {}
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
                    # Skip gates with non-numeric parameters. ``complex`` is excluded
                    # deliberately: gate parameters must be real, and float(complex) raises.
                    if not all(isinstance(p, (float, int)) for p in params):
                        continue

                    numeric_params: list[float] = [float(p) for p in params if isinstance(p, (float, int))]
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
                    if fidelity is None:
                        # Don't mark the qubit seen on a fidelity-less entry; a later
                        # MeasureInfo for the same qubit may carry a usable fidelity.
                        continue
                    seen_measure_qubits.add(qubit_idx)
                    # Matching the gate branch, a perfect readout gets no channel at all rather
                    # than an identity one.
                    if fidelity >= 1.0:
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
                    if not all(isinstance(p, (float, int)) for p in params):
                        continue

                    numeric_params = [float(p) for p in params if isinstance(p, (float, int))]
                    inst = Gate(name=gate_name, params=numeric_params, qubits=qubits)
                    if fidelity is not None and fidelity < 1.0:
                        channels[inst] = Channel.from_gate_fidelity(inst=inst, fidelity=fidelity)

        return cls(channels=channels)

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize NoiseModel to a JSON string.

        :return: JSON string representation.
        """
        # Every member of ChannelType derives from _ChannelBase or _ResetChannelBase, or is a
        # MeasurementChannel or CycleChannel — all of which implement to_json — so there is no
        # unsupported case to skip here.
        channel_data = [{"type": type(ch).__name__, "data": ch.to_json()} for ch in self.channels.values()]
        return json.dumps({"channels": channel_data})

    @classmethod
    def from_json(cls: type[NoiseModel], json_str: str) -> NoiseModel:
        """Deserialize a NoiseModel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: NoiseModel instance.
        """
        data = json.loads(json_str)
        _type_map = {
            "SuperopChannel": SuperopChannel,
            "Channel": Channel,
            "MeasurementChannel": MeasurementChannel,
            "SuperopResetChannel": SuperopResetChannel,
            "ResetChannel": ResetChannel,
            "CycleChannel": CycleChannel,
        }
        channels: list[ChannelType] = []
        for ch_data in data["channels"]:
            ch_cls = _type_map.get(ch_data["type"])
            if ch_cls is None:
                raise ValueError(f"Unknown channel type: {ch_data['type']}")
            channels.append(ch_cls.from_json(ch_data["data"]))  # type: ignore[attr-defined]
        return cls.from_channels(channels)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        """Check if two NoiseModels contain equivalent channel maps."""
        if not isinstance(other, NoiseModel):
            return False
        return dict(self.channels) == dict(other.channels)

    # Unhashable: its channels hold jax arrays and are themselves unhashable, and an
    # ``id``-based hash would be inconsistent with the value-based ``__eq__``.
    __hash__ = None  # type: ignore[assignment]

    def __add__(self, other: NoiseModel) -> NoiseModel:
        """Combine two NoiseModels into their disjoint union.

        The two models must not both define a channel for the same instruction.
        Addition is a union of channels, not a composition: overlapping channels are
        a conflict, not an "addition". Compose channels explicitly
        (``channel_a @ channel_b``) if that is what you intend.

        :raises ValueError: If both models define a channel for the same instruction.
        """
        if not isinstance(other, NoiseModel):
            return NotImplemented

        overlap = set(self.channels) & set(other.channels)
        if overlap:
            insts = ", ".join(repr(inst) for inst in overlap)
            raise ValueError(
                f"Cannot add NoiseModels: both define a channel for the same instruction(s): {insts}. "
                "Addition is a disjoint union; compose the channels explicitly if that is intended."
            )

        return NoiseModel(channels={**self.channels, **other.channels})

    def with_channels(self, channels: Iterable[ChannelType]) -> NoiseModel:
        """Return a new model with additional channels.

        :param channels: New channels to add.
        :return: A NoiseModel containing the existing and new channels.
        :raises ValueError: If a new channel targets an instruction already in the model.
        """
        combined = dict(self.channels)
        for channel in channels:
            if channel.inst in combined:
                raise ValueError(f"Duplicate noise channel for instruction {channel.inst!r}.")
            combined[channel.inst] = channel
        return NoiseModel(channels=combined)


# ──────────────────────────────────────────────────────────
# Convenience NoiseModelLike implementations
# ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DepolarizingNoiseModel:
    r"""A noise model that applies uniform depolarizing noise to every gate.

    For any ``Gate`` instruction, returns a :class:`Channel` with the specified
    depolarizing constant.  Measurements and resets are treated as ideal.

    :param depolarizing_constant: The depolarization constant :math:`p` where
        :math:`\mathcal{D}_p(\rho) = p \, \rho + (1-p) \, I/d`.
        A value of 1.0 means no noise; 0.0 means full depolarization.
    """

    depolarizing_constant: float

    @overload
    def get_channel(self, inst: Gate) -> SuperopChannel | Channel | CycleChannel | None: ...

    @overload
    def get_channel(self, inst: Measurement) -> MeasurementChannel | None: ...

    @overload
    def get_channel(self, inst: ResetQubit) -> SuperopResetChannel | ResetChannel | None: ...

    def get_channel(self, inst: Gate | Measurement | ResetQubit) -> ChannelType | None:
        """Return a depolarizing channel for gates; ``None`` for measurements/resets."""
        if isinstance(inst, Gate):
            return _depolarizing_channel(inst, self.depolarizing_constant)
        return None


@lru_cache(maxsize=1024)
def _depolarizing_channel(inst: Gate, depolarizing_constant: float) -> Channel:
    """Build (and memoize) the depolarizing channel for one gate.

    :class:`DepolarizingNoiseModel` synthesizes channels on demand, and a resolver walking a
    program will ask for the same handful of gates thousands of times. Building the channel
    involves resolving the gate unitary and deriving its Hamiltonian, so the results are cached
    on the (hashable) instruction and constant.
    """
    return Channel.from_depolarizing_constant(inst, depolarizing_constant)


# ──────────────────────────────────────────────────────────
# Program-level fidelity estimation
# ──────────────────────────────────────────────────────────


def estimate_program_fidelity(program: Program, noise_model: NoiseModelLike) -> float:
    """Estimate the program fidelity for a given noise model.

    Works by multiplying the gate process fidelities together.

    .. note::
        Readout and reset noise are **not** considered: measurement and reset channels are
        skipped, as is the readout portion of any :class:`~pyquil.noise._channels.CycleChannel`.
        The result describes the coherent part of the program only.

    :param program: The program of interest.
    :param noise_model: A noise model.
    :return: The estimated process fidelity.
    """
    gate_fidelities = [1.0]
    for inst in program.instructions:
        if isinstance(inst, Gate):
            channel = noise_model.get_channel(inst)
            # Gate channels (SuperopChannel, Channel) and cycle channels both expose a
            # process (Pauli) fidelity; treat all gate-based channels the same.
            if isinstance(channel, (_ChannelBase, CycleChannel)):
                gate_fidelities.append(channel.process_fidelity)

    return math.prod(gate_fidelities)


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
        inst_qubits = {q.index for q in inst.qubits}  # type: ignore[union-attr]
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
    observable: PauliSum | PauliTerm,
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

    qubits = [int(q) for term in observable.terms for q, _ in term.operations_as_set()]  # type: ignore[arg-type]
    reduced_program = _light_cone_program(program, qubits)
    return estimate_program_fidelity(reduced_program, noise_model)
