##############################################################################
# Copyright 2019 Rigetti Computing
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

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Optional, cast
from warnings import warn

from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
    AbstractInstruction,
    ArithmeticBinaryOp,
    ClassicalComparison,
    ClassicalConvert,
    ClassicalExchange,
    ClassicalLoad,
    ClassicalMove,
    ClassicalStore,
    Gate,
    Jump,
    JumpUnless,
    JumpWhen,
    LogicalBinaryOp,
    Measurement,
    Pragma,
    ResetQubit,
    UnaryClassicalInstruction,
    Wait,
)


@dataclass
class DiagramSettings:
    """Settings to control the layout and rendering of circuits."""

    texify_numerical_constants: bool = True
    """
    Convert numerical constants, such as pi, to LaTeX form.
    """

    impute_missing_qubits: bool = False
    """
    Include qubits with indices between those explicitly referenced in the Quil program.

    For example, if true, the diagram for `CNOT 0 2` would have three qubit lines: 0, 1, 2.
    """

    label_qubit_lines: bool = True
    """
    Label qubit lines.
    """

    abbreviate_controlled_rotations: bool = False
    """
    Write controlled rotations in a compact form.

    For example,  `RX(pi)` as `X_{\\pi}`, instead of the longer `R_X(\\pi)`
    """

    qubit_line_open_wire_length: int = 1
    """
    The length by which qubit lines should be extended with open wires at the right of the diagram.

    The default of 1 is the natural choice. The main reason for including this option
    is that it may be appropriate for this to be 0 in subdiagrams.
    """

    right_align_terminal_measurements: bool = True
    """
    Align measurement operations which appear at the end of the program.
    """


# Constants


PRAGMA_BEGIN_GROUP = "LATEX_GATE_GROUP"
PRAGMA_END_GROUP = "END_LATEX_GATE_GROUP"

UNSUPPORTED_INSTRUCTION_CLASSES = (
    Wait,
    JumpWhen,
    JumpUnless,
    Jump,
    UnaryClassicalInstruction,
    LogicalBinaryOp,
    ArithmeticBinaryOp,
    ClassicalMove,
    ClassicalExchange,
    ClassicalConvert,
    ClassicalLoad,
    ClassicalStore,
    ClassicalComparison,
)


# TikZ operators


def TIKZ_LEFT_KET(qubit: int) -> str:
    return rf"\lstick{{\ket{{q_{{{qubit}}}}}}}"


def TIKZ_CONTROL(control: int, offset: int) -> str:
    return rf"\ctrl{{{offset}}}"


def TIKZ_CNOT_TARGET() -> str:
    return r"\targ{}"


def TIKZ_CPHASE_TARGET() -> str:
    return r"\control{}"


def TIKZ_SWAP(source: int, offset: int) -> str:
    return rf"\swap{{{offset}}}"


def TIKZ_SWAP_TARGET() -> str:
    return r"\targX{}"


def TIKZ_NOP() -> str:
    return r"\qw"


def TIKZ_MEASURE() -> str:
    return r"\meter{}"


def _format_parameter(param: ParameterDesignator, settings: Optional[DiagramSettings] = None) -> str:
    formatted: str = format_parameter(param)
    if settings and settings.texify_numerical_constants:
        formatted = formatted.replace("pi", r"\pi")
    return formatted


def _format_parameters(params: Iterable[ParameterDesignator], settings: Optional[DiagramSettings] = None) -> str:
    return "(" + ",".join(_format_parameter(param, settings) for param in params) + ")"


def TIKZ_GATE(
    name: str,
    size: int = 1,
    params: Optional[Sequence[ParameterDesignator]] = None,
    dagger: bool = False,
    settings: Optional[DiagramSettings] = None,
) -> str:
    cmd = r"\gate"
    rotations = ["RX", "RY", "RZ"]
    if settings and settings.abbreviate_controlled_rotations and name in rotations and params:
        name = name[1] + f"_{{{_format_parameter(params[0], settings)}}}"
        return cmd + f"{{{name}}}"
    # now, handle the general case
    if size > 1:
        cmd += f"[wires={size}]"
    # TeXify names
    if name in ["RX", "RY", "RZ"]:
        name = name[0] + "_" + name[1].lower()
    if dagger:
        name += r"^{\dagger}"
    if params:
        name += _format_parameters(params, settings)
    return cmd + f"{{{name}}}"


def TIKZ_GATE_GROUP(qubits: Sequence[int], width: int, label: str) -> str:
    num_qubits = max(qubits) - min(qubits) + 1
    return (
        f"\\gategroup[{num_qubits},steps={width},style={{dashed, rounded corners,"
        f"fill=blue!20, inner xsep=2pt}}, background]{{{label}}}"
    )


SOURCE_TARGET_OP = {
    "CNOT": (TIKZ_CONTROL, TIKZ_CNOT_TARGET),
    "SWAP": (TIKZ_SWAP, TIKZ_SWAP_TARGET),
    "CZ": (TIKZ_CONTROL, lambda: TIKZ_GATE("Z")),
}


class DiagramState:
    """A representation of a circuit diagram.

    This maintains an ordered list of qubits, and for each qubit a 'line': that is, a list of
    TikZ operators.
    """

    def __init__(self, qubits: Sequence[int]):
        self.qubits = qubits
        self.lines: Mapping[int, list[str]] = defaultdict(list)

    def extend_lines_to_common_edge(self, qubits: Iterable[int], offset: int = 0) -> None:
        """Add NOP operations on the lines associated with the given qubits, until all lines are of the same width."""
        max_width = max(self.width(q) for q in qubits) + offset
        for q in qubits:
            while self.width(q) < max_width:
                self.append(q, TIKZ_NOP())

    def width(self, qubit: int) -> int:
        """Calculate the width of the diagram, in terms of the number of operations, on the specified qubit line."""
        return len(self.lines[qubit])

    def append(self, qubit: int, op: str) -> None:
        """Add an operation to the rightmost edge of the specified qubit line."""
        self.lines[qubit].append(op)

    def append_diagram(self, diagram: "DiagramState", group: Optional[str] = None) -> "DiagramState":
        """Add all operations represented by the given diagram to their corresponding qubit lines in this diagram.

        If group is not None, then a TIKZ_GATE_GROUP is created with the label indicated by group.
        """
        grouped_qubits = diagram.qubits
        diagram.extend_lines_to_common_edge(grouped_qubits)
        # NOTE: this may create new lines, no big deal
        self.extend_lines_to_common_edge(grouped_qubits)

        # record info for later (optional) group placement
        # the group is marked with a rectangle. we compute the upper-left corner and the width of
        # the rectangle
        corner_row = grouped_qubits[0]
        corner_col = len(self.lines[corner_row]) + 1
        group_width = diagram.width(corner_row) - 1

        # append ops to this diagram
        for q in diagram.qubits:
            for op in diagram.lines[q]:
                self.append(q, op)
        # add tikz grouping command
        if group is not None:
            self.lines[corner_row][corner_col] += " " + TIKZ_GATE_GROUP(grouped_qubits, group_width, group)
        return self

    def interval(self, low: int, high: int) -> list[int]:
        """All qubits in the diagram, from low to high, inclusive."""
        full_interval = range(low, high + 1)
        qubits = list(set(full_interval) & set(self.qubits))
        return sorted(qubits)

    def is_interval(self, qubits: Sequence[int]) -> bool:
        """Return True if the specified qubits correspond to an interval in this diagram, False otherwise."""
        return qubits == self.interval(min(qubits), max(qubits))


def split_on_terminal_measures(
    program: Program,
) -> tuple[list[AbstractInstruction], list[AbstractInstruction]]:
    """Split a program into two lists of instructions.

    1. A set of measurement instructions occurring as the final operation on their qubit.
    2. The rest.
    """
    # handle the easy case explicitly (mainly to avoid warning when we can avoid it)
    if not any(isinstance(instr, Measurement) for instr in program.instructions):
        return [], program.instructions

    seen_qubits: set[QubitDesignator] = set()

    measures: list[AbstractInstruction] = []
    remaining: list[AbstractInstruction] = []
    in_group = False
    for instr in reversed(program.instructions):
        if not in_group and isinstance(instr, Measurement) and instr.qubit not in seen_qubits:
            measures.insert(0, instr)
            seen_qubits.add(instr.qubit)
        else:
            remaining.insert(0, instr)
            if isinstance(instr, (Gate, ResetQubit)):
                seen_qubits |= set(instr.get_qubit_indices() or {})
            elif isinstance(instr, Pragma):
                if instr.command == PRAGMA_END_GROUP:
                    warn(
                        "Alignment of terminal MEASURE operations may" "conflict with gate group declaration.",
                        stacklevel=2,
                    )
                    in_group = True
                elif instr.command == PRAGMA_BEGIN_GROUP:
                    in_group = False
    return measures, remaining


class DiagramBuilder:
    """Constructs DiagramStates from a given circuit and settings.

    This is essentially a state machine, represented by a few instance variables and some mutually
    recursive methods.
    """

    def __init__(self, circuit: Program, settings: DiagramSettings):
        self.circuit = circuit
        self.settings = settings
        # instructions currently being processed
        self.working_instructions: Optional[list[AbstractInstruction]] = None
        # index into working instructions. we maintain the invariant that
        # working_instructions[0:index] has been processed, with the diagram
        # updated accordingly
        self.index = 0
        # partially constructed diagram
        self.diagram: Optional[DiagramState] = None

    def build(self) -> DiagramState:
        """Build the diagram."""
        qubits = cast(set[int], self.circuit.get_qubits(indices=True))
        all_qubits = range(min(qubits), max(qubits) + 1) if self.settings.impute_missing_qubits else sorted(qubits)
        self.diagram = DiagramState(all_qubits)

        if self.settings.right_align_terminal_measurements:
            measures, instructions = split_on_terminal_measures(self.circuit)
        else:
            measures, instructions = [], self.circuit.instructions

        # setup the left fringe
        if self.settings.label_qubit_lines:
            for qubit in self.diagram.qubits:
                self.diagram.append(qubit, TIKZ_LEFT_KET(qubit))
        else:  # initial exposed wires
            self.diagram.extend_lines_to_common_edge(self.diagram.qubits, offset=1)

        # setup working state
        self.working_instructions = instructions
        self.index = 0

        # main loop
        while self.index < len(self.working_instructions):
            instr = self.working_instructions[self.index]
            if isinstance(instr, Pragma) and instr.command == PRAGMA_BEGIN_GROUP:
                self._build_group()
            elif isinstance(instr, Pragma) and instr.command == PRAGMA_END_GROUP:
                raise ValueError(f"PRAGMA {PRAGMA_END_GROUP} found without matching {PRAGMA_BEGIN_GROUP}.")
            elif isinstance(instr, Measurement):
                self._build_measure()
            elif isinstance(instr, Gate):
                if "FORKED" in instr.modifiers:
                    raise ValueError("LaTeX output does not currently support" f"FORKED modifiers: {instr}.")
                # the easy case is 1q operations
                if len(instr.qubits) == 1:
                    self._build_1q_unitary()
                else:
                    if instr.name in SOURCE_TARGET_OP and not instr.modifiers:
                        self._build_custom_source_target_op()
                    else:
                        self._build_generic_unitary()
            elif isinstance(instr, UNSUPPORTED_INSTRUCTION_CLASSES):
                raise ValueError("LaTeX output does not currently support" f"the following instruction: {instr.out()}")
            else:
                self.index += 1

        self.diagram.extend_lines_to_common_edge(self.diagram.qubits)

        # handle terminal measurements
        self.index = 0
        self.working_instructions = measures

        for _ in self.working_instructions:
            self._build_measure()

        offset = max(self.settings.qubit_line_open_wire_length, 0)
        self.diagram.extend_lines_to_common_edge(self.diagram.qubits, offset=offset)
        return self.diagram

    def _build_group(self) -> None:
        """Update the partial diagram with the subcircuit delimited by the grouping PRAGMA.

        Advances the index beyond the ending pragma.
        """
        if self.working_instructions is None:
            raise RuntimeError("Internal error: working_instructions is None.")
        instr = self.working_instructions[self.index]
        if not isinstance(instr, Pragma):
            raise RuntimeError("Internal error: expected a PRAGMA instruction.")
        if len(instr.args) != 0:
            raise ValueError(f"PRAGMA {PRAGMA_BEGIN_GROUP} expected a freeform string, or nothing at all.")
        start = self.index + 1
        # walk instructions until the group end
        for j in range(start, len(self.working_instructions)):
            instruction_j = self.working_instructions[j]
            if isinstance(instruction_j, Pragma) and instruction_j.command == PRAGMA_END_GROUP:
                # recursively build the diagram for this block
                # we do not want labels here!
                block_settings = replace(self.settings, label_qubit_lines=False, qubit_line_open_wire_length=0)
                subcircuit = Program(*self.working_instructions[start:j])
                block = DiagramBuilder(subcircuit, block_settings).build()
                block_name = instr.freeform_string if instr.freeform_string else ""
                if self.diagram is None:
                    raise RuntimeError("Internal error: expected diagram to exist.")
                self.diagram.append_diagram(block, group=block_name)
                # advance to the instruction following this one
                self.index = j + 1
                return

        raise ValueError(f"Unable to find PRAGMA {PRAGMA_END_GROUP} matching {instr}.")

    def _build_measure(self) -> None:
        """Update the partial diagram with a measurement operation.

        Advances the index by one.
        """
        if self.working_instructions is None:
            raise RuntimeError("Internal error: working_instructions is None.")
        instr = self.working_instructions[self.index]
        if not isinstance(instr, Measurement):
            raise RuntimeError("Internal error: expected a Measurement instruction.")
        if self.diagram is None:
            raise RuntimeError("Internal error: expected diagram to exist.")
        self.diagram.append(instr.get_qubit_indices().pop(), TIKZ_MEASURE())
        self.index += 1

    def _build_custom_source_target_op(self) -> None:
        """Update the partial diagram with a single operation involving a source and a target (e.g. a controlled gate).

        Advances the index by one.
        """
        if self.working_instructions is None:
            raise RuntimeError("Internal error: working_instructions is None.")
        instr = self.working_instructions[self.index]
        if not isinstance(instr, Gate):
            raise RuntimeError("Internal error: expected a Gate instruction, got ({type(instr)}).")
        source, target = qubit_indices(instr)
        if self.diagram is None:
            raise RuntimeError("Internal error: expected diagram to exist.")
        displaced = self.diagram.interval(min(source, target), max(source, target))
        self.diagram.extend_lines_to_common_edge(displaced)
        source_op, target_op = SOURCE_TARGET_OP[instr.name]
        offset = (-1 if source > target else 1) * (len(displaced) - 1)  # a directed quantity
        self.diagram.append(source, source_op(source, offset))
        self.diagram.append(target, target_op())
        self.diagram.extend_lines_to_common_edge(displaced)
        self.index += 1

    def _build_1q_unitary(self) -> None:
        """Update the partial diagram with a 1Q gate.

        Advances the index by one.
        """
        if self.working_instructions is None:
            raise RuntimeError("Internal error: working_instructions is None.")
        instr = self.working_instructions[self.index]
        if not isinstance(instr, Gate):
            raise RuntimeError("Internal error: expected a Gate instruction, got ({type(instr)}).")
        qubits = qubit_indices(instr)
        dagger = sum(m == "DAGGER" for m in instr.modifiers) % 2 == 1
        if self.diagram is None:
            raise RuntimeError("Internal error: expected diagram to exist.")
        self.diagram.append(
            qubits[0],
            TIKZ_GATE(instr.name, params=instr.params, dagger=dagger, settings=self.settings),
        )
        self.index += 1

    def _build_generic_unitary(self) -> None:
        """Update the partial diagram with a unitary operation.

        Advances the index by one.
        """
        if self.working_instructions is None:
            raise RuntimeError("Internal error: working_instructions is None.")
        instr = self.working_instructions[self.index]
        if not isinstance(instr, Gate):
            raise RuntimeError("Internal error: expected a Gate instruction, got ({type(instr)}).")
        qubits = qubit_indices(instr)
        dagger = sum(m == "DAGGER" for m in instr.modifiers) % 2 == 1
        controls = sum(m == "CONTROLLED" for m in instr.modifiers)
        if self.diagram is None:
            raise RuntimeError("Internal error: expected diagram to exist.")
        self.diagram.extend_lines_to_common_edge(qubits)
        control_qubits = qubits[:controls]
        # sort the target qubit list because the first qubit indicates wire placement on the diagram
        target_qubits = sorted(qubits[controls:])
        if not self.diagram.is_interval(target_qubits):
            raise ValueError(f"Unable to render instruction {instr} which targets non-adjacent qubits.")

        for q in control_qubits:
            offset = target_qubits[0] - q
            self.diagram.append(q, TIKZ_CONTROL(q, offset))

        # we put the gate on the first target line, and nop on the others
        self.diagram.append(
            target_qubits[0],
            TIKZ_GATE(instr.name, size=len(target_qubits), params=instr.params, dagger=dagger),
        )
        for q in target_qubits[1:]:
            self.diagram.append(q, TIKZ_NOP())

        self.index += 1


def qubit_indices(instr: AbstractInstruction) -> list[int]:
    """Get a list of indices associated with the given instruction."""
    if isinstance(instr, (Measurement, Gate)):
        return list(instr.get_qubit_indices())
    else:
        return []
