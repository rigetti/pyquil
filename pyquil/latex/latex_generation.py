##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
# THIS FILE IS DERIVED AND MODIFIED FROM PROJECTQ. COPYRIGHT PROVIDED HERE:
#
#   Copyright 2017 ProjectQ-Framework (www.projectq.ch)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##############################################################################

import sys
from copy import copy
from warnings import warn

from pyquil import Program
from pyquil.quil import Measurement, Gate, Pragma
from pyquil.quilatom import format_parameter
from pyquil.quilbase import AbstractInstruction
from collections import defaultdict
from typing import Optional

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass, replace
else:
    from dataclasses import dataclass, replace

# Overview of LaTeX generation.
#
# The main entry point is the `to_latex` function below. Here are some high
# points of the generation procedure:
#
# - The most basic building block are the TikZ operators, which are constructed
#   by the functions below (e.g. TIKZ_CONTROL, TIKZ_NOP, TIKZ_MEASURE).
# - TikZ operators are maintained by a DiagramState object, with roughly each
#   qubit line in a diagram represented as a list of TikZ operators on the DiagramState.
# - The DiagramBuilder is the actual driver. This traverses a Program and, for
#   each instruction, performs a suitable manipulation of the DiagramState. At
#   the end of this, the DiagramState is traversed and raw LaTeX is emitted.
# - Most options are specified by DiagramSettings. One exception is this: it is possible
#   to request that a certain subset of the program is rendered as a group (and colored
#   as such). This is specified by a new pragma in the Program source:
#
#     PRAGMA LATEX_GATE_GROUP <name>?
#     ...
#     PRAGMA END_LATEX_GATE_GROUP
#
#   The <name> is optional, and will be used to label the group. Nested gate
#   groups are currently not supported.


@dataclass
class DiagramSettings:
    texify_numerical_constants: bool = True
    """
    Numerical constants (e.g. Pi) will be converted to pretty latex form.
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
    Write controlled rotations such as `RX(pi)` as `X_{\\pi}`, instead of the longer `R_X(\\pi)`
    """

    qubit_line_open_wire_length: int = 1
    """
    The length by which qubit lines should be extended, at the right of the diagram, with open wires.

    The default of 1 is the natural choice. The main reason for including this option is that it may
    be appropriate for this to be 0 in subdiagrams.
    """


def to_latex(circuit: Program, settings: Optional[DiagramSettings] = None) -> str:
    """
    Translates a given pyquil Program to a TikZ picture in a LaTeX document.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param DiagramSettings settings: An optional object of settings controlling diagram rendering and layout.
    :return: LaTeX document string which can be compiled.
    :rtype: string
    """
    if settings is None:
        settings = DiagramSettings()
    text = header(settings)
    text += "\n"
    text += body(circuit, settings)
    text += "\n"
    text += footer()
    return text


def header(settings):
    """
    Writes the LaTeX header using the settings file.

    The header includes all packages and defines all tikz styles.

    :param dictionary settings: LaTeX settings for document.
    :return: Header of the LaTeX document.
    :rtype: string
    """
    packages = (r"\documentclass[convert={density=300,outext=.png}]{standalone}",
                r"\usepackage[margin=1in]{geometry}",
                r"\usepackage{tikz}",
                r"\usetikzlibrary{quantikz}")

    init = (r"\begin{document}",
            r"\begin{tikzcd}")

    return "\n".join(("\n".join(packages), "\n".join(init)))


def footer():
    """
    Return the footer of the LaTeX document.

    :return: LaTeX document footer.
    :rtype: string
    """
    return "\\end{tikzcd}\n\\end{document}"


def body(circuit, settings):
    """
    Return the body of the LaTeX document, including the entire circuit in
    TikZ format.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings:

    :return: LaTeX string to draw the entire circuit.
    :rtype: string
    """

    diagram = DiagramBuilder(circuit, settings).build()

    # flush lines
    quantikz_out = []
    for qubit in diagram.qubits:
        quantikz_out.append(" & ".join(diagram.lines[qubit]))

    return " \\\\\n".join(quantikz_out)


# Constants


PRAGMA_BEGIN_GROUP = 'LATEX_GATE_GROUP'
PRAGMA_END_GROUP = 'END_LATEX_GATE_GROUP'

# TikZ operators


def TIKZ_LEFT_KET(qubit):
    return r"\lstick{{\ket{{q_{{{qubit}}}}}}}".format(qubit=qubit)


def TIKZ_CONTROL(control, offset):
    return r"\ctrl{{{offset}}}".format(offset=offset)


def TIKZ_CNOT_TARGET():
    return r"\targ{}"


def TIKZ_CPHASE_TARGET():
    return r"\control{}"


def TIKZ_SWAP(source, offset):
    return r"\swap{{{offset}}}".format(offset=offset)


def TIKZ_SWAP_TARGET():
    return r"\targX{}"


def TIKZ_NOP():
    return r"\qw"


def TIKZ_MEASURE():
    return r"\meter{}"


def _format_parameter(param, settings=None):
    formatted = format_parameter(param)
    if settings and settings.texify_numerical_constants:
        formatted = formatted.replace("pi", r"\pi")
    return formatted


def _format_parameters(params, settings=None):
    return "(" + ",".join(_format_parameter(param, settings) for param in params) + ")"


def TIKZ_GATE(name, size=1, params=None, dagger=False, settings=None):
    cmd = r"\gate"
    if settings and settings.abbreviate_controlled_rotations and name in ["RX", "RY", "RZ"]:
        name = name[1] + "_{{{param}}}".format(param=_format_parameter(params[0], settings))
        return cmd + "{{{name}}}".format(name=name)
    # now, handle the general case
    if size > 1:
        cmd += "[wires={size}]".format(size=size)
    # TeXify names
    if name in ["RX", "RY", "RZ"]:
        name = name[0] + "_" + name[1].lower()
    if dagger:
        name += r"^{\dagger}"
    if params:
        name += _format_parameters(params, settings)
    return cmd + "{{{name}}}".format(name=name)


def TIKZ_GATE_GROUP(qubits, width, label):
    num_qubits = max(qubits) - min(qubits) + 1
    return "\\gategroup[{qubits},steps={width},style={{dashed, rounded corners,fill=blue!20, inner xsep=2pt}}, background]{{{label}}}".format(
        qubits=num_qubits, width=width, label=label)


SOURCE_TARGET_OP = {
    "CNOT": (TIKZ_CONTROL, TIKZ_CNOT_TARGET),
    "SWAP": (TIKZ_SWAP, TIKZ_SWAP_TARGET),
    "CZ": (TIKZ_CONTROL, lambda: TIKZ_GATE("Z")),
    "CPHASE": (TIKZ_CONTROL, TIKZ_CPHASE_TARGET),
}


# DiagramState


class DiagramState:
    """
    A representation of a circuit diagram.

    This maintains an ordered list of qubits, and for each qubit a 'line': that is, a list of TikZ operators.
    """
    def __init__(self, qubits):
        self.qubits = qubits
        self.lines = defaultdict(list)

    def extend_lines_to_common_edge(self, qubits, offset=0):
        """
        Add NOP operations on the lines associated with the given qubits, until
        all lines are of the same width.
        """
        max_width = max(self.width(q) for q in qubits) + offset
        for q in qubits:
            while self.width(q) < max_width:
                self.append(q, TIKZ_NOP())

    def width(self, qubit):
        """
        The width of the diagram, in terms of the number of operations, on the
        specified qubit line.
        """
        return len(self.lines[qubit])

    def append(self, qubit, op):
        """
        Add an operation to the rightmost edge of the specified qubit line.
        """
        self.lines[qubit].append(op)

    def append_diagram(self, diagram, group=None):
        """
        Add all operations represented by the given diagram to their
        corresponding qubit lines in this diagram.

        If group is not None, then a TIKZ_GATE_GROUP is created with the label indicated by group.
        """
        grouped_qubits = diagram.qubits
        diagram.extend_lines_to_common_edge(grouped_qubits)
        # NOTE: this may create new lines, no big deal
        self.extend_lines_to_common_edge(grouped_qubits)

        # record info for later (optional) group placement
        # the group is marked with a rectangle. we compute the upper-left corner and the width of the rectangle
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

    def interval(self, low, high):
        """
        All qubits in the diagram, from low to high, inclusive.
        """
        full_interval = range(low, high + 1)
        qubits = list(set(full_interval) & set(self.qubits))
        return sorted(qubits)

    def is_interval(self, qubits):
        """
        Do the specified qubits correspond to an interval in this diagram?
        """
        return qubits == self.interval(min(qubits), max(qubits))


class DiagramBuilder:
    """
    Constructs DiagramStates from a given circuit and settings.

    This is essentially a state machine, represented by a few instance variables and some mutually
    recursive methods.
    """
    def __init__(self, circuit, settings):
        self.circuit = circuit
        self.settings = settings
        # partially constructed diagram
        self.diagram = None
        # index into circuit. we maintain the invariant that circuit[0:index]
        # has been processed, with the diagram updated accordingly
        self.index = 0

    def build(self):
        """
        Actually build the diagram.
        """
        qubits = self.circuit.get_qubits()
        all_qubits = range(min(qubits), max(qubits) + 1) if self.settings.impute_missing_qubits else sorted(qubits)
        self.diagram = DiagramState(all_qubits)
        self.index = 0

        if self.settings.label_qubit_lines:
            for qubit in self.diagram.qubits:
                self.diagram.append(qubit, TIKZ_LEFT_KET(qubit))
        else:  # initial exposed wires
            self.diagram.extend_lines_to_common_edge(self.diagram.qubits, offset=1)

        while self.index < len(self.circuit):
            instr = self.circuit[self.index]
            if isinstance(instr, Pragma) and instr.command == PRAGMA_BEGIN_GROUP:
                self._build_group()
            elif isinstance(instr, Pragma) and instr.command == PRAGMA_END_GROUP:
                raise ValueError("PRAGMA {} found without matching {}.".format(PRAGMA_END_GROUP, PRAGMA_BEGIN_GROUP))
            elif isinstance(instr, Measurement):
                self._build_measure()
            elif isinstance(instr, Gate):
                if 'FORKED' in instr.modifiers:
                    raise ValueError("LaTeX output does not currently support FORKED modifiers: {}.".format(instr))
                # the easy case is 1q operations
                if len(instr.qubits) == 1:
                    self._build_1q_unitary()
                else:
                    if instr.name in SOURCE_TARGET_OP and not instr.modifiers:
                        self._build_custom_source_target_op()
                    else:
                        self._build_generic_unitary()
            else:
                self.index += 1

        self.diagram.extend_lines_to_common_edge(self.diagram.qubits,
                                                 offset=max(self.settings.qubit_line_open_wire_length, 0))
        return self.diagram

    def _build_group(self):
        """
        Update the partial diagram with the subcircuit delimited by the grouping PRAGMA.

        Advances the index beyond the ending pragma.
        """
        instr = self.circuit[self.index]
        if len(instr.args) != 0:
            raise ValueError("PRAGMA {} expected a freeform string, or nothing at all.".format(PRAGMA_BEGIN_GROUP))
        start = self.index + 1
        # walk instructions until the group end
        for j in range(start, len(self.circuit)):
            if isinstance(self.circuit[j], Pragma) and self.circuit[j].command == PRAGMA_END_GROUP:
                # recursively build the diagram for this block
                # we do not want labels here!
                block_settings = replace(self.settings,
                                         label_qubit_lines=False,
                                         qubit_line_open_wire_length=0)
                block = DiagramBuilder(self.circuit[start:j], block_settings).build()
                block_name = instr.freeform_string if instr.freeform_string else ""
                self.diagram.append_diagram(block, group=block_name)
                # advance to the instruction following this one
                self.index = j + 1
                return

        raise ValueError("Unable to find PRAGMA {} matching {}.".format(PRAGMA_END_GROUP, instr))

    def _build_measure(self):
        """
        Update the partial diagram with a measurement operation.

        Advances the index by one.
        """
        instr = self.circuit[self.index]
        self.diagram.append(instr.qubit.index, TIKZ_MEASURE())
        self.index += 1

    def _build_custom_source_target_op(self):
        """
        Update the partial diagram with a single operation involving a source and a target (e.g. a controlled gate, a swap).

        Advances the index by one.
        """
        instr = self.circuit[self.index]
        source, target = qubit_indices(instr)
        displaced = self.diagram.interval(min(source, target), max(source, target))
        self.diagram.extend_lines_to_common_edge(displaced)
        source_op, target_op = SOURCE_TARGET_OP[instr.name]
        offset = (-1 if source > target else 1) * (len(displaced) - 1)  # this is a directed quantity
        self.diagram.append(source, source_op(source, offset))
        self.diagram.append(target, target_op())
        self.diagram.extend_lines_to_common_edge(displaced)
        self.index += 1

    def _build_1q_unitary(self):
        """
        Update the partial diagram with a 1Q gate.

        Advances the index by one.
        """
        instr = self.circuit[self.index]
        qubits = qubit_indices(instr)
        dagger = sum(m == 'DAGGER' for m in instr.modifiers) % 2 == 1
        self.diagram.append(qubits[0], TIKZ_GATE(instr.name, params=instr.params, dagger=dagger, settings=self.settings))
        self.index += 1

    def _build_generic_unitary(self):
        """
        Update the partial diagram with a unitary operation.

        Advances the index by one.
        """
        instr = self.circuit[self.index]
        qubits = qubit_indices(instr)
        dagger = sum(m == 'DAGGER' for m in instr.modifiers) % 2 == 1
        controls = sum(m == 'CONTROLLED' for m in instr.modifiers)

        self.diagram.extend_lines_to_common_edge(qubits)

        control_qubits = qubits[:controls]
        target_qubits = qubits[controls:]
        if not self.diagram.is_interval(sorted(target_qubits)):
            raise ValueError("Unable to render instruction {} which targets non-adjacent qubits.".format(instr))

        for q in control_qubits:
            self.diagram.append(q, TIKZ_CONTROL(q, target_qubits[0]))

        # we put the gate on the first target line, and nop on the others
        self.diagram.append(target_qubits[0], TIKZ_GATE(instr.name, size=len(qubits), params=instr.params, dagger=dagger))
        for q in target_qubits[1:]:
            self.diagram.append(q, TIKZ_NOP())

        self.index += 1


def qubit_indices(instr: AbstractInstruction):
    """
    Get a list of indices associated with the given instruction.
    """
    if isinstance(instr, Measurement):
        return [instr.qubit.index]
    elif isinstance(instr, Gate):
        return [qubit.index for qubit in instr.qubits]
    else:
        return []
