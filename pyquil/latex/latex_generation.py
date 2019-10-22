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


from copy import copy
from warnings import warn

from pyquil import Program
from pyquil.quil import Measurement, Gate, Pragma
from pyquil.quilatom import format_parameter
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

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
    Write controlled rotations such as `RX(pi)` as `X_{\pi}`, instead of the longer `R_X(\pi)`
    """

def to_latex(circuit: Program, settings: Optional[DiagramSettings]=None):
    """
    Translates a given pyquil Program to a TikZ picture in a Latex document.

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
    Writes the Latex header using the settings file.

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

    # TODO: set styles?

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
    Return the body of the Latex document, including the entire circuit in
    TikZ format.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings:

    :return: Latex string to draw the entire circuit.
    :rtype: string
    """


    diagram = DiagramBuilder(circuit, settings).build()

    # flush lines
    quantikz_out = []
    for qubit in diagram.qubits:
        quantikz_out.append(" & ".join(diagram.lines[qubit]))

    return " \\\\\n".join(quantikz_out)

### Constants ###

PRAGMA_BEGIN_GROUP = 'LATEX_GATE_GROUP'
PRAGMA_END_GROUP = 'END_LATEX_GATE_GROUP'

#### TikZ operators ###

def TIKZ_LEFT_KET(qubit):
    return r"\lstick{{\ket{{q_{{{qubit}}}}}}}".format(qubit=qubit)

def TIKZ_CONTROL(control, offset):
    return r"\ctrl{{{offset}}}".format(offset=offset)

def TIKZ_CNOT_TARGET():
    return r"\targ{}"

def TIKZ_CPHASE_TARGET():
    return r"\control{}"

def TIKZ_SWAP(source, offset):
    return "\swap{{{offset}}}".format(offset=offset)

def TIKZ_SWAP_TARGET():
    return r"\targX{}"

def TIKZ_NOP():
    return r"\qw"

def TIKZ_MEASURE():
    return r"\meter{}"

def _format_parameter(param, settings=None):
    formatted = format_parameter(param)
    if settings and settings.texify_numerical_constants:
        # TODO we should do a better job than just dumb str.replace
        formatted = formatted.replace("pi", "\pi")
    return formatted

def _format_parameters(params, settings=None):
    return "(" + ",".join(_format_parameter(param, settings) for param in params) + ")"

def TIKZ_GATE(name, size=1, params=None, dagger=False, settings=None):
    cmd = r"\gate"
    if settings and settings.abbreviate_controlled_rotations and name in ["RX", "RY", "RZ"]:
        name = name[1] + "_{{{param}}}".format(param=_format_parameter(params[0]))
        return cmd + "{{{name}}}".format(name=name)
    # now, handle the general case
    if size > 1:
        cmd += "[wires={size}]".format(size=size)
    # TeXify names
    if name in ["RX", "RY", "RZ"]:
        name = name[0] + "_" + name[1].lower()
    if dagger:
        name += "^{\dagger}"
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
    "CPHASE": (TIKZ_CONTROL, TIKZ_CPHASE_TARGET)
}

def qubit_indices(instr):
    if isinstance(instr, Measurement):
        return [instr.qubit.index]
    elif isinstance(instr, Gate):
        return [qubit.index for qubit in instr.qubits]
    else:
        return []

### DiagramState

class DiagramState:
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

    def append_diagram(self, diagram, grouped=True):
        """
        Add all operations represented by the given diagram to their
        corresponding qubit lines in this diagram.
        """
        grouped_qubits = diagram.qubits
        diagram.extend_lines_to_common_edge(grouped_qubits)
        # NOTE: this may create new lines, no big deal
        self.extend_lines_to_common_edge(grouped_qubits)
        # add tikz grouping command
        if grouped:
            q = grouped_qubits[0]
            self.lines[q][-1] += " " + TIKZ_GATE_GROUP(grouped_qubits, diagram.width(q), "")
        # append ops to this diagram
        for q in diagram.qubits:
            for op in diagram.lines[q]:
                self.append(q, op)
        return self

    def interval(self, low, high):
        """
        All qubits in the diagram, from low to high, inclusive.
        """
        full_interval = range(low, high+1)
        qubits = list(set(full_interval) & set(self.qubits))
        return sorted(qubits)

    def is_interval(self, qubits):
        return qubits == self.interval(min(qubits), max(qubits))

class DiagramBuilder:
    def __init__(self, circuit, settings):
        self.circuit = circuit
        self.settings = settings
        self.diagram = None
        self.index = 0

    def build(self):
        qubits = self.circuit.get_qubits()
        all_qubits = range(min(qubits), max(qubits) + 1) if self.settings.impute_missing_qubits else sorted(qubits)
        self.diagram = DiagramState(all_qubits)
        self.index = 0

        if self.settings.label_qubit_lines:
            for qubit in self.diagram.qubits:
                self.diagram.append(qubit, TIKZ_LEFT_KET(qubit))
        else: # initial exposed wires
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
                    raise ValueError("LaTeX output does not currently support FORKED modifiers: {}".format(instr))
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

        self.diagram.extend_lines_to_common_edge(self.diagram.qubits, offset=1)
        return self.diagram

    def _build_group(self):
        instr = self.circuit[self.index]
        if len(instr.args) > 1:
            raise ValueError("PRAGMA {} expected exactly one argument.".format(PRAGMA_BEGIN_GROUP))
        for j in range(self.index+1, len(self.circuit)):
            if isinstance(self.circuit[j], Pragma) and self.circuit[j].command == PRAGMA_END_GROUP:
                # recursively build the diagram for this block
                block = DiagramBuilder(self.circuit[self.index+1:j]).build()
                self.diagram.append_diagram(block)
                # advance to the instruction following this one
                self.index = j+1
                return

        raise ValueError("Unable to find PRAGMA {} matching {}".format(PRAGMA_END_GROUP, instr))

    def _build_measure(self):
        instr = self.circuit[self.index]
        self.diagram.append(instr.qubit.index, TIKZ_MEASURE())
        self.index += 1

    def _build_custom_source_target_op(self):
        instr = self.circuit[self.index]
        source, target = qubit_indices(instr)
        displaced = self.diagram.interval(min(source, target), max(source, target))
        self.diagram.extend_lines_to_common_edge(displaced)
        source_op, target_op = SOURCE_TARGET_OP[instr.name]
        offset = (len(displaced) - 1)*(-1 if source > target else 1)
        self.diagram.append(source, source_op(source, offset))
        self.diagram.append(target, target_op())
        self.diagram.extend_lines_to_common_edge(displaced)
        self.index += 1

    def _build_1q_unitary(self):
        instr = self.circuit[self.index]
        qubits = qubit_indices(instr)
        dagger = sum(m == 'DAGGER' for m in instr.modifiers) % 2 == 1
        self.diagram.append(qubits[0], TIKZ_GATE(instr.name, params=instr.params, dagger=dagger, settings=self.settings))
        self.index += 1

    def _build_generic_unitary(self):
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

