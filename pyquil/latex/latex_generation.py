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

from pyquil.latex.latex_config import get_default_settings, header, footer
from pyquil.quil import Measurement
from collections import namedtuple


command = namedtuple("command", ("gate", "lines", "ctrl_lines", "target_lines", "id"))
"""command is used as an intermediate representation to hold meta-information about the circuit, and dependencies."""
ALLOCATE = "ALLOCATE"
CZ = "CZ"
CNOT = "CNOT"
Z = "Z"
SWAP = "SWAP"
MEASURE = "MEASURE"
X = "X"


def to_latex(circuit, settings=None):
    """
    Translates a given pyquil Program to a TikZ picture in a Latex document.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings: An optional dictionary with settings for drawing the circuit. See `get_default_settings`
     in `latex_config` for more information about what settings should contain.
    :return: LaTeX document string which can be compiled.
    :rtype: string
    """
    if settings is None:
        settings = get_default_settings()
    text = header(settings)
    text += body(circuit, settings)
    text += footer()
    return text


def body(circuit, settings):
    """
    Return the body of the Latex document, including the entire circuit in
    TikZ format.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings:

    :return: Latex string to draw the entire circuit.
    :rtype: string
    """
    qubit_instruction_mapping = {}

    # Allocate each qubit.
    for inst in circuit:
        if isinstance(inst, Measurement):
            inst.qubits = [inst.qubit]
            inst.name = "MEASURE"
        else:
            qubits = inst.qubits
        for qubit in qubits:
            qubit_instruction_mapping[qubit.index] = []
    for k, v in list(qubit_instruction_mapping.items()):
        v.append(command(ALLOCATE, [k], [], [k], k))

    for inst in circuit:
        qubits = [qubit.index for qubit in inst.qubits]
        gate = inst.name
        # If this is a single qubit instruction.
        if len(qubits) == 1:
            for qubit in qubits:
                qubit_instruction_mapping[qubit].append(command(gate, [qubit], [], [qubit], qubit))

        # If this is a many-qubit operation.
        else:
            # All qubits explicitly involved in the gate.
            explicit_lines = [qubit for qubit in copy(qubits)]
            # All lines to consider that will run through this circuit element.
            all_lines = list(range(min(explicit_lines), max(explicit_lines) + 1))
            # Include all lines that are in-use and in the range of lines used in this instruction.
            for line in all_lines:
                if line not in qubit_instruction_mapping.keys() and line in all_lines:
                    all_lines.remove(line)
            for i, qubit in enumerate(all_lines):
                if gate == CZ:
                    ctrl_lines = list(explicit_lines)
                    ctrl_lines.remove(qubits[-1])
                    qubit_instruction_mapping[qubit].append(command(Z, list(all_lines), list(ctrl_lines),
                                                                    qubits[-1:], None))
                elif gate == CNOT:
                    ctrl_lines = list(explicit_lines)
                    ctrl_lines.remove(qubits[-1])
                    qubit_instruction_mapping[qubit].append(command(X, list(all_lines), list(ctrl_lines),
                                                                    qubits[-1:], None))
                else:
                    qubit_instruction_mapping[qubit].append(command(gate, list(all_lines), [],
                                                                    list(explicit_lines), None))

    # Zero index, and remove gaps in spacing.
    relabeled_circuit = {}
    # Store a mapping so we can relabel command labels.
    index_map = {}
    for i, key in enumerate(sorted(qubit_instruction_mapping.keys())):
        relabeled_circuit[i] = qubit_instruction_mapping[key]
        index_map[key] = i

    for line in list(relabeled_circuit.values()):
        for cmd in line:
            for i, qubit in enumerate(cmd.lines):
                cmd.lines[i] = index_map[qubit]
            for i, qubit in enumerate(cmd.ctrl_lines):
                cmd.ctrl_lines[i] = index_map[qubit]
            for i, qubit in enumerate(cmd.target_lines):
                cmd.target_lines[i] = index_map[qubit]
    code_generator = CircuitTikzGenerator(settings)
    return code_generator.generate_circuit(relabeled_circuit)


class CircuitTikzGenerator(object):
    """
    Takes a circuit and turns it into Latex/TikZ code.
    """
    def __init__(self, settings):
        """
        Initialize a circuit to LaTeX converter object.

        :param dict settings: Dictionary of settings to use for the TikZ image.
        :param int num_lines: Number of qubit lines to use for the entire circuit.
        """
        self.pos = None
        self.op_count = None
        self.is_quantum = None
        self.settings = settings

    def generate_circuit(self, circuit_description):

        num_lines = len(circuit_description)
        self.pos = [0.] * num_lines
        self.op_count = [0] * num_lines
        self.is_quantum = [self.settings['lines']['init_quantum']] * num_lines
        code = []
        for line in circuit_description.keys():
            code.append(self.to_tikz(line, circuit_description))
        return "".join(code)

    def to_tikz(self, line, circuit, end=None):
        """
        Generate the TikZ code for one line of the circuit up to a certain
        gate.

        It modifies the circuit to include only the gates which have not been
        drawn. It automatically switches to other lines if the gates on those
        lines have to be drawn earlier.

        :param int line: Line to generate the TikZ code for.
        :param Program circuit: The circuit to draw.
        :param int end: Gate index to stop at (for recursion).
        :return:TikZ code representing the current qubit line and, if it was
         necessary to draw other lines, those lines as well.
        :rtype: string
        """
        if end is None:
            end = len(circuit[line])
        tikz_code = []
        cmds = circuit[line]
        for i in range(0, end):
            gate = cmds[i].gate
            lines = cmds[i].lines
            ctrl_lines = cmds[i].ctrl_lines
            all_lines = list(lines)
            all_lines.remove(line)
            for l in all_lines:
                gate_idx = 0
                while not (circuit[l][gate_idx] == cmds[i]):
                    gate_idx += 1
                tikz_code.append(self.to_tikz(l, circuit, gate_idx))

                # We remove the first element of the current line, since we have processed it.
                circuit[l] = circuit[l][1:]
            all_lines = cmds[i].target_lines + cmds[i].ctrl_lines
            pos = max([self.pos[l] for l in range(min(all_lines),
                                                  max(all_lines) + 1)])
            for l in range(min(all_lines), max(all_lines) + 1):
                self.pos[l] = pos + self._gate_pre_offset(gate)

            connections = ""
            for l in all_lines:
                connections += self._line(self.op_count[l] - 1, self.op_count[l], line=l)
            add_str = ""
            if gate == X:
                # draw NOT-gate with controls
                add_str = self._x_gate(cmds[i].target_lines, ctrl_lines)
                # and make the target qubit quantum if one of the controls is
                if not self.is_quantum[lines[0]]:
                    if sum([self.is_quantum[i] for i in ctrl_lines]) > 0:
                        self.is_quantum[lines[0]] = True
            elif gate == Z and len(ctrl_lines) > 0:
                add_str = self._cz_gate(lines + ctrl_lines)
            elif gate == SWAP:
                add_str = self._swap_gate(lines, ctrl_lines)
            elif gate == MEASURE:
                # draw measurement gate
                for l in lines:
                    op = self._op(l)
                    width = self._gate_width(MEASURE)
                    height = self._gate_height(MEASURE)
                    shift0 = .07 * height
                    shift1 = .36 * height
                    shift2 = .1 * width
                    add_str += ("\n\\node[measure,edgestyle] ({op}) at ({pos},-{line}) {{}};\n"
                                "\\draw[edgestyle] ([yshift=-{shift1}cm,xshift={shift2}cm]{op}.west)"
                                " to [out=60,in=180] ([yshift={shift0}cm]{op}."
                                "center) to [out=0, in=120] ([yshift=-{shift1}"
                                "cm,xshift=-{shift2}cm]{op}.east);\n"
                                "\\draw[edgestyle] ([yshift=-{shift1}cm]{op}."
                                "center) to ([yshift=-{shift2}cm,xshift=-{shift1}cm]{op}.north east);").format(
                        op=op, pos=self.pos[l], line=l, shift0=shift0, shift1=shift1, shift2=shift2)
                    self.op_count[l] += 1
                    self.pos[l] += (self._gate_width(gate) + self._gate_offset(gate))
                    self.is_quantum[l] = False
            elif gate == ALLOCATE:
                # Draw 'begin line'.
                add_str = "\n\\node[none] ({}) at ({},-{}) {{$\\Ket{{0}}{}$}};"
                id_str = ""
                if self.settings['gates']['AllocateQubitGate']['draw_id']:
                    id_str = "^{{\\textcolor{{red}}{{{}}}}}".format(cmds[i].id)
                add_str = add_str.format(self._op(line), self.pos[line], line,
                                         id_str)
                self.op_count[line] += 1
                self.pos[line] += (self._gate_offset(gate) + self._gate_width(gate))
                self.is_quantum[line] = self.settings['lines']['init_quantum']
            else:
                # Regular gate must draw the lines it does not act upon.
                # If it spans multiple qubits.
                add_str = self._regular_gate(gate, lines, ctrl_lines, cmds[i].target_lines)
                for l in lines:
                    self.is_quantum[l] = True
            tikz_code.append(add_str)
            if not gate == ALLOCATE:
                tikz_code.append(connections)
        circuit[line] = circuit[line][end:]
        return "".join(tikz_code)

    def _gate_name(self, gate):
        """
        Return the string representation of the gate.

        Tries to use gate.tex_str and, if that is not available, uses str(gate)
        instead.

        :param string gate: Gate object of which to get the name / LaTeX representation.
        :return: LaTeX gate name.
        :rtype: string
        """
        try:
            name = gate.tex_str()
        except AttributeError:
            name = str(gate)
        return name

    def _swap_gate(self, lines, ctrl_lines):
        """
        Return the TikZ code for a Swap-gate.

        :param lines: List of length 2 denoting the target qubit of the Swap gate.
        :type: list[int]
        :param ctrl_lines: List of qubit lines which act as controls.
        :type: list[int]
        """
        delta_pos = self._gate_offset(SWAP)
        gate_width = self._gate_width(SWAP)
        lines.sort()

        gate_str = ""
        for line in lines:
            op = self._op(line)
            w = "{}cm".format(.5 * gate_width)
            s1 = "[xshift=-{w},yshift=-{w}]{op}.center".format(w=w, op=op)
            s2 = "[xshift={w},yshift={w}]{op}.center".format(w=w, op=op)
            s3 = "[xshift=-{w},yshift={w}]{op}.center".format(w=w, op=op)
            s4 = "[xshift={w},yshift=-{w}]{op}.center".format(w=w, op=op)
            swap_style = "swapstyle,edgestyle"
            if self.settings['gate_shadow']:
                swap_style += ",shadowed"
            gate_str += ("\n\\node[swapstyle] ({op}) at ({pos},-{line}) {{}};"
                         "\n\\draw[{swap_style}] ({s1})--({s2});\n"
                         "\\draw[{swap_style}] ({s3})--({s4});"
                         ).format(op=op, s1=s1, s2=s2, s3=s3, s4=s4,
                                  line=line, pos=self.pos[line],
                                  swap_style=swap_style)
        gate_str += self._line(lines[0], lines[1])
        all_lines = ctrl_lines + lines
        new_pos = self.pos[lines[0]] + delta_pos + gate_width
        for i in all_lines:
            self.op_count[i] += 1
        for i in range(min(all_lines), max(all_lines) + 1):
            self.pos[i] = new_pos
        return gate_str

    def _x_gate(self, lines, ctrl_lines):
        """
        Return the TikZ code for a NOT-gate.

        :param lines: List of length 1 denoting the target qubit of the NOT / X gate.
        :type: list[int]
        :param ctrl_lines: List of qubit lines which act as controls.
        :type: list[int]
        """
        line = lines[0]
        delta_pos = self._gate_offset('X')
        gate_width = self._gate_width('X')
        op = self._op(line)
        gate_str = ("\n\\node[xstyle] ({op}) at ({pos},-{line}) {{}};\n\\draw"
                    "[edgestyle] ({op}.north)--({op}.south);\n\\draw"
                    "[edgestyle] ({op}.west)--({op}.east);").format(op=op, line=line, pos=self.pos[line])
        if len(ctrl_lines) > 0:
            for ctrl in ctrl_lines:
                gate_str += self._phase(ctrl, self.pos[line])
                gate_str += self._line(ctrl, line)

        all_lines = ctrl_lines + [line]
        new_pos = self.pos[line] + delta_pos + gate_width
        for i in all_lines:
            self.op_count[i] += 1
        for i in range(min(all_lines), max(all_lines) + 1):
            self.pos[i] = new_pos
        return gate_str

    def _cz_gate(self, lines):
        """
        Return the TikZ code for an n-controlled Z-gate.

        :param lines: List of all qubits involved.
        :type: list[int]
        """
        line = lines[0]
        delta_pos = self._gate_offset(Z)
        gate_width = self._gate_width(Z)
        gate_str = self._phase(line, self.pos[line])

        for ctrl in lines[1:]:
            gate_str += self._phase(ctrl, self.pos[line])
            gate_str += self._line(ctrl, line)

        new_pos = self.pos[line] + delta_pos + gate_width
        for i in lines:
            self.op_count[i] += 1
        for i in range(min(lines), max(lines) + 1):
            self.pos[i] = new_pos
        return gate_str

    def _gate_width(self, gate):
        """
        Return the gate width, using the settings (if available).

        :param string gate: The name of the gate whose height is desired.
        :return: Width of the gate.
        :rtype: float
        """
        try:
            gates = self.settings['gates']
            gate_width = gates[gate.__class__.__name__]['width']
        except KeyError:
            gate_width = .5
        return gate_width

    def _gate_pre_offset(self, gate):
        """
        Return the offset to use before placing this gate.

        :param string gate: The name of the gate whose pre-offset is desired.
        :return: Offset to use before the gate.
        :rtype: float
        """
        try:
            gates = self.settings['gates']
            delta_pos = gates[gate.__class__.__name__]['pre_offset']
        except KeyError:
            delta_pos = self._gate_offset(gate)
        return delta_pos

    def _gate_offset(self, gate):
        """
        Return the offset to use after placing this gate and, if no pre_offset
        is defined, the same offset is used in front of the gate.

        :param string gate: The name of the gate whose offset is desired.
        :return: Offset.
        :rtype: float
        """
        try:
            gates = self.settings['gates']
            delta_pos = gates[gate.__class__.__name__]['offset']
        except KeyError:
            delta_pos = .2
        return delta_pos

    def _gate_height(self, gate):
        """
        Return the height to use for this gate.

        :param string gate: The name of the gate whose height is desired.
        :return: Height of the gate.
        :rtype: float
        """
        try:
            height = self.settings['gates'][gate.__class__.__name__]['height']
        except KeyError:
            height = .5
        return height

    def _phase(self, line, pos):
        """
        Places a phase / control circle on a qubit line at a given position.

        :param int line: Qubit line at which to place the circle.
        :param float pos: Position at which to place the circle.
        :return: Latex string representing a control circle at the given position.
        :rtype: string
        """
        phase_str = "\n\\node[phase] ({}) at ({},-{}) {{}};"
        return phase_str.format(self._op(line), pos, line)

    def _op(self, line, op=None, offset=0):
        """
        Returns the gate name for placing a gate on a line.

        :param int line: Line number.
        :param int op: Operation number or, by default, uses the current op count.
        :return: Gate name.
        :rtype: string
        """
        if op is None:
            op = self.op_count[line]
        return "line{}_gate{}".format(line, op + offset)

    def _line(self, p1, p2, line=None):
        """
        Connects p1 and p2, where p1 and p2 are either to qubit line indices,
        in which case the two most recent gates are connected, or two gate
        indices, in which case line denotes the line number and the two gates
        are connected on the given line.

        :param int p1: Index of the first object to connect.
        :param int p2: Index of the second object to connect.
        :param int line: Line index - if provided, p1 and p2 are gate indices.
        :return: Latex code to draw this / these line(s).
        :rtype: string
        """
        dbl_classical = self.settings['lines']['double_classical']

        if line is None:
            quantum = not dbl_classical or self.is_quantum[p1]
            op1, op2 = self._op(p1), self._op(p2)
            loc1, loc2 = 'north', 'south'
            shift = "xshift={}cm"
        else:
            quantum = not dbl_classical or self.is_quantum[line]
            op1, op2 = self._op(line, p1), self._op(line, p2)
            loc1, loc2 = 'west', 'east'
            shift = "yshift={}cm"

        if quantum:
            return "\n\\draw ({}) edge[edgestyle] ({});".format(op1, op2)
        else:
            if p2 > p1:
                loc1, loc2 = loc2, loc1
            edge_str = "\n\\draw ([{shift}]{op1}.{loc1}) edge[edgestyle] ([{shift}]{op2}.{loc2});"
            line_sep = self.settings['lines']['double_lines_sep']
            shift1 = shift.format(line_sep / 2.)
            shift2 = shift.format(-line_sep / 2.)
            edges_str = edge_str.format(shift=shift1, op1=op1, op2=op2, loc1=loc1, loc2=loc2)
            edges_str += edge_str.format(shift=shift2, op1=op1, op2=op2, loc1=loc1, loc2=loc2)
            return edges_str

    def _regular_gate(self, gate, lines, ctrl_lines, used_lines):
        """
        Draw a regular gate.

        :param string gate: Gate to draw.
        :param lines: Lines the gate acts on.
        :type: list[int]
        :param int ctrl_lines: Control lines.
        :param int used_lines: The lines that are actually involved in the gate.
        :return: LaTeX string drawing a regular gate at the given location.
        :rtype: string
        """
        imax = max(lines)
        imin = min(lines)

        delta_pos = self._gate_offset(gate)
        gate_width = self._gate_width(gate)
        gate_height = self._gate_height(gate)

        name = self._gate_name(gate)

        lines = list(range(imin, imax + 1))

        tex_str = ""
        pos = self.pos[lines[0]]

        node_str = "\n\\node[none] ({}) at ({},-{}) {{}};"
        for l in lines:
            node1 = node_str.format(self._op(l), pos, l)
            if l in used_lines:
                tex_str += self._phase(l, pos)
            node2 = ("\n\\node[none,minimum height={}cm,outer sep=0] ({}) at"
                     " ({},-{}) {{}};").format(gate_height, self._op(l, offset=1), pos + gate_width / 2., l)
            node3 = node_str.format(self._op(l, offset=2), pos + gate_width, l)
            tex_str += node1 + node2 + node3
        tex_str += ("\n\\draw[operator,edgestyle,outer sep={width}cm]"
                    " ([yshift={half_height}cm]{op1})"
                    " rectangle ([yshift=-{half_height}cm]{op2}) node[pos=.5]{{\\verb|{name}|}};"
                    ).format(width=gate_width, op1=self._op(imin), op2=self._op(imax, offset=2),
                             half_height=.5 * gate_height, name=name)
        for l in lines:
            self.pos[l] = pos + gate_width / 2.
            self.op_count[l] += 3
        for l in range(min(ctrl_lines + lines), max(ctrl_lines + lines) + 1):
            self.pos[l] = pos + delta_pos + gate_width
        return tex_str
