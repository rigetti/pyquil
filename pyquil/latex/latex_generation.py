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

from pyquil.quil import Measurement, Gate, Pragma
from pyquil.quilbase import _format_params
from collections import defaultdict

# TODO: consider what settings are meaningful
# - texify names
# - texify params
# - ... ?

def to_latex(circuit, settings=None):
    """
    Translates a given pyquil Program to a TikZ picture in a Latex document.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings: An optional dictionary with settings for drawing the circuit.
        See `get_default_settings` in `latex_config` for more information about what settings
        should contain.
    :return: LaTeX document string which can be compiled.
    :rtype: string
    """
    settings = None
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

def TIKZ_CONTROL(control, target):
    return r"\ctrl{{{offset}}}".format(offset=target-control)

def TIKZ_CNOT_TARGET():
    return r"\targ{}"

def TIKZ_CPHASE_TARGET():
    return r"\control{}"

def TIKZ_SWAP(source, target):
    return "\swap{{{offset}}}".format(offset=target-source)

def TIKZ_SWAP_TARGET():
    return r"\targX{}"

def TIKZ_NOP():
    return r"\qw"

def TIKZ_MEASURE():
    return r"\meter{}"

def TIKZ_GATE(name, size=1, params=None, dagger=False):
    cmd = r"\gate"
    if size > 1:
        cmd += "[wires={size}]".format(size=size)
    # TeXify names
    if name in ["RX", "RY", "RZ"]:
        name = name[0] + "_" + name[1].lower()
    if dagger:
        name += "^{\dagger}"
    if params:
        # TODO we should do a better job than just dumb str.replace
        name += _format_params(params).replace("pi", "\pi")
    return cmd + "{{{name}}}".format(name=name)

def TIKZ_GATE_GROUP(qubits, width, label):
    num_qubits = max(qubits) - min(qubits) + 1
    return "\\gategroup[{qubits},steps={width},style={{dashed, rounded corners,fill=blue!20, inner xsep=2pt}}, background]{{{label}}}".format(
        qubits=num_qubits, width=width, label=label)

def is_interval(indices):
    return all(j == i + 1 for i,j in zip(indices, indices[1:]))

def interval(source, target):
    return list(range(min(source, target), max(source,target)+1))

def scan_for_grouped_qubits(program):
    qubits = set()
    for instr in program:
        if isinstance(instr, Pragma) and instr.command == "END_LATEX_GATE_GROUP":
            return sorted(qubits)
        elif isinstance(instr, Measurement):
            qubits.add(instr.qubit.index)
        elif isinstance(instr, Gate):
            qubits = qubits.union(q.index for q in instr.qubits)
    # no matching END_LATEX_GROUP
    return None

def body(circuit, settings):
    """
    Return the body of the Latex document, including the entire circuit in
    TikZ format.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings:

    :return: Latex string to draw the entire circuit.
    :rtype: string
    """

    all_qubits = range(min(circuit.get_qubits()), max(circuit.get_qubits()) + 1)

    # for each qubit, a list of quantikz commands
    lines = defaultdict(list)

    def qubit_clock(q):
        "Get the current depth of the line on qubit q."
        return len(lines[q])

    def nop_to_latest_edge(qubits, offset=0):
        "Add NOP instructions to qubit lines until they have the same clock values."
        latest = max(qubit_clock(q) for q in qubits) + offset
        for q in qubits:
            while qubit_clock(q) < latest:
                lines[q].append(TIKZ_NOP())

    # initial exposed wires
    nop_to_latest_edge(all_qubits, offset=1)

    # fill lines
    for i,instr in enumerate(circuit):
        # TODO error on classical control flow
        if isinstance(instr, Pragma) and instr.command == 'LATEX_GATE_GROUP':
            grouped_qubits = scan_for_grouped_qubits(circuit[i+1:])
            if grouped_qubits is None:
                raise ValueError("PRAGMA LATEX_GATE_GROUP found without matching END_LATEX_GATE_GROUP")
            elif len(grouped_qubits) == 0:
                warn("Ignoring empty gate group.")
            else:
                nop_to_latest_edge(grouped_qubits)
                # we track the place where the grouping command will be INSERTED once we have reached
                # the end of the block
                group_op_index = qubit_clock(grouped_qubits[0])
                # until then, business as usual
        elif isinstance(instr, Pragma) and instr.command == 'END_LATEX_GATE_GROUP':
            # add nop_to_latest_edge on grouped qubits
            if grouped_qubits is None:
                raise ValueError("PRAGMA END_LATEX_GATE_GROUP found, but there is no active gate group.")
            elif len(grouped_qubits) == 0:
                pass
            else:
                nop_to_latest_edge(grouped_qubits)
                q = grouped_qubits[0]
                # TODO label? we need to keep the pragma, also can allow for nested groups
                lines[q][group_op_index] += " " + TIKZ_GATE_GROUP(grouped_qubits, qubit_clock(q)-group_op_index, "")
                grouped_qubits = None
        elif isinstance(instr, Measurement):
            lines[instr.qubit].append(TIKZ_MEASURE())
        elif isinstance(instr, Gate):
            if 'FORKED' in instr.modifiers:
                raise ValueError("LaTeX output does not currently support FORKED modifiers: {}".format(instr))


            dagger = sum(m == 'DAGGER' for m in instr.modifiers) % 2 == 1
            controls = sum(m == 'CONTROLLED' for m in instr.modifiers)
            qubits = [qubit.index for qubit in instr.qubits]
            # the easy case is 1q operations
            if len(qubits) == 1:
                lines[qubits[0]].append(TIKZ_GATE(instr.name, params=instr.params, dagger=dagger))
            else:
                # We have a bunch of special cases here for
                # gates which are controlled etc but without explicit control
                # modifiers.
                if instr.name == "CNOT" and not instr.modifiers:
                    control, target = qubits
                    displaced = set(qubits + interval(control,target))
                    nop_to_latest_edge(displaced)
                    lines[control].append(TIKZ_CONTROL(control, target))
                    lines[target].append(TIKZ_CNOT_TARGET())
                    nop_to_latest_edge(displaced)
                elif instr.name == "SWAP" and not instr.modifiers:
                    source, target = qubits
                    displaced = set(qubits + interval(source,target))
                    nop_to_latest_edge(displaced)
                    lines[source].append(TIKZ_SWAP(source, target))
                    lines[target].append(TIKZ_SWAP_TARGET())
                    nop_to_latest_edge(displaced)
                elif instr.name == "CZ" and not instr.modifiers:
                    # we destructure to make this show as a controlled-Z
                    control, target = qubits
                    displaced = set(qubits + interval(control,target))
                    nop_to_latest_edge(displaced)
                    lines[control].append(TIKZ_CONTROL(control, target))
                    lines[target].append(TIKZ_GATE("Z"))
                    nop_to_latest_edge(displaced)
                elif instr.name == "CPHASE" and not instr.modifiers:
                    control, target = qubits
                    displaced = set(qubits + interval(control,target))
                    nop_to_latest_edge(displaced)
                    lines[control].append(TIKZ_CONTROL(control, target))
                    lines[target].append(TIKZ_CPHASE_TARGET())
                    nop_to_latest_edge(displaced)
                else:
                    # generic unitary
                    nop_to_latest_edge(qubits)

                    control_qubits = qubits[:controls]
                    target_qubits = qubits[controls:]
                    if not is_interval(sorted(target_qubits)):
                        raise ValueError("Unable to render instruction {} which targets non-adjacent qubits.".format(instr))

                    for q in control_qubits:
                        lines[q].append(TIKZ_CONTROL(q, target_qubits[0]))

                    # we put the gate on the first target line, and nop on the others
                    lines[target_qubits[0]].append(TIKZ_GATE(instr.name, size=len(qubits), params=instr.params, dagger=dagger))
                    for q in target_qubits[1:]:
                        lines[q].append(TIKZ_NOP())

    # fill in qubit lines, leaving exposed wires on the right
    nop_to_latest_edge(all_qubits, offset=1)

    # flush lines
    quantikz_out = []
    for qubit in all_qubits:
        quantikz_out.append(" & ".join(lines[qubit]))

    return " \\\\\n".join(quantikz_out)
