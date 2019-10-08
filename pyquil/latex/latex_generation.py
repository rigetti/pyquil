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

from pyquil.quil import Measurement, Gate
from pyquil.quilbase import _format_params
from collections import defaultdict


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

def TIKZ_GATE(instr):
    size = len(instr.qubits)
    cmd = r"\gate"
    if size > 1:
        cmd += "[wires={size}]".format(size=size)
    # TODO: R_x etc
    name = instr.name
    if instr.params:
        name += _format_params(instr.params)
    return cmd + "{{{name}}}".format(name=name)

def is_interval(indices):
    for i,j in zip(indices, indices[1:]):
        if j != i + 1:
            return False
    return True

def body(circuit, settings):
    """
    Return the body of the Latex document, including the entire circuit in
    TikZ format.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param dict settings:

    :return: Latex string to draw the entire circuit.
    :rtype: string
    """

    # TODO we assume that qubit indices are remapped to the interval [0,1,...,k]
    all_qubits = range(min(circuit.get_qubits()), max(circuit.get_qubits()) + 1)

    # for each qubit, a list of quantikz commands
    lines = defaultdict(list)

    def qubit_clock(q):
        return len(lines[q])

    def nop_to_latest_edge(qubits, offset=0):
        latest = max(qubit_clock(q) for q in qubits) + offset
        for q in qubits:
            while qubit_clock(q) < latest:
                lines[q].append(TIKZ_NOP())

    # initial exposed wires
    nop_to_latest_edge(all_qubits, offset=1)

    # fill lines
    for instr in circuit:
        if isinstance(instr, Measurement):
            lines[instr.qubit].append(TIKZ_MEASURE())
        elif isinstance(instr, Gate):
            # TODO add support for dagger and controlled modifiers
            if instr.modifiers:
                raise ValueError("LaTeX output does not currently support gate modifiers: {}".format(instr))
            qubits = [qubit.index for qubit in instr.qubits]
            if len(qubits) == 1:
                lines[qubits[0]].append(TIKZ_GATE(instr))
            else:
                # fill to latest edge
                nop_to_latest_edge(qubits)

                # dispatch on name
                if instr.name == "CNOT":
                    control, target = qubits
                    lines[control].append(TIKZ_CONTROL(control, target))
                    lines[target].append(TIKZ_CNOT_TARGET())
                elif instr.name == "SWAP":
                    source, target = qubits
                    lines[source].append(TIKZ_SWAP(source, target))
                    lines[target].append(TIKZ_SWAP_TARGET(target))
                elif instr.name == "CZ":
                    control, target = qubits
                    lines[control].append(TIKZ_CONTROL(control, target))
                    lines[target].append(TIKZ_GATE(instr))
                else: # generic unitary
                    if not is_interval(sorted(qubits)):
                        raise ValueError("Unable to render instruction {} which spans non-adjacent qubits.".format(instr))

                    # we put the gate on the first line, and nop on the others
                    qubit, *remaining = qubits
                    lines[qubit].append(TIKZ_GATE(instr))
                    for q in remaining:
                        lines[q].append(TIKZ_NOP())

    # fill in qubit lines, leaving exposed wires on the right
    nop_to_latest_edge(all_qubits, offset=1)

    # flush lines
    quantikz_out = []
    for qubit in all_qubits:
        quantikz_out.append(" & ".join(lines[qubit]))

    return " \\\\\n".join(quantikz_out)
