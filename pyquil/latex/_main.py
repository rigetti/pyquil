##############################################################################
# Copyright 2016-2019 Rigetti Computing
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
"""The main entry point to the LaTeX generation functionality in pyQuil."""

from typing import Optional

from pyquil.latex._diagram import DiagramBuilder, DiagramSettings
from pyquil.quil import Program


def to_latex(circuit: Program, settings: Optional[DiagramSettings] = None) -> str:
    """Translate a given pyQuil Program to a TikZ picture in a LaTeX document.

    Here are some high points of the generation procedure (see ``pyquil/latex/_diagram.py``):

    - The most basic building block are the TikZ operators, which are constructed
      by the functions in ``_diagram.py`` (e.g. TIKZ_CONTROL, TIKZ_NOP, TIKZ_MEASURE).
    - TikZ operators are maintained by a DiagramState object, with roughly each
      qubit line in a diagram represented as a list of TikZ operators on the ``DiagramState``.
    - The ``DiagramBuilder`` is the actual driver. This traverses a ``Program`` and, for
      each instruction, performs a suitable manipulation of the ``DiagramState``. At
      the end of this, the ``DiagramState`` is traversed and raw LaTeX is emitted.
    - Most options are specified by ``DiagramSettings``. One exception is this: it is possible
      to request that a certain subset of the program is rendered as a group (and colored
      as such). This is specified by a new pragma in the ``Program`` source:

        PRAGMA LATEX_GATE_GROUP <name>?
        ...
        PRAGMA END_LATEX_GATE_GROUP

      The <name> is optional, and will be used to label the group. Nested gate
      groups are currently not supported.

    :param circuit: The circuit to be drawn, represented as a pyquil program.
    :param settings: An optional object of settings controlling diagram rendering and layout.
    :return: LaTeX document string which can be compiled.
    """
    if settings is None:
        settings = DiagramSettings()
    text = header()
    text += "\n"
    text += body(circuit, settings)
    text += "\n"
    text += footer()
    return text


def header() -> str:
    """Write the LaTeX header using the settings file.

    The header includes all packages and defines all tikz styles.

    :return: Header of the LaTeX document.
    """
    packages = (
        r"\documentclass[convert={density=300,outext=.png}]{standalone}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{quantikz}",
    )

    init = (r"\begin{document}", r"\begin{tikzcd}")

    return "\n".join(("\n".join(packages), "\n".join(init)))


def footer() -> str:
    """Return the footer of the LaTeX document.

    :return: LaTeX document footer.
    """
    return "\\end{tikzcd}\n\\end{document}"


def body(circuit: Program, settings: DiagramSettings) -> str:
    """Return the body of the LaTeX document, including the entire circuit in TikZ format.

    :param circuit: The circuit to be drawn, represented as a pyquil program.
    :param settings: Options controlling rendering and layout.

    :return: LaTeX string to draw the entire circuit.
    """
    diagram = DiagramBuilder(circuit, settings).build()

    # flush lines
    quantikz_out = []
    for qubit in diagram.qubits:
        quantikz_out.append(" & ".join(diagram.lines[qubit]))

    return " \\\\\n".join(quantikz_out)
