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

import sys

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass


@dataclass
class DiagramSettings:
    """
    Settings to control the layout and rendering of circuits.
    """

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
