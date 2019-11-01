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
from typing import Optional

from pyquil import Program
from pyquil.latex._diagram import header, body, footer, DiagramSettings


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
    text = header()
    text += "\n"
    text += body(circuit, settings)
    text += "\n"
    text += footer()
    return text
