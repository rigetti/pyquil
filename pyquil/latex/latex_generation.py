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

import warnings
from typing import Optional

from pyquil.latex._diagram import DiagramSettings
from pyquil.quil import Program


def to_latex(circuit: Program, settings: Optional[DiagramSettings] = None) -> str:
    from pyquil.latex._main import to_latex

    warnings.warn(
        '"pyquil.latex.latex_generation.to_latex" has been moved -- please import it'
        'as "from pyquil.latex import to_latex going forward"',
        FutureWarning,
    )
    return to_latex(circuit, settings)
