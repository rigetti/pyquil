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
"""Convert a ``Program`` to a LaTeX quantum circuit diagram.

.. deprecated:: 4.0
    This module has been moved: import from ``pyquil.latex`` instead.

.. deprecated:: 4.22.0
    ``pyquil.latex`` is itself deprecated and will be removed in pyQuil v5, with no replacement in pyQuil.
"""

import warnings

from deprecated.sphinx import deprecated

from pyquil._deprecation import DEPRECATED_IN_VERSION, LATEX_REASON, PyQuilDeprecationWarning
from pyquil.latex._diagram import DiagramSettings
from pyquil.quil import Program

warnings.warn(
    f"The module {__name__} is deprecated. ({LATEX_REASON}) -- Deprecated since version {DEPRECATED_IN_VERSION}.",
    PyQuilDeprecationWarning,
    stacklevel=2,
)


@deprecated(
    version=DEPRECATED_IN_VERSION,
    reason=LATEX_REASON,
    category=PyQuilDeprecationWarning,
)
def to_latex(circuit: Program, settings: DiagramSettings | None = None) -> str:
    """Produce a circuit diagram in LaTeX for a given pyQuil Program."""
    from pyquil.latex._main import to_latex

    return to_latex(circuit, settings)
