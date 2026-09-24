"""Generate LaTeX diagrams from a ``Program``.

.. deprecated:: 4.22.0
    To be removed in pyQuil v5, with no replacement in pyQuil.
"""

__all__ = [
    "DiagramSettings",
    "display",
    "to_latex",
]

import warnings

from pyquil._deprecation import DEPRECATED_IN_VERSION, LATEX_REASON, PyQuilDeprecationWarning
from pyquil.latex._diagram import DiagramSettings
from pyquil.latex._ipython import display
from pyquil.latex._main import to_latex

warnings.warn(
    f"The module {__name__} is deprecated. ({LATEX_REASON}) -- Deprecated since version {DEPRECATED_IN_VERSION}.",
    PyQuilDeprecationWarning,
    stacklevel=2,
)
