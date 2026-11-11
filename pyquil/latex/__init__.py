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

from pyquil._deprecation import PyQuilDeprecationWarning
from pyquil.latex._diagram import DiagramSettings
from pyquil.latex._ipython import display
from pyquil.latex._main import to_latex

warnings.warn(
    f"The module {__name__} is deprecated and will be removed in pyQuil v5, with no replacement "
    f"in pyQuil. -- Deprecated since version 4.22.0.",
    PyQuilDeprecationWarning,
    stacklevel=2,
)
