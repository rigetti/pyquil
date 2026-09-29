"""Generate LaTeX diagrams from a ``Program``.

.. deprecated:: PENDING_DEPRECATION_RELEASE
    To be removed in pyQuil v5, with no replacement in pyQuil.
"""

__all__ = [
    "DiagramSettings",
    "display",
    "to_latex",
]

import warnings

from pyquil._deprecation import PENDING_DEPRECATION_RELEASE, PyQuilDeprecationWarning
from pyquil.latex._diagram import DiagramSettings
from pyquil.latex._ipython import display
from pyquil.latex._main import to_latex

warnings.warn(
    f"The module {__name__} is deprecated and will be removed in pyQuil v5, with no replacement "
    f"in pyQuil. -- Deprecated since version {PENDING_DEPRECATION_RELEASE}.",
    PyQuilDeprecationWarning,
    stacklevel=2,
)
