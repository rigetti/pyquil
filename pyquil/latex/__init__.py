"""Generate LaTeX diagrams from a ``Program``."""

__all__ = [
    "DiagramSettings",
    "display",
    "to_latex",
]

from pyquil.latex._diagram import DiagramSettings
from pyquil.latex._ipython import display
from pyquil.latex._main import to_latex
