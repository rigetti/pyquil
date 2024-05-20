import os

from pyquil._core import *
from pyquil._version import pyquil_version
from pyquil.api import get_qc, list_quantum_computers

if os.environ.get("PYQUIL_EXPERIMENTAL_BACKEND", False):
    from pyquil._core.program import Program
else:
    from pyquil.quil import Program

__version__ = pyquil_version

__all__ = ["__version__", "Program", "get_qc", "list_quantum_computers"]
