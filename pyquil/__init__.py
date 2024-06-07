"""pyQuil is a Python library for quantum programming using Quil. pyQuil enables users to construct, manipulate, and execute quantum programs on a quantum virtual machine (QVM) and Rigetti quantum processing units (QPUs).

Refer to the `pyQuil Documentation <https://pyquil-docs.rigetti.com/en/stable/>`_ for more information.
"""

from pyquil._version import pyquil_version
from pyquil.api import get_qc, list_quantum_computers
from pyquil.quil import Program

__version__ = pyquil_version

__all__ = ["get_qc", "list_quantum_computers", "Program", "__version__"]
