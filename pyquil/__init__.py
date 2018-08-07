__version__ = "2.0.0.dev0"

from pyquil.quil import Program, Pragma
from pyquil.api import QVMConnection, QPUConnection, CompilerConnection, get_devices
from pyquil.parameters import Parameter, quil_sin, quil_cos, quil_sqrt, quil_exp, quil_cis
from pyquil.quilbase import DefGate
