#!/usr/bin/env python

"""
A script appropriate for featuring on rigetti.com to show how easy it is to get started!

This version exists on the website as of September, 2018. It should continue working
even if PyQuil changes.
"""

from pyquil.quil import Program
from pyquil.gates import H, CNOT
from pyquil.api import QVMConnection

# construct a Bell State program
p = Program(H(0), CNOT(0, 1))
# run the program on a QVM
qvm = QVMConnection()
result = qvm.wavefunction(p)
print(result)
