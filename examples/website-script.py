#!/usr/bin/env python

"""
A script appropriate for featuring on rigetti.com to show how easy it is to get started!

The website should be updated to use this idiomatic program once Forest 2 is released.

 - new imports
 - don't use a wavefunction-based method, as it won't work on a QPU
"""

from pyquil import Program, get_qc
from pyquil.gates import H, CNOT

# construct a Bell State program
p = Program(H(0), CNOT(0, 1))
# run the program on a QVM
qvm = get_qc("9q-generic-qvm")
result = qvm.run_and_measure(p, trials=10)
print(result)
