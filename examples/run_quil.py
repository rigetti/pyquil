"""
This module run basic Quil text files agains the Forest QVM API. Usage
"""
import sys

from pyquil.quil import Program
import pyquil.forest as forest

qvm = forest.Connection()

help_string = "Script takes two arguments. Quil program filename is required as the first " \
              "argument and number of classical registers to return is the optional second " \
              "argument (defaults to 8)."

if __name__ == '__main__':
    assert len(sys.argv) == 2 or len(sys.argv) == 3, help_string
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print help_string
        sys.exit()
    filepath = str(sys.argv[1])
    if len(sys.argv) == 3:
        classical_register_num = int(sys.argv[2])
    else:
        classical_register_num = 8

    with open(filepath) as file:
        quil_prog = file.read()
        p = Program(quil_prog)

    print "Running Quil Program from: ", filepath
    print "---------------------------"
    print "Output: "
    print qvm.run(p, range(classical_register_num))
