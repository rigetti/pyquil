#!/usr/bin/env python

"""
This module runs basic Quil text files against the Forest QVM API.
"""

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyquil.quil import Program
from pyquil import api

qvm = api.QVMConnection()

help_string = "Script takes two arguments. Quil program filename is required as the first " \
              "argument and number of classical registers to return is the optional second " \
              "argument (defaults to 8)."


def parse():
    parser = ArgumentParser(__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('filepath', help='Quil program filename')
    parser.add_argument('--classical-register-num', '-n', metavar='N', default=8, type=int,
                        help="Number of classical registers to return.")
    args = parser.parse_args()
    main(args.filepath, args.classical_register_num)


def main(filepath, classical_register_num):
    with open(filepath) as file:
        quil_prog = file.read()
        program = Program(quil_prog)

    print("Running Quil Program from: ", filepath)
    print("---------------------------")
    print("Output: ")
    print(qvm.run(program, list(range(classical_register_num))))


if __name__ == '__main__':
    parse()
