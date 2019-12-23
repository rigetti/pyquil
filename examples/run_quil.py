#!/usr/bin/env python

"""
This module runs basic Quil text files against the Forest QVM API.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyquil import Program, get_qc

help_string = (
    "Script takes two arguments. Quil program filename is required as the first "
    "argument and number of classical registers to return is the optional second "
    "argument (defaults to 8)."
)


def parse():
    parser = ArgumentParser(__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath", help="Quil program filename")
    parser.add_argument(
        "--qubit-num", "-q", metavar="Q", default=8, type=int, help="Number of qubits to use."
    )
    args = parser.parse_args()
    main(args.filepath, args.qubit_num)


def main(filepath, qubit_num):
    with open(filepath) as file:
        quil_prog = file.read()
        program = Program(quil_prog)

    qc = get_qc(f"{qubit_num}q-qvm")

    print("Running Quil Program from: ", filepath)
    print("---------------------------")
    print("Output: ")
    print(qc.run(program))


if __name__ == "__main__":
    parse()
