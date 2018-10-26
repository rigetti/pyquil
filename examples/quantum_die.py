#!/usr/bin/env python

##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import math
from functools import reduce
import pyquil
from pyquil import Program, get_qc
from pyquil.gates import H
from six.moves import range


def qubits_needed(number_of_sides):
    """
    The number of qubits needed for a die of n faces.
    """
    return int(math.ceil(math.log(number_of_sides, 2)))


def get_qvm(number_of_sides):
    """
    Get a QVM to simulate the requested number of sides.
    """
    return get_qc(f"{qubits_needed(number_of_sides)}q-qvm")


def die_program(number_of_sides):
    """
    Generate a quantum program to roll a die of n faces.
    """
    prog = Program()
    n_qubits = qubits_needed(number_of_sides)
    ro = prog.declare('ro', 'BIT', n_qubits)
    # Hadamard initialize.
    for q in range(qubits):
        prog.inst(H(q))
    # Measure everything.
    for q in range(qubits):
        prog.measure(q, ro[q])
    return prog


def process_results(results):
    """
    Convert n digit binary result from the QVM to a value on a die.
    """
    raw_results = results[0]
    processing_result = 0
    for each_qubit_measurement in raw_results:
        processing_result = 2*processing_result + each_qubit_measurement
    # Convert from 0 indexed to 1 indexed
    die_value = processing_result + 1
    return die_value


def roll_die(qvm, number_of_sides):
    """
    Roll an n-sided quantum die.
    """
    die_compiled = qvm.compile(die_program(number_of_sides))
    return process_results(qvm.run(die_compiled))


if __name__ == '__main__':
    number_of_sides = int(input("Please enter number of sides: "))
    qvm = get_qvm(number_of_sides)
    print(f"The result is: {roll_die(qvm, number_of_sides)}")
