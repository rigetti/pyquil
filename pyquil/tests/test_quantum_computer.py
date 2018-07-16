from pyquil import Program
from pyquil.gates import *

from pyquil.api.quantum_computer import _get_flipped_protoquil_program


def test_get_flipped_program():
    program = Program([
        I(0),
        RX(2.3, 1),
        CNOT(0, 1),
        MEASURE(0, 0),
        MEASURE(1, 1),
    ])

    flipped_program = _get_flipped_protoquil_program(program)
    assert flipped_program.out().splitlines()[-6::] == [
        'PRAGMA PRESERVE_BLOCK',
        'RX(pi) 0',
        'RX(pi) 1',
        'PRAGMA END_PRESERVE_BLOCK',
        'MEASURE 0 [0]',
        'MEASURE 1 [1]',
    ]


def test_get_flipped_program_only_measure():
    program = Program([
        MEASURE(0, 0),
        MEASURE(1, 1),
    ])

    flipped_program = _get_flipped_protoquil_program(program)
    assert flipped_program.out().splitlines() == [
        'PRAGMA PRESERVE_BLOCK',
        'RX(pi) 0',
        'RX(pi) 1',
        'PRAGMA END_PRESERVE_BLOCK',
        'MEASURE 0 [0]',
        'MEASURE 1 [1]',
    ]
