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
import numpy as np
import sys

import pytest

from pyquil.gates import *
from pyquil.parser import parse
from pyquil.quilbase import Label, JumpTarget, Jump, JumpWhen, JumpUnless, RawInstr, DefGate


def test_simple_gate():
    _test("A 0", Gate("A", [], [DirectQubit(0)]))
    _test("A 1 10 100", Gate("A", [], [DirectQubit(1), DirectQubit(10), DirectQubit(100)]))


def test_standard_gates():
    _test("H 0", H(0))
    _test("CNOT 0 1", CNOT(0, 1))
    _test("SWAP 0 1", SWAP(0, 1))


def test_def_gate():
    sqrt_x = DefGate("SQRT-X", np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                                         [0.5 - 0.5j, 0.5 + 0.5j]]))
    hadamard = DefGate("HADAMARD", np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                                             [1 / np.sqrt(2), -1 / np.sqrt(2)]]))
    defgates = """
DEFGATE SQRT-X:
    0.5+0.5i, 0.5-0.5i
    0.5-0.5i, 0.5+0.5i

DEFGATE HADAMARD:
    1/sqrt(2), 1/sqrt(2)
    1/sqrt(2), -1/sqrt(2)
    """.strip()

    _test(defgates, sqrt_x, hadamard)


def test_parameters():
    _test("RX(123) 0", RX(123)(0))
    _test("CPHASE00(0) 0 1", CPHASE00(0)(0, 1))
    _test("A(8,9) 0", Gate("A", [8, 9], [DirectQubit(0)]))
    _test("A(8, 9) 0", Gate("A", [8, 9], [DirectQubit(0)]))


def test_expressions():
    # Test expressions by wrapping them in an RX gate for convenience
    def _expr(expression, expected):
        _test("RX(" + expression + ") 0", RX(expected)(0))

    # Decimals
    _expr("+123", 123)
    _expr("-123", -123)
    _expr("123.456", 123.456)
    _expr("+123.456", 123.456)
    _expr("-123.456", -123.456)

    # Exponential
    _expr("1e3", 1000.0)
    _expr("1.5e2", 150.0)

    # Complex
    _expr("123.456i", complex(0, 123.456))
    _expr("+123.456i", complex(0, 123.456))
    _expr("-123.456i", complex(0, -123.456))
    _expr("777+123.456i", complex(777, 123.456))
    _expr("777-123.456i", complex(777, -123.456))
    _expr("+777-123.456i", complex(777, -123.456))

    # Imaginary
    _expr("i * 2", complex(0, 2))
    _expr("2i", complex(0, 2))
    _expr("1 ^ 2", 1)

    # Pi
    _expr("pi", np.pi)
    _expr("pi / 2", np.pi / 2)

    # Expressions
    _expr("1+2", 3)
    _expr("1-2", -1)
    _expr("3*4", 12)
    _expr("6/2", 3.0)
    _expr("2^3", 8)

    # Order of operations
    _expr("3 + 6 * (5 + 4) / 3 - 7", 14.0)
    _expr("3 ^ 2 + 5", 14)

    # Functions
    _expr("sin(0)", 0.0)
    _expr("cos(0)", 1.0)
    _expr("sqrt(4)", 2.0)
    _expr("cis(0)", complex(1, 0))


def test_measure():
    _test("MEASURE 0", MEASURE(0))
    _test("MEASURE 0 [1]", MEASURE(0, 1))


def test_jumps():
    _test("LABEL @test_1", JumpTarget(Label("test_1")))
    _test("JUMP @test_1", Jump(Label("test_1")))
    _test("JUMP-WHEN @test_1 [0]", JumpWhen(Label("test_1"), Addr(0)))
    _test("JUMP-UNLESS @test_1 [1]", JumpUnless(Label("test_1"), Addr(1)))


def test_others():
    _test("RESET", RESET)
    _test("WAIT", WAIT)
    _test("NOP", NOP)


def test_classical():
    _test("TRUE [0]", TRUE(0))
    _test("FALSE [0]", FALSE(0))
    _test("NOT [0]", NOT(0))
    _test("AND [0] [1]", AND(0, 1))
    _test("OR [0] [1]", OR(0, 1))
    _test("MOVE [0] [1]", MOVE(0, 1))
    _test("EXCHANGE [0] [1]", EXCHANGE(0, 1))


def test_pragma():
    _test('PRAGMA gate_time H "10 ns"', RawInstr('PRAGMA gate_time H "10 ns"'))
    _test('PRAGMA qubit 0', RawInstr('PRAGMA qubit 0'))


def test_invalid():
    with pytest.raises(RuntimeError):
        parse("H X")


def _test(quil_string, *instructions):
    # Currently doesn't support Python 2
    if sys.version_info.major == 2:
        return
    assert parse(quil_string) == list(instructions)
