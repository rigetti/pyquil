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
import pytest

from pyquil.gates import *
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.parser import parse
from pyquil.quilatom import Addr
from pyquil.quilbase import Label, JumpTarget, Jump, JumpWhen, JumpUnless, DefGate, Qubit, Pragma


def test_simple_gate():
    _test("A 0", Gate("A", [], [Qubit(0)]))
    _test("A 1 10 100", Gate("A", [], [Qubit(1), Qubit(10), Qubit(100)]))


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


def test_def_gate_with_variables():
    # Note that technically the RX gate includes -i instead of just i but this messes a bit with the test since
    # it's not smart enough to figure out that -1*i == -i
    theta = Parameter('theta')
    rx = np.array([[quil_cos(theta / 2), 1j * quil_sin(theta / 2)],
                   [1j * quil_sin(theta / 2), quil_cos(theta / 2)]])

    defgate = 'DEFGATE RX(%theta):\n' \
              '    cos(%theta/2), i*sin(%theta/2)\n' \
              '    i*sin(%theta/2), cos(%theta/2)\n\n'

    _test(defgate, DefGate('RX', rx, [theta]))


def test_parameters():
    _test("RX(123) 0", RX(123)(0))
    _test("CPHASE00(0) 0 1", CPHASE00(0)(0, 1))
    _test("A(8,9) 0", Gate("A", [8, 9], [Qubit(0)]))
    _test("A(8, 9) 0", Gate("A", [8, 9], [Qubit(0)]))


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
    _expr("3.5919865395417361e-05", 3.5919865395417361e-05)
    _expr("3.5919865395417361E-05", 3.5919865395417361e-05)

    # Complex
    _expr("123.456i", complex(0, 123.456))
    _expr("+123.456i", complex(0, 123.456))
    # Edge case: making the whole complex number negative makes the real part -0.0
    _expr("-123.456i", complex(-0., -123.456))
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
    _expr("-pi / 2", np.pi / -2)

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
    _expr("exp(0)", 1.0)
    _expr("cis(0)", complex(1, 0))

    # Unary precedence
    # https://github.com/rigetticomputing/pyquil/issues/246
    _expr("-3+4", 1)
    _expr("-(3+4)", -7)
    _expr("-(3-4)", 1)
    _expr("-0.1423778799706841+0.5434363975682295i", complex(-0.1423778799706841, 0.5434363975682295))


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
    _test('PRAGMA gate_time H "10 ns"', Pragma('gate_time', ['H'], '10 ns'))
    _test('PRAGMA qubit 0', Pragma('qubit', [0]))
    _test('PRAGMA NO-NOISE', Pragma('NO-NOISE'))


def test_invalid():
    with pytest.raises(RuntimeError):
        parse("H X")


def test_empty_program():
    _test("")


def test_comments():
    parse('#Comment')
    parse('#Comment\n#more comments')
    parse('#Comment\n  #more comments')
    parse('#Comment\n    #more comments')
    parse('TRUE     [0]      #more comments')
    _test("TRUE     [0]\n    # Tabbed comment", TRUE(0))
    parse("DEFCIRCUIT FOO:\n    #Comment \n    TRUE [0]\n")


def test_extra_spaces():
    _test("TRUE     [0]", TRUE(0))
    _test("TRUE     [0]\n    # Tabbed comment", TRUE(0))
    parse("DEFCIRCUIT FOO:\n    TRUE [0]")
    parse("DEFCIRCUIT FOO:\n    #Comment \n    TRUE [0]\n")
    parse("DEFCIRCUIT FOO:\n             \n    TRUE [0]\n")  # Blank line


def _test(quil_string, *instructions):
    assert list(instructions) == parse(quil_string)
