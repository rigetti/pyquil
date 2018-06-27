#!/usr/bin/python
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

from pyquil.quil import Program
from pyquil.gates import RX, RY, Z
from pyquil.parametric import ParametricProgram, parametric
from pyquil.slot import Slot
from pyquil.paulis import PauliTerm, exponential_map
import pytest


def test_slot_algebra():
    # slot algebra
    x = Slot()
    expression = (2 * x) + 1
    assert isinstance(expression, Slot)
    assert expression.value() == 1.0
    x._value = 4
    assert expression.value() == 9.0

    x._value = 0.0
    expression = 1 + (x * 2)
    assert isinstance(expression, Slot)
    assert expression.value() == 1.0
    x._value = 4
    assert expression.value() == 9.0

    # slot slot algebra
    x._value = 0
    y = Slot(2)
    expression = x * y
    assert isinstance(expression, Slot)
    assert expression.value() == 0.0
    x._value = 2.0
    assert expression.value() == 4.0

    x._value = 4.0
    expression = x / y
    assert expression.value() == 2.0
    expression = x / 2.0
    assert expression.value() == 2.0
    expression = y / x
    assert expression.value() == 0.5
    expression = 2.0 / x
    assert expression.value() == 0.5

    expression = x - y
    assert expression.value() == 2.0
    expression = 4 - y
    assert expression.value() == 2.0

    assert max(x, y) == 4.0
    assert -x == -4.0

    z = Slot(-2)
    assert abs(z) == 2


def test_slot_logic():
    x = Slot()
    assert x < 2
    x._value = 2.1
    assert not x < 2
    x._value = 2.0
    assert x <= 2.0
    assert x == 2.0
    assert x >= 2
    x._value = 3.0
    assert x > 2
    assert x != 2.0

    y = Slot()
    assert x != y
    y._value = 3.0
    assert x == y
    assert x >= y
    assert x <= y
    y._value = 2.1
    assert not x < y
    assert not y > x


def test_pp():
    def rx(alpha):
        p = Program()
        p += RX(alpha, 0)
        return p

    pp = ParametricProgram(rx)

    assert "RX(1.0) 0\n" == pp(1.0).out()
    assert "RX(2.0) 0\n" == pp(2.0).out()


def test_parametric_decorator():
    @parametric
    def rx(alpha):
        p = Program()
        p += RX(alpha, 0)
        return p

    assert "RX(1.0) 0\n" == rx(1.0).out()
    assert "RX(2.0) 0\n" == rx(2.0).out()


def test_bad_parametric_functions():
    with pytest.raises(TypeError):
        @parametric
        def bad(alpha):
            return 5

    with pytest.raises(RuntimeError):
        @parametric
        def bad(*varargs_arent_allowed):
            return Program()


def test_fuse():
    @parametric
    def rx(alpha):
        p = Program()
        p += RX(alpha, 0)
        return p

    @parametric
    def ry(beta):
        p = Program()
        p += RY(beta, 1)
        return p

    z = Program().inst(Z(2))
    fused = rx.fuse(ry).fuse(z)

    assert fused(1.0, 2.0).out() == "RX(1.0) 0\nRY(2.0) 1\nZ 2\n"


def test_parametric_arith():
    @parametric
    def rx(alpha):
        p = Program()
        p += RX(2.0 * alpha + 1, 0)
        return p

    assert "RX(3.0) 0\n" == rx(1.0).out()
    assert "RX(5.0) 0\n" == rx(2.0).out()


def test_exponentiate_paraprog():
    xterm = PauliTerm("X", 2) * PauliTerm("X", 1)
    paraprog = exponential_map(xterm)
    prog = paraprog(1)
    assert prog.out() == ("H 2\n"
                          "H 1\n"
                          "CNOT 2 1\n"
                          "RZ(2.0) 1\n"
                          "CNOT 2 1\n"
                          "H 2\n"
                          "H 1\n")
