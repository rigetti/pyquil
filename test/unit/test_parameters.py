from math import pi

import numpy as np

from pyquil.quilatom import (
    Parameter,
    quil_sin,
    quil_cos,
    quil_sqrt,
    quil_exp,
    _contained_parameters,
    format_parameter,
    quil_cis,
    substitute,
    substitute_array,
)


def test_format_parameter():
    test_cases = [
        (1, "1"),
        (1.0, "1.0"),
        (1j, "i"),
        (0 + 1j, "i"),
        (-1j, "-i"),
        (1e-15 + 1j, "i"),
        (1e-15 - 1j, "-i"),
    ]

    for test_case in test_cases:
        assert format_parameter(test_case[0]) == test_case[1]


# https://github.com/rigetti/pyquil/issues/184
def test_pretty_print_pi():
    test_cases = [
        (0.0, "0"),
        (pi, "pi"),
        (-pi, "-pi"),
        (2 * pi / 3.0, "2*pi/3"),
        (pi / 9, "0.3490658503988659"),
        (pi / 8, "pi/8"),
        (-90 * pi / 2, "-45*pi"),
    ]

    for test_case in test_cases:
        assert format_parameter(test_case[0]) == test_case[1]


def test_expression_to_string():
    x = Parameter("x")
    assert str(x) == "%x"

    y = Parameter("y")
    assert str(y) == "%y"

    assert str(x + y) == "%x + %y"
    assert str(3 * x + y) == "3*%x + %y"
    assert str(3 * (x + y)) == "3*(%x + %y)"

    assert str(x + y + 2) == "%x + %y + 2"
    assert str(x - y - 2) == "%x - %y - 2"
    assert str(x - (y - 2)) == "%x - (%y - 2)"

    assert str((x + y) - 2) == "%x + %y - 2"
    assert str(x + (y - 2)) == "%x + %y - 2"

    assert str(x ** y ** 2) == "%x^%y^2"
    assert str(x ** (y ** 2)) == "%x^%y^2"
    assert str((x ** y) ** 2) == "(%x^%y)^2"

    assert str(quil_sin(x)) == "SIN(%x)"
    assert str(3 * quil_sin(x + y)) == "3*SIN(%x + %y)"


def test_contained_parameters():
    x = Parameter("x")
    assert _contained_parameters(x) == {x}

    y = Parameter("y")
    assert _contained_parameters(x + y) == {x, y}

    assert _contained_parameters(x ** y ** quil_sin(x * y * 4)) == {x, y}


def test_eval():
    x = Parameter("x")
    assert substitute(x, {x: 5}) == 5

    y = Parameter("y")
    assert substitute(x + y, {x: 5, y: 6}) == 11
    assert substitute(x + y, {x: 5}) == 5 + y
    assert substitute(quil_exp(x), {y: 5}) != np.exp(5)
    assert substitute(quil_exp(x), {x: 5}) == np.exp(5)

    assert np.isclose(substitute(quil_sin(x * x ** 2 / y), {x: 5.0, y: 10.0}), np.sin(12.5))
    assert np.isclose(substitute(quil_sqrt(x), {x: 5.0, y: 10.0}), np.sqrt(5.0))
    assert np.isclose(substitute(quil_cis(x), {x: 5.0, y: 10.0}), np.exp(1j * 5.0))
    assert np.isclose(substitute(x - y, {x: 5.0, y: 10.0}), -5.0)

    assert substitute(quil_cis(x), {y: 5}) == quil_cis(x)
    assert np.allclose(substitute_array([quil_sin(x), quil_cos(x)], {x: 5}), [np.sin(5), np.cos(5)])
