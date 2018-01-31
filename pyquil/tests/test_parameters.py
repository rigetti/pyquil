from math import pi
import numpy as np
from pyquil.parameters import (Parameter, sin, _contained_parameters, format_parameter, Expression,
                               sqrt, cis, exp, cos)


def test_format_parameter():
    test_cases = [
        (1, '1'),
        (1.0, '1.0'),

        (1j, 'i'),
        (0 + 1j, 'i'),
        (-1j, '-i'),
    ]

    for test_case in test_cases:
        assert format_parameter(test_case[0]) == test_case[1]


# https://github.com/rigetticomputing/pyquil/issues/184
def test_pretty_print_pi():
    test_cases = [
        (0., '0'),
        (pi, 'pi'),
        (-pi, '-pi'),
        (2 * pi / 3., '2*pi/3'),
        (pi / 9, '0.3490658503988659'),
        (pi / 8, 'pi/8'),
        (-90 * pi / 2, '-45*pi'),
    ]

    for test_case in test_cases:
        assert format_parameter(test_case[0]) == test_case[1]


def test_expression_to_string():
    x = Parameter('x')
    assert str(x) == '%x'

    y = Parameter('y')
    assert str(y) == '%y'

    assert str(x + y) == '%x+%y'
    assert str(3 * x + y) == '3*%x+%y'
    assert str(3 * (x + y)) == '3*(%x+%y)'

    assert str(x + y + 2) == '%x+%y+2'
    assert str(x - y - 2) == '%x-%y-2'
    assert str(x - (y - 2)) == '%x-(%y-2)'

    assert str((x + y) - 2) == '%x+%y-2'
    assert str(x + (y - 2)) == '%x+%y-2'

    assert str(x ** y ** 2) == '%x^%y^2'
    assert str(x ** (y ** 2)) == '%x^%y^2'
    assert str((x ** y) ** 2) == '(%x^%y)^2'

    assert str(sin(x)) == 'sin(%x)'
    assert str(3 * sin(x + y)) == '3*sin(%x+%y)'


def test_contained_parameters():
    x = Parameter('x')
    assert _contained_parameters(x) == {x}

    y = Parameter('y')
    assert _contained_parameters(x + y) == {x, y}

    assert _contained_parameters(x ** y ** sin(x * y * 4)) == {x, y}


def test_eval():
    x = Parameter('x')
    assert Expression.eval(x, {x: 5}) == 5

    y = Parameter('y')
    assert Expression.eval(x + y, {x: 5, y: 6}) == 11
    assert Expression.eval(x + y, {x: 5}) == 5 + y
    assert Expression.eval(exp(x), {y: 5}) != exp(5)
    assert Expression.eval(exp(x), {x: 5}) == np.exp(5)

    assert np.isclose(Expression.eval(sin(x * x ** 2 / y), {x: 5.0, y: 10.0}), np.sin(12.5))
    assert np.isclose(Expression.eval(sqrt(x), {x: 5.0, y: 10.0}), np.sqrt(5.0))
    assert np.isclose(Expression.eval(cis(x), {x: 5.0, y: 10.0}), np.exp(1j * 5.0))
    assert np.isclose(Expression.eval(x - y, {x: 5.0, y: 10.0}), -5.)

    assert Expression.eval(cis(x), {y: 5}) == cis(x)
    assert np.allclose(Expression.eval_array([sin(x), cos(x)], {x: 5}), [np.sin(5), np.cos(5)])
