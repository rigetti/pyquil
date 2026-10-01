import numpy as np

from pyquil.quilatom import (
    Parameter,
    quil_cis,
    quil_cos,
    quil_exp,
    quil_sin,
    quil_sqrt,
    substitute,
    substitute_array,
)


def test_expression_to_string():
    x = Parameter("x")
    assert str(x) == "%x"

    y = Parameter("y")
    assert str(y) == "%y"

    # NOTE: quil-rs's `Expression` formatting differs cosmetically from legacy pyquil in several
    # ways (no spaces around `+`/`*`/`^`, added disambiguating parentheses, lowercase function
    # names). This is a deliberate, accepted breaking change -- see MIGRATION-NOTES.md.
    assert str(x + y) == "%x+%y"
    assert str(3 * x + y) == "(3*%x)+%y"
    assert str(3 * (x + y)) == "3*(%x+%y)"

    assert str(x + y + 2) == "(%x+%y)+2"
    assert str(x - y - 2) == "(%x - %y) - 2"
    assert str(x - (y - 2)) == "%x - (%y - 2)"

    assert str((x + y) - 2) == "(%x+%y) - 2"
    assert str(x + (y - 2)) == "%x+(%y - 2)"

    assert str(x**y**2) == "%x^(%y^2)"
    assert str(x ** (y**2)) == "%x^(%y^2)"
    assert str((x**y) ** 2) == "(%x^%y)^2"

    assert str(quil_sin(x)) == "sin(%x)"
    assert str(3 * quil_sin(x + y)) == "3*sin(%x+%y)"
    assert (
        str(quil_exp(-1j * x / 2) * np.exp(1j * np.pi / 4))
        == "exp((-1.0i*%x)/2)*0.7071067811865476+0.7071067811865475i"
    )


def test_eval():
    x = Parameter("x")
    assert substitute(x, {x: 5}) == 5

    y = Parameter("y")
    assert substitute(x + y, {x: 5, y: 6}) == 11
    assert substitute(x + y, {x: 5}) == 5 + y
    assert substitute(quil_exp(x), {y: 5}) != np.exp(5)
    assert substitute(quil_exp(x), {x: 5}) == np.exp(5)

    assert np.isclose(substitute(quil_sin(x * x**2 / y), {x: 5.0, y: 10.0}), np.sin(12.5))
    assert np.isclose(substitute(quil_sqrt(x), {x: 5.0, y: 10.0}), np.sqrt(5.0))
    assert np.isclose(substitute(quil_cis(x), {x: 5.0, y: 10.0}), np.exp(1j * 5.0))
    assert np.isclose(substitute(x - y, {x: 5.0, y: 10.0}), -5.0)

    assert substitute(quil_cis(x), {y: 5}) == quil_cis(x)
    assert np.allclose(substitute_array([quil_sin(x), quil_cos(x)], {x: 5}), [np.sin(5), np.cos(5)])
