from lark import UnexpectedCharacters, UnexpectedToken
from pytest import raises

from pyquil.paulis import (
    sI,
    sX,
    sY,
    sZ,
)
from pyquil.paulis_parser import parse_pauli_str


def test_paulis_parser_basic_sanity():
    pass


def test_complex_number_parsing():
    assert parse_pauli_str("(1+0j) * X1") == (1.0 + 0j) * sX(1)
    assert parse_pauli_str("(1.1 + 0.1j) * Z2") == (1.1 + 0.1j) * sZ(2)
    assert parse_pauli_str("(0 + 1j) * Y1") == (0 + 1j) * sY(1)

    with raises(UnexpectedCharacters, match="Expecting:"):
        # If someone uses 'i' instead of 'j' we get a useful message
        # in an UnexpectedToken exception stating what's acceptable
        parse_pauli_str("(1 + 0i) * X1")


def test_pauli_terms_parsing():
    # A PauliTerm consists of: operator, index, coefficient,
    # where the index and coefficient are sometimes optional
    # Eg. in the simplest case we just have I, which is fine
    assert parse_pauli_str("I") == sI(0)

    # ...but just having the operator without an index is
    # *not* ok for X, Y or Z...
    with raises(UnexpectedToken):
        parse_pauli_str("X")
    with raises(UnexpectedToken):
        parse_pauli_str("Y")
    with raises(UnexpectedToken):
        parse_pauli_str("Z")

    # ...these operators require an index to be included as well
    assert parse_pauli_str("X0") == sX(0)
    assert parse_pauli_str("X1") == sX(1)
    assert parse_pauli_str("Y0") == sY(0)
    assert parse_pauli_str("Y1") == sY(1)
    assert parse_pauli_str("Z0") == sZ(0)
    assert parse_pauli_str("Z1") == sZ(1)
    assert parse_pauli_str("Z2") == sZ(2)

    # The other optional item for a pauli term is the coefficient,
    # which in the simplest case could just be this:
    result = parse_pauli_str("1.5 * Z1")
    assert result == 1.5 * sZ(1)

    # the simple cases should also be the same as a complex coefficient
    # with 1. and 0j
    result = parse_pauli_str("Z1")
    assert result == (1.0 + 0j) * sZ(1)

    # we also need to support short-hand versions of floats like this:
    result = parse_pauli_str(".5 * Z0")
    assert result == 0.5 * sZ(0)

    # ...and just to check it parses the same without whitespace
    result = parse_pauli_str(".5*X0")
    assert result == 0.5 * sX(0)

    # TODO: do we want to support even shorter notation like ??
    # result = parse_pauli_str(".5X0")
    # assert result == 0.5 * sX(0)

    # Obviously the coefficients can also be complex, so we need to
    # support this:
    result = parse_pauli_str("(0 + 1j) * Z0")
    assert result == (0 + 1j) * sZ(0)

    result = parse_pauli_str("(1.0 + 0j) * X0")
    assert result == (1.0 + 0j) * sX(0)


def test_pauli_parser_parentheses():
    """ TODO: check that nested parentheses (ie. for coefficients) are
        parsed and executed in the correct order
    """
    pass


def test_pauli_sum_parsing():
    """ TODO: check that multiple PauliTerms are parsed and executed
        correctly
    """
    pass


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
