
from functools import lru_cache
from typing import Callable

from lark import Lark, Token, Transformer, v_args

PAULI_GRAMMAR = """

?start: operator_term
      | coefficient "*" operator_term -> op_term_with_coefficient

?operator_term: operator_with_index
             | "I"                 -> op_i

?operator_with_index: operator_taking_index INT -> op_with_index

?operator_taking_index: "X"        -> op_x
                      | "Y"        -> op_y
                      | "Z"        -> op_z
                      
?coefficient: NUMBER
            | complex -> to_complex

?complex: "(" NUMBER "+" NUMBER "j" ")"

%import common.INT
%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""


@v_args(inline=True)
class PauliTree(Transformer):
    """ The imports here are lazy to prevent cyclic import issues """

    def __init__(self):
        self.vars = {}

    def op_x(self):
        from pyquil.paulis import sX
        return sX

    def op_y(self):
        from pyquil.paulis import sY
        return sY

    def op_z(self):
        from pyquil.paulis import sZ
        return sZ

    def op_i(self):
        from pyquil.paulis import sI
        return sI(0)

    def op_with_index(self, op: Callable, index: Token):
        return op(int(index.value))

    def op_term_with_coefficient(self, coeff, op):
        coeff = coeff if isinstance(coeff, complex) else float(coeff.value)
        return coeff * op

    def to_complex(self, *args):
        assert len(args[0].children) == 2, "Parsing error"
        real, imag = args[0].children
        return float(real.value) + float(imag.value) * 1j


@lru_cache(maxsize=None)
def pauli_parser() -> Lark:
    """
    This returns the parser object for Pauli compact string
    parsing, however it will only ever instantiate one parser
    per python process, and will re-use it for all subsequent
    calls to `from_compact_str`.

    :return: An instance of a Lark parser for Pauli strings
    """
    return Lark(PAULI_GRAMMAR, parser='lalr', transformer=PauliTree())


def parse_pauli_str(data: str):
    """
    Examples of Pauli Strings:

    => (1.5 + 0.5j)*X0*Z2+.7*Z1
    => "(1.5 + 0.5j)*X0*Z2+.7*I"

    A Pauli Term is a product of Pauli operators operating on
    different qubits - the operator can be one of "X", "Y", "Z", "I",
    including an index (ie. the qubit index such as 0, 1 or 2) and
    the coefficient multiplying the operator, eg. `1.5 * Z1`.

    Note: "X", "Y" and "Z" are always followed by the qubit index,
          but "I" being the identity is not.

    So we need to support
    """
    parser = pauli_parser()
    return parser.parse(data)


if __name__ == '__main__':
    from pyquil.paulis import sI, sX, sY, sZ
    assert parse_pauli_str("I") == sI(0)
    assert parse_pauli_str("Z1") == sZ(1)
    assert parse_pauli_str("Z1") == (1.0 + 0j) * sZ(1)
