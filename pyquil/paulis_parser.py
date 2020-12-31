from functools import lru_cache
from typing import Callable

from lark import Lark, Token, Transformer, v_args

from pyquil.paulis import PauliTerm, sI, sX, sY, sZ


PAULI_GRAMMAR = r"""
?start: pauli_term
      | start "-" start -> pauli_sub_pauli
      | start "+" start -> pauli_add_pauli

?pauli_term: operator_term
           | coefficient "*" pauli_term -> op_term_with_coefficient
           | coefficient pauli_term -> op_term_with_coefficient
           | pauli_term "*" coefficient -> coefficient_with_op_term
           | pauli_term "*" pauli_term -> op_term_with_op_term
           | pauli_term pauli_term -> op_term_with_op_term

?operator_term: operator_with_index
              | "I"                 -> op_i

?operator_with_index: operator_taking_index INT -> op_with_index

?operator_taking_index: "X"        -> op_x
                      | "Y"        -> op_y
                      | "Z"        -> op_z

?coefficient: NUMBER
            | complex -> to_complex

?complex: "(" SIGNED_NUMBER "+" NUMBER "j" ")"

%import common.INT
%import common.SIGNED_NUMBER
%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE

"""


@v_args(inline=True)
class PauliTree(Transformer):  # type: ignore
    """ The imports here are lazy to prevent cyclic import issues """

    def op_x(self) -> Callable[[int], PauliTerm]:
        return sX

    def op_y(self) -> Callable[[int], PauliTerm]:
        return sY

    def op_z(self) -> Callable[[int], PauliTerm]:
        return sZ

    def op_i(self) -> PauliTerm:
        return sI()

    def op_with_index(self, op: Callable, index: Token) -> PauliTerm:
        return op(int(index.value))

    def op_term_with_coefficient(self, coeff, op) -> PauliTerm:
        coeff = coeff if isinstance(coeff, complex) else float(coeff.value)
        return coeff * op

    def coefficient_with_op_term(self, op, coeff) -> PauliTerm:
        # This shouldn't be necessary, the grammar should take care
        # of it.
        return self.op_term_with_coefficient(coeff, op)

    def op_term_with_op_term(self, first, second) -> PauliTerm:
        return first * second

    def to_complex(self, *args) -> complex:
        assert len(args[0].children) == 2, "Parsing error"
        real, imag = args[0].children
        return float(real.value) + float(imag.value) * 1j

    def pauli_mul_pauli(self, first, second) -> PauliTerm:
        return first * second

    def pauli_sub_pauli(self, first, second) -> PauliTerm:
        return first - second

    def pauli_add_pauli(self, first, second) -> PauliTerm:
        return first + second


@lru_cache(maxsize=None)
def pauli_parser() -> Lark:
    """
    This returns the parser object for Pauli compact string
    parsing, however it will only ever instantiate one parser
    per python process, and will re-use it for all subsequent
    calls to `from_compact_str`.

    :return: An instance of a Lark parser for Pauli strings
    """
    return Lark(PAULI_GRAMMAR, parser="lalr", transformer=PauliTree())


def parse_pauli_str(data: str) -> PauliTerm:
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
    """
    parser = pauli_parser()
    return parser.parse(data)
