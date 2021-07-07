from functools import lru_cache
from typing import Callable, Tuple, Union

from lark import Lark, Token, Transformer, Tree, v_args

from pyquil.paulis import PauliSum, PauliTerm, sI, sX, sY, sZ


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
    """ An AST Transformer to convert the given string into a tree """

    def op_x(self) -> Callable[[int], PauliTerm]:
        return sX

    def op_y(self) -> Callable[[int], PauliTerm]:
        return sY

    def op_z(self) -> Callable[[int], PauliTerm]:
        return sZ

    def op_i(self) -> PauliTerm:
        return sI()

    def op_with_index(self, op: Callable[[int], PauliTerm], index: Token) -> PauliTerm:
        return op(int(index.value))

    def op_term_with_coefficient(self, coeff: Union[complex, Tree], op: PauliTerm) -> PauliTerm:
        coeff = coeff if isinstance(coeff, complex) else float(coeff.value)
        return coeff * op

    def coefficient_with_op_term(self, op: PauliTerm, coeff: Union[complex, Tree]) -> PauliTerm:
        return self.op_term_with_coefficient(coeff, op)

    def op_term_with_op_term(self, first: PauliTerm, second: PauliTerm) -> PauliTerm:
        return first * second

    def to_complex(self, *args: Tuple[Tree, Tree]) -> complex:
        assert len(args[0].children) == 2, "Parsing error"
        real, imag = args[0].children
        return float(real.value) + float(imag.value) * 1j

    def pauli_mul_pauli(self, first: PauliTerm, second: PauliTerm) -> Union[PauliTerm, PauliSum]:
        return first * second

    def pauli_add_pauli(self, first: PauliTerm, second: PauliTerm) -> Union[PauliTerm, PauliSum]:
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


def parse_pauli_str(data: str) -> Union[Tree, PauliTerm]:
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
