##############################################################################
# Copyright 2017-2018 Rigetti Computing
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

from fractions import Fraction
from numbers import Number
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Iterable,
    Mapping,
    NoReturn,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
    cast,
)
from typing_extensions import Self
from deprecation import deprecated

import numpy as np

import quil.instructions as quil_rs
import quil.expression as quil_rs_expr

from pyquil._version import pyquil_version


class QuilAtom(object):
    """
    Abstract class for atomic elements of Quil.
    """

    def out(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        raise NotImplementedError()


class Qubit(QuilAtom):
    """
    Representation of a qubit.

    :param index: Index of the qubit.
    """

    def __init__(self, index: int):
        if not (isinstance(index, int) and index >= 0):
            raise TypeError("Addr index must be a non-negative int")
        self.index = index

    def out(self) -> str:
        return str(self.index)

    def __str__(self) -> str:
        return str(self.index)

    def __repr__(self) -> str:
        return "<Qubit {0}>".format(self.index)

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Qubit) and other.index == self.index


class FormalArgument(QuilAtom):
    """
    Representation of a formal argument associated with a DEFCIRCUIT or DEFGATE ... AS PAULI-SUM
    or DEFCAL form.
    """

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Formal arguments must be named by a string.")
        self.name = name

    def out(self) -> str:
        return str(self)

    @property
    def index(self) -> NoReturn:
        raise RuntimeError(f"Cannot derive an index from a FormalArgument {self}")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<FormalArgument {self.name}>"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FormalArgument) and other.name == self.name


class QubitPlaceholder(QuilAtom):
    def out(self) -> str:
        raise RuntimeError("Qubit {} has not been assigned an index".format(self))

    @property
    def index(self) -> NoReturn:
        raise RuntimeError("Qubit {} has not been assigned an index".format(self))

    def __str__(self) -> str:
        return "q{}".format(id(self))

    def __repr__(self) -> str:
        return "<QubitPlaceholder {}>".format(id(self))

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QubitPlaceholder) and id(other) == id(self)

    @classmethod
    def register(cls, n: int) -> List["QubitPlaceholder"]:
        """Return a 'register' of ``n`` QubitPlaceholders.

        >>> qs = QubitPlaceholder.register(8) # a qubyte
        >>> prog = Program(H(q) for q in qs)
        >>> address_qubits(prog).out()
        H 0
        H 1
        ...
        >>>

        The returned register is a Python list of QubitPlaceholder objects, so all
        normal list semantics apply.

        :param n: The number of qubits in the register
        """
        return [cls() for _ in range(n)]


QubitDesignator = Union[Qubit, QubitPlaceholder, FormalArgument, int]


def _convert_to_rs_qubit(qubit: Union[QubitDesignator, quil_rs.Qubit]) -> quil_rs.Qubit:
    if isinstance(qubit, quil_rs.Qubit):
        return qubit
    if isinstance(qubit, Qubit):
        return quil_rs.Qubit.from_fixed(qubit.index)
    if isinstance(qubit, QubitPlaceholder):
        raise NotImplementedError("QubitPlaceholders aren't implemented in quil-rs")
    if isinstance(qubit, FormalArgument):
        return quil_rs.Qubit.from_variable(qubit.name)
    if isinstance(qubit, int):
        return quil_rs.Qubit.from_fixed(qubit)
    raise ValueError(f"{type(qubit)} is not a valid QubitDesignator")


def _convert_to_rs_qubits(qubits: Iterable[QubitDesignator]) -> List[quil_rs.Qubit]:
    return [_convert_to_rs_qubit(qubit) for qubit in qubits]


def _convert_to_py_qubit(qubit: Union[QubitDesignator, quil_rs.Qubit]) -> QubitDesignator:
    if isinstance(qubit, quil_rs.Qubit):
        if qubit.is_fixed():
            return Qubit(qubit.to_fixed())
        if qubit.is_variable():
            return FormalArgument(qubit.to_variable())
    if isinstance(qubit, (Qubit, QubitPlaceholder, FormalArgument, Parameter, int)):
        return qubit
    raise ValueError(f"{type(qubit)} is not a valid QubitDesignator")


def _convert_to_py_qubits(qubits: Iterable[Union[QubitDesignator, quil_rs.Qubit]]) -> List[QubitDesignator]:
    return [_convert_to_py_qubit(qubit) for qubit in qubits]


def unpack_qubit(qubit: Union[QubitDesignator, FormalArgument]) -> Union[Qubit, QubitPlaceholder, FormalArgument]:
    """
    Get a qubit from an object.

    :param qubit: the qubit designator to unpack.
    :return: A Qubit or QubitPlaceholder instance
    """
    if isinstance(qubit, int):
        return Qubit(qubit)
    elif isinstance(qubit, Qubit):
        return qubit
    elif isinstance(qubit, QubitPlaceholder):
        return qubit
    elif isinstance(qubit, FormalArgument):
        return qubit
    else:
        raise TypeError("qubit should be an int or Qubit or QubitPlaceholder instance")


def qubit_index(qubit: QubitDesignator) -> int:
    """
    Get the index of a QubitDesignator.

    :param qubit: the qubit designator.
    :return: An int that is the qubit index.
    """
    if isinstance(qubit, Qubit):
        return qubit.index
    elif isinstance(qubit, int):
        return qubit
    else:
        raise TypeError(f"Cannot unwrap unaddressed QubitPlaceholder: {qubit}")


# Like the Tuple, the List must be length 2, where the first item is a string and the second an
# int. However, specifying Union[str, int] as the generic type argument to List doesn't sufficiently
# constrain the types, and mypy gets confused in unpack_classical_reg, below. Hence, just specify
# List[Any] here.
MemoryReferenceDesignator = Union["MemoryReference", quil_rs.MemoryReference, Tuple[str, int], List[Any], str]


def unpack_classical_reg(c: MemoryReferenceDesignator) -> "MemoryReference":
    """
    Get the address for a classical register.

    :param c: A list of length 2, a pair, a string (to be interpreted as name[0]), or a
        MemoryReference.
    :return: The address as a MemoryReference.
    """
    if isinstance(c, list) or isinstance(c, tuple):
        if len(c) > 2 or len(c) == 0:
            raise ValueError("if c is a list/tuple, it should be of length <= 2")
        if len(c) == 1:
            c = (c[0], 0)
        if not isinstance(c[0], str):
            raise ValueError("if c is a list/tuple, its first member should be a string")
        if not isinstance(c[1], int):
            raise ValueError("if c is a list/tuple, its second member should be an int")
        return MemoryReference(c[0], c[1])
    if isinstance(c, MemoryReference):
        return c
    elif isinstance(c, str):
        return MemoryReference(c, 0)
    else:
        raise TypeError("c should be a list of length 2, a pair, a string, or a MemoryReference")


class Label(QuilAtom):
    """
    Representation of a label.

    :param label_name: The label name.
    """

    def __init__(self, label_name: str):
        self.name = label_name

    def out(self) -> str:
        return "@{name}".format(name=self.name)

    def __str__(self) -> str:
        return "@{name}".format(name=self.name)

    def __repr__(self) -> str:
        return "<Label {0}>".format(repr(self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Label) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class LabelPlaceholder(QuilAtom):
    def __init__(self, prefix: str = "L"):
        self.prefix = prefix

    def out(self) -> str:
        raise RuntimeError("Label has not been assigned a name")

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return "<LabelPlaceholder {} {}>".format(self.prefix, id(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LabelPlaceholder) and id(other) == id(self)

    def __hash__(self) -> int:
        return hash(id(self))


ParameterDesignator = Union["Expression", "MemoryReference", int, float, complex, np.number]


def _convert_to_rs_expression(
    parameter: Union[ParameterDesignator, quil_rs_expr.Expression]
) -> quil_rs_expr.Expression:
    if isinstance(parameter, quil_rs_expr.Expression):
        return parameter
    elif isinstance(parameter, (int, float, complex, np.number)):
        return quil_rs_expr.Expression.from_number(complex(parameter))
    elif isinstance(parameter, (Expression, MemoryReference)):
        return quil_rs_expr.Expression.parse(str(parameter))
    raise ValueError(f"{type(parameter)} is not a valid ParameterDesignator")


def _convert_to_rs_expressions(
    parameters: Sequence[Union[ParameterDesignator, quil_rs_expr.Expression]]
) -> List[quil_rs_expr.Expression]:
    return [_convert_to_rs_expression(parameter) for parameter in parameters]


def _convert_to_py_parameter(
    parameter: Union[ParameterDesignator, quil_rs.MemoryReference, quil_rs_expr.Expression]
) -> ParameterDesignator:
    if isinstance(parameter, quil_rs_expr.Expression):
        if parameter.is_address():
            return MemoryReference._from_rs_memory_reference(parameter.to_address())
        if parameter.is_function_call():
            return Function._from_rs_function_call(parameter.to_function_call())
        if parameter.is_infix():
            return BinaryExp._from_rs_infix_expression(parameter.to_infix())
        if parameter.is_prefix():
            raise NotImplementedError("prefix conversion not implemented")
        if parameter.is_number():
            return parameter.to_number()
        if parameter.is_pi():
            return np.pi
        if parameter.is_variable():
            return Parameter(parameter.to_variable())
    elif isinstance(parameter, (Expression, MemoryReference, Number, complex)):
        return parameter
    raise ValueError(f"{type(parameter)} is not a valid ParameterDesignator")


def _convert_to_py_parameters(
    parameters: Sequence[Union[ParameterDesignator, quil_rs.MemoryReference, quil_rs_expr.Expression]]
) -> List[ParameterDesignator]:
    return [_convert_to_py_parameter(parameter) for parameter in parameters]


@deprecated(
    deprecated_in="4.0",
    removed_in="5.0",
    current_version=pyquil_version,
    details="The format_parameter function will be removed.",
)
def format_parameter(element: ParameterDesignator) -> str:
    """
    Formats a particular parameter. Essentially the same as built-in formatting except using 'i'
    instead of 'j' for the imaginary number.

    :param element: The parameter to format for Quil output.
    """
    if isinstance(element, (int, np.integer)):
        return repr(element)
    elif isinstance(element, float):
        return _check_for_pi(element)
    elif isinstance(element, complex):
        out = ""
        r = element.real
        i = element.imag
        if i == 0:
            return repr(r)

        if r != 0:
            out += repr(r)

        if i == 1:
            assert np.isclose(r, 0, atol=1e-14)
            out = "i"
        elif i == -1:
            assert np.isclose(r, 0, atol=1e-14)
            out = "-i"
        elif i < 0:
            out += repr(i) + "i"
        elif r != 0:
            out += "+" + repr(i) + "i"
        else:
            out += repr(i) + "i"

        return out
    elif isinstance(element, MemoryReference):
        return str(element)
    elif isinstance(element, Expression):
        return _expression_to_string(element)
    raise AssertionError("Invalid parameter: %r" % element)


ExpressionValueDesignator = Union[int, float, complex]
ExpressionDesignator = Union["Expression", ExpressionValueDesignator]


def _convert_to_py_expression(expression: Union[ExpressionDesignator, quil_rs_expr.Expression]) -> ExpressionDesignator:
    if isinstance(expression, (Expression, Number)):
        return expression
    if isinstance(expression, quil_rs_expr.Expression):
        if expression.is_pi():
            return np.pi
        if expression.is_number():
            return expression.to_number()
        if expression.is_variable():
            return Parameter(expression.to_variable())
        if expression.is_infix():
            return BinaryExp._from_rs_infix_expression(expression.to_infix())
        if expression.is_address():
            return MemoryReference._from_rs_memory_reference(expression.to_address())
        if expression.is_prefix():
            prefix = expression.to_prefix()
            py_expression = _convert_to_py_expression(prefix.expression)
            if prefix == quil_rs_expr.PrefixOperator.Plus:
                return py_expression
            elif isinstance(py_expression, (int, float, complex, Expression)):
                return -py_expression
    raise TypeError(f"{type(expression)} is not a valid ExpressionDesignator")


class Expression(object):
    """
    Expression involving some unbound parameters. Parameters in Quil are represented as a label
    like '%x' for the parameter named 'x'. An example expression therefore may be '%x*(%y/4)'.

    Expressions may also have function calls, supported functions in Quil are sin, cos, sqrt, exp,
    and cis.

    This class overrides all the Python operators that are supported by Quil.
    """

    def __str__(self) -> str:
        return _expression_to_string(self)

    def __repr__(self) -> str:
        return str(self.__class__.__name__) + "(" + ",".join(map(repr, self.__dict__.values())) + ")"

    def __add__(self, other: ExpressionDesignator) -> "Add":
        return Add(self, other)

    def __radd__(self, other: ExpressionDesignator) -> "Add":
        return Add(other, self)

    def __sub__(self, other: ExpressionDesignator) -> "Sub":
        return Sub(self, other)

    def __rsub__(self, other: ExpressionDesignator) -> "Sub":
        return Sub(other, self)

    def __mul__(self, other: ExpressionDesignator) -> "Mul":
        return Mul(self, other)

    def __rmul__(self, other: ExpressionDesignator) -> "Mul":
        return Mul(other, self)

    def __div__(self, other: ExpressionDesignator) -> "Div":
        return Div(self, other)

    __truediv__ = __div__

    def __rdiv__(self, other: ExpressionDesignator) -> "Div":
        return Div(other, self)

    __rtruediv__ = __rdiv__

    def __pow__(self, other: ExpressionDesignator) -> "Pow":
        return Pow(self, other)

    def __rpow__(self, other: ExpressionDesignator) -> "Pow":
        return Pow(other, self)

    def __neg__(self) -> "Mul":
        return Mul(-1, self)

    def _substitute(self, d: Any) -> ExpressionDesignator:
        return self


ParameterSubstitutionsMapDesignator = Mapping[Union["Parameter", "MemoryReference"], ExpressionValueDesignator]


def substitute(expr: ExpressionDesignator, d: ParameterSubstitutionsMapDesignator) -> ExpressionDesignator:
    """
    Using a dictionary of substitutions ``d``, try and explicitly evaluate as much of ``expr`` as
    possible. This supports substitution of both parameters and memory references. Each memory
    reference must be individually assigned a value at each memory offset to be substituted.

    :param expr: The expression whose parameters or memory references are to be substituted.
    :param d: Numerical substitutions for parameters or memory references.
    :return: A partially simplified Expression, or a number.
    """
    if isinstance(expr, Expression):
        return expr._substitute(d)
    return expr


def substitute_array(a: Union[Sequence[Expression], np.ndarray], d: ParameterSubstitutionsMapDesignator) -> np.ndarray:
    """
    Apply ``substitute`` to all elements of an array ``a`` and return the resulting array.

    :param a: The array of expressions whose parameters or memory references are to be substituted.
    :param d: Numerical substitutions for parameters or memory references, for all array elements.
    :return: An array of partially substituted Expressions, or numbers.
    """
    a = np.asarray(a, order="C")
    return np.array([substitute(v, d) for v in a.flat]).reshape(a.shape)


class Parameter(QuilAtom, Expression):
    """
    Parameters in Quil are represented as a label like '%x' for the parameter named 'x'.
    """

    def __init__(self, name: str):
        self.name = name

    def out(self) -> str:
        return "%" + self.name

    def _substitute(self, d: ParameterSubstitutionsMapDesignator) -> Union["Parameter", ExpressionValueDesignator]:
        return d.get(self, self)

    def __str__(self) -> str:
        return "%" + self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Parameter) and other.name == self.name


class Function(Expression):
    """
    Supported functions in Quil are sin, cos, sqrt, exp, and cis
    """

    def __init__(
        self,
        name: str,
        expression: ExpressionDesignator,
        fn: Callable[[ExpressionValueDesignator], ExpressionValueDesignator],
    ):
        self.name = name
        self.expression = expression
        self.fn = fn

    @classmethod
    def _from_rs_function_call(cls, function_call: quil_rs_expr.FunctionCallExpression) -> "Function":
        expression = _convert_to_py_expression(function_call.expression)
        if function_call.function == quil_rs_expr.ExpressionFunction.Cis:
            return quil_cis(expression)
        if function_call.function == quil_rs_expr.ExpressionFunction.Cosine:
            return quil_cos(expression)
        if function_call.function == quil_rs_expr.ExpressionFunction.Exponent:
            return quil_exp(expression)
        if function_call.function == quil_rs_expr.ExpressionFunction.Sine:
            return quil_sin(expression)
        return quil_sqrt(expression)

    def _substitute(self, d: ParameterSubstitutionsMapDesignator) -> Union["Function", ExpressionValueDesignator]:
        sop = substitute(self.expression, d)
        if isinstance(sop, Expression):
            return Function(self.name, sop, self.fn)
        return self.fn(sop)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Function) and self.name == other.name and self.expression == other.expression

    def __neq__(self, other: "Function") -> bool:
        return not self.__eq__(other)


def quil_sin(expression: ExpressionDesignator) -> Function:
    return Function("SIN", expression, np.sin)


def quil_cos(expression: ExpressionDesignator) -> Function:
    return Function("COS", expression, np.cos)


def quil_sqrt(expression: ExpressionDesignator) -> Function:
    return Function("SQRT", expression, np.sqrt)


def quil_exp(expression: ExpressionDesignator) -> Function:
    return Function("EXP", expression, np.exp)


def quil_cis(expression: ExpressionDesignator) -> Function:
    def _cis(x: ExpressionValueDesignator) -> complex:
        # numpy doesn't ship with type stubs, so mypy assumes anything coming from numpy has type
        # Any, hence we need to cast the return type to complex here to satisfy the type checker.
        return cast(complex, np.exp(1j * x))

    return Function("CIS", expression, _cis)


class BinaryExp(Expression):
    operator: ClassVar[str]
    precedence: ClassVar[int]
    associates: ClassVar[str]

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["BinaryExp", ExpressionValueDesignator]:
        raise NotImplementedError

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        self.op1 = op1
        self.op2 = op2

    @classmethod
    def _from_rs_infix_expression(cls, infix_expression: quil_rs_expr.InfixExpression) -> "BinaryExp":
        left = _convert_to_py_expression(infix_expression.left)
        right = _convert_to_py_expression(infix_expression.right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Plus:
            return Add(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Minus:
            return Sub(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Slash:
            return Div(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Star:
            return Mul(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Caret:
            return Pow(left, right)
        raise ValueError(f"{type(infix_expression)} is not a valid InfixExpression")

    def _substitute(self, d: ParameterSubstitutionsMapDesignator) -> Union["BinaryExp", ExpressionValueDesignator]:
        sop1, sop2 = substitute(self.op1, d), substitute(self.op2, d)
        return self.fn(sop1, sop2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.op1 == other.op1 and self.op2 == other.op2

    def __neq__(self, other: "BinaryExp") -> bool:
        return not self.__eq__(other)


class Add(BinaryExp):
    operator = " + "
    precedence = 1
    associates = "both"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Add", ExpressionValueDesignator]:
        return a + b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        super(Add, self).__init__(op1, op2)


class Sub(BinaryExp):
    operator = " - "
    precedence = 1
    associates = "left"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Sub", ExpressionValueDesignator]:
        return a - b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        super(Sub, self).__init__(op1, op2)


class Mul(BinaryExp):
    operator = "*"
    precedence = 2
    associates = "both"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Mul", ExpressionValueDesignator]:
        return a * b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        super(Mul, self).__init__(op1, op2)


class Div(BinaryExp):
    operator = "/"
    precedence = 2
    associates = "left"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Div", ExpressionValueDesignator]:
        return a / b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        super(Div, self).__init__(op1, op2)


class Pow(BinaryExp):
    operator = "^"
    precedence = 3
    associates = "right"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Pow", ExpressionValueDesignator]:
        return a**b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        super(Pow, self).__init__(op1, op2)


def _expression_to_string(expression: ExpressionDesignator) -> str:
    """
    Recursively converts an expression to a string taking into account precedence and associativity
    for placing parenthesis.

    :param expression: expression involving parameters
    :return: string such as '%x*(%y-4)'
    """
    if isinstance(expression, BinaryExp):
        left = _expression_to_string(expression.op1)
        if isinstance(expression.op1, BinaryExp) and not (
            expression.op1.precedence > expression.precedence
            or expression.op1.precedence == expression.precedence
            and expression.associates in ("left", "both")
        ):
            left = "(" + left + ")"

        right = _expression_to_string(expression.op2)
        if isinstance(expression.op2, BinaryExp) and not (
            expression.precedence < expression.op2.precedence
            or expression.precedence == expression.op2.precedence
            and expression.associates in ("right", "both")
        ):
            right = "(" + right + ")"
        # If op2 is a float, it will maybe represented as a multiple
        # of pi in right. If that is the case, then we need to take
        # extra care to insert parens. See gh-943.
        elif isinstance(expression.op2, float) and (("pi" in right and right != "pi")):
            right = "(" + right + ")"

        return left + expression.operator + right
    elif isinstance(expression, Function):
        return expression.name + "(" + _expression_to_string(expression.expression) + ")"
    elif isinstance(expression, Parameter):
        return str(expression)
    else:
        return format_parameter(expression)


def _contained_parameters(expression: ExpressionDesignator) -> Set[Parameter]:
    """
    Determine which parameters are contained in this expression.

    :param expression: expression involving parameters
    :return: set of parameters contained in this expression
    """
    if isinstance(expression, BinaryExp):
        return _contained_parameters(expression.op1) | _contained_parameters(expression.op2)
    elif isinstance(expression, Function):
        return _contained_parameters(expression.expression)
    elif isinstance(expression, Parameter):
        return {expression}
    else:
        return set()


def _check_for_pi(element: float) -> str:
    """
    Check to see if there exists a rational number r = p/q
    in reduced form for which the difference between element/np.pi
    and r is small and q <= 8.

    :param element: the number to check
    :return element: pretty print string if true, else standard representation.
    """
    frac = Fraction(element / np.pi).limit_denominator(8)
    num, den = frac.numerator, frac.denominator
    sign = "-" if num < 0 else ""
    if num / float(den) == element / np.pi:
        if num == 0:
            return "0"
        elif abs(num) == 1 and den == 1:
            return sign + "pi"
        elif abs(num) == 1:
            return sign + "pi/" + repr(den)
        elif den == 1:
            return repr(num) + "*pi"
        else:
            return repr(num) + "*pi/" + repr(den)
    else:
        return repr(element)


class MemoryReference(QuilAtom, Expression):
    """
    Representation of a reference to a classical memory address.

    :param name: The name of the variable
    :param offset: Everything in Quil is a C-style array, so every memory reference has an offset.
    :param declared_size: The optional size of the named declaration. This can be used for bounds
        checking, but isn't. It is used for pretty-printing to quil by deciding whether to output
        memory references with offset 0 as either e.g. ``ro[0]`` or ``beta`` depending on whether
        the declared variable is of length >1 or 1, resp.
    """

    def __init__(self, name: str, offset: int = 0, declared_size: Optional[int] = None):
        if not isinstance(offset, int) or offset < 0:
            raise TypeError("MemoryReference offset must be a non-negative int")
        self.name = name
        self.offset = offset
        self.declared_size = declared_size

    @classmethod
    def _from_rs_memory_reference(cls, memory_reference: quil_rs.MemoryReference) -> "MemoryReference":
        return cls(memory_reference.name, memory_reference.index)

    @classmethod
    def _from_parameter_str(cls, memory_reference_str: str) -> "MemoryReference":
        expression = quil_rs_expr.Expression.parse(memory_reference_str)
        if expression.is_address():
            return cls._from_rs_memory_reference(expression.to_address())
        raise ValueError(f"{memory_reference_str} is not a memory reference")

    def _to_rs_memory_reference(self) -> quil_rs.MemoryReference:
        return quil_rs.MemoryReference(self.name, self.offset)

    def out(self) -> str:
        if self.declared_size is not None and self.declared_size == 1 and self.offset == 0:
            return "{}".format(self.name)
        else:
            return "{}[{}]".format(self.name, self.offset)

    def __str__(self) -> str:
        if self.declared_size is not None and self.declared_size == 1 and self.offset == 0:
            return "{}".format(self.name)
        else:
            return "{}[{}]".format(self.name, self.offset)

    def __repr__(self) -> str:
        return "<MRef {}[{}]>".format(self.name, self.offset)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryReference) and other.name == self.name and other.offset == self.offset

    def __hash__(self) -> int:
        return hash((self.name, self.offset))

    def __getitem__(self, offset: int) -> "MemoryReference":
        if self.offset != 0:
            raise ValueError("Please only index off of the base MemoryReference (offset = 0)")

        # NOTE If a MemoryReference is the result of parsing (not
        # manually instantiated), it will likely be instantiated
        # without a declared_size, and so bounds checking will be
        # impossible.
        if self.declared_size and offset >= self.declared_size:
            raise IndexError("MemoryReference index out of range")

        return MemoryReference(name=self.name, offset=offset)

    def _substitute(
        self, d: ParameterSubstitutionsMapDesignator
    ) -> Union["MemoryReference", ExpressionValueDesignator]:
        if self in d:
            return d[self]

        return self


def _contained_mrefs(expression: ExpressionDesignator) -> Set[MemoryReference]:
    """
    Determine which memory references are contained in this expression.

    :param expression: expression involving parameters
    :return: set of parameters contained in this expression
    """
    if isinstance(expression, BinaryExp):
        return _contained_mrefs(expression.op1) | _contained_mrefs(expression.op2)
    elif isinstance(expression, Function):
        return _contained_mrefs(expression.expression)
    elif isinstance(expression, MemoryReference):
        return {expression}
    else:
        return set()


class Frame(quil_rs.FrameIdentifier):
    """
    Representation of a frame descriptor.
    """

    def __new__(cls, qubits: Sequence[QubitDesignator], name: str) -> Self:
        return super().__new__(cls, name, _convert_to_rs_qubits(qubits))

    @classmethod
    def _from_rs_frame_identifier(cls, frame: quil_rs.FrameIdentifier) -> "Frame":
        return super().__new__(cls, frame.name, frame.qubits)

    @property  # type: ignore[override]
    def qubits(self) -> Tuple[QubitDesignator, ...]:
        return tuple(_convert_to_py_qubits(super().qubits))

    @qubits.setter
    def qubits(self, qubits: Tuple[Qubit, FormalArgument]) -> None:
        quil_rs.FrameIdentifier.qubits.__set__(self, _convert_to_rs_qubits(qubits))  # type: ignore[attr-defined]

    def out(self) -> str:
        return str(self)


class WaveformInvocation(quil_rs.WaveformInvocation, QuilAtom):
    def __new__(cls, name: str, parameters: Optional[Dict[str, ParameterDesignator]] = None) -> Self:
        if parameters is None:
            parameters = {}
        rs_parameters = {key: _convert_to_rs_expression(value) for key, value in parameters.items()}
        return super().__new__(cls, name, rs_parameters)

    @property  # type: ignore[override]
    def parameters(self) -> Dict[str, ParameterDesignator]:
        return {key: _convert_to_py_parameter(value) for key, value in super().parameters.items()}

    @parameters.setter
    def parameters(self, parameters: Dict[str, ParameterDesignator]) -> None:
        rs_parameters = {key: _convert_to_rs_expression(value) for key, value in parameters.items()}
        quil_rs.WaveformInvocation.parameters.__set__(self, rs_parameters)  # type: ignore[attr-defined]

    def out(self) -> str:
        return str(self)


@deprecated(
    deprecated_in="4.0",
    removed_in="5.0",
    current_version=pyquil_version,
    details="The WaveformReference class will be removed, consider using WaveformInvocation instead.",
)
class WaveformReference(WaveformInvocation):
    """
    Representation of a Waveform reference.
    """

    def __new__(cls, name: str) -> Self:
        return super().__new__(cls, name, {})


def _template_waveform_property(name: str, doc: Optional[str] = None) -> property:
    """
    Helper method for initializing getters, setters, and deleters for
    parameters on a ``TemplateWaveform``. Should only be used inside of
    ``TemplateWaveform`` or one its base classes.
    """

    def fget(self: "TemplateWaveform") -> Optional[ParameterDesignator]:
        return self.get_parameter(name)

    def fset(self: "TemplateWaveform", value: ParameterDesignator) -> None:
        self.set_parameter(name, value)

    def fdel(self: "TemplateWaveform") -> None:
        self.set_parameter(name, None)

    return property(fget, fset, fdel, doc)


class TemplateWaveform(quil_rs.WaveformInvocation, QuilAtom):
    @deprecated(
        deprecated_in="4.0",
        removed_in="5.0",
        current_version=pyquil_version,
        details="The TemplateWaveform class will be removed, consider using WaveformInvocation instead.",
    )
    def __new__(cls, name: str, *, duration: float, **kwargs: Union[ParameterDesignator, ExpressionDesignator]) -> Self:
        rs_parameters = {key: _convert_to_rs_expression(value) for key, value in kwargs.items() if value is not None}
        rs_parameters["duration"] = _convert_to_rs_expression(duration)
        return super().__new__(cls, name, rs_parameters)

    def out(self) -> str:
        return str(self)

    def get_parameter(self, name: str) -> Optional[ParameterDesignator]:
        parameter = super().parameters.get(name, None)
        if parameter is None:
            return None
        return _convert_to_py_parameter(parameter)

    def set_parameter(self, name: str, value: Optional[ParameterDesignator]) -> None:
        parameters = super().parameters
        if value is None:
            parameters.pop(name, None)
        else:
            parameters[name] = _convert_to_rs_expression(value)
        quil_rs.WaveformInvocation.parameters.__set__(self, parameters)  # type: ignore[attr-defined]

    duration = _template_waveform_property("duration")

    def num_samples(self, rate: float) -> int:
        """The number of samples in the reference implementation of the waveform.

        Note: this does not include any hardware-enforced alignment (cf.
        documentation for `samples`).

        :param rate: The sample rate, in Hz.
        :return: The number of samples.

        """
        return int(np.ceil(self.duration * rate))

    def samples(self, rate: float) -> np.ndarray:
        """A reference implementation of waveform sample generation.

        Note: this is close but not always exactly equivalent to the actual IQ
        values produced by the waveform generators on Rigetti hardware. The
        actual ADC process imposes some alignment constraints on the waveform
        duration (in particular, it must be compatible with the clock rate).

        :param rate: The sample rate, in Hz.
        :returns: An array of complex samples.

        """
        raise NotImplementedError()

    @classmethod
    def _from_rs_waveform_invocation(cls, waveform: quil_rs.WaveformInvocation) -> "TemplateWaveform":
        return super().__new__(cls, waveform.name, waveform.parameters)


def _update_envelope(
    iqs: np.ndarray,
    rate: float,
    scale: Optional[float],
    phase: Optional[float],
    detuning: Optional[float],
) -> np.ndarray:
    """Update a pulse envelope by optional shape parameters.

    The optional parameters are: 'scale', 'phase', 'detuning'.

    :param iqs: The basic pulse envelope.
    :param rate: The sample rate (in Hz).
    :return: The updated pulse envelope.
    """

    def default(obj: Optional[float], val: float) -> float:
        return obj if obj is not None else val

    scale = default(scale, 1.0)
    phase = default(phase, 0.0)
    detuning = default(detuning, 0.0)

    iqs *= scale * np.exp(1j * phase) * np.exp(1j * 2 * np.pi * detuning * np.arange(len(iqs)) / rate)

    return iqs


Waveform = Union[WaveformReference, TemplateWaveform]


def _convert_to_py_waveform(waveform: quil_rs.WaveformInvocation) -> Waveform:
    if not isinstance(waveform, quil_rs.WaveformInvocation):
        raise TypeError(f"{type(waveform)} is not a WaveformInvocation")
    if len(waveform.parameters) == 0:
        return WaveformReference(waveform.name)

    return TemplateWaveform._from_rs_waveform_invocation(waveform)
