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
"""Classes that represent the atomic building blocks of Quil expressions."""

import inspect
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from numbers import Number
from typing import (
    Any,
    Callable,
    ClassVar,
    NoReturn,
    Optional,
    Union,
    cast,
)

import numpy as np
import quil.expression as quil_rs_expr
import quil.instructions as quil_rs
from deprecated.sphinx import deprecated
from typing_extensions import Self


class QuilAtom:
    """Abstract class for atomic elements of Quil."""

    def out(self) -> str:
        """Return the element as a valid Quil string."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Get a string representation of the element, possibly not valid Quil."""
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        """Return True if the other object is equal to this one."""
        raise NotImplementedError()

    def __hash__(self) -> int:
        """Return a hash of the object."""
        raise NotImplementedError()


class Qubit(QuilAtom):
    """Representation of a qubit.

    :param index: Index of the qubit.
    """

    def __init__(self, index: int):
        """Initialize a qubit."""
        if not (isinstance(index, int) and index >= 0):
            raise TypeError("Addr index must be a non-negative int")
        self.index = index

    def out(self) -> str:
        """Return the element as a valid Quil string."""
        return str(self.index)

    def __str__(self) -> str:
        return str(self.index)

    def __repr__(self) -> str:
        return f"<Qubit {self.index}>"

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Qubit) and other.index == self.index


class FormalArgument(QuilAtom):
    """Representation of a formal argument associated with a DEFCIRCUIT or DEFGATE ... AS PAULI-SUM or DEFCAL form."""

    def __init__(self, name: str):
        """Initialize a formal argument."""
        if not isinstance(name, str):
            raise TypeError("Formal arguments must be named by a string.")
        self.name = name

    def out(self) -> str:
        """Return the element as a valid Quil string."""
        return str(self)

    @property
    @deprecated(
        "Getting the index of a FormalArgument is invalid. This method will be removed in a future release.",
        version="4.0",
    )
    def index(self) -> NoReturn:
        """Formal arguments do not have an index. Using this property raises a RuntimeError."""
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
    """A placeholder for a qubit.

    This is useful for constructing circuits without assigning qubits to specific indices.
    Qubit placeholders must be resolved to actual qubits before they can be used in a program.
    """

    def __init__(self, placeholder: Optional[quil_rs.QubitPlaceholder] = None):
        """Initialize a qubit placeholder, or get a new handle for an existing placeholder."""
        if placeholder is not None:
            self._placeholder = placeholder
        else:
            self._placeholder = quil_rs.QubitPlaceholder()

    @staticmethod
    def register(n: int) -> list["QubitPlaceholder"]:
        r"""Return a 'register' of ``n`` QubitPlaceholders.

        >>> from pyquil import Program
        >>> from pyquil.gates import H
        >>> from pyquil.quil import address_qubits
        >>> from pyquil.quilatom import QubitPlaceholder
        >>> qs = QubitPlaceholder.register(8)  # a qubyte
        >>> prog = Program(H(q) for q in qs)
        >>> address_qubits(prog).out()
        'H 0\nH 1\nH 2\nH 3\nH 4\nH 5\nH 6\nH 7\n'
        >>>

        The returned register is a Python list of QubitPlaceholder objects, so all
        normal list semantics apply.

        :param n: The number of qubits in the register
        """
        return [QubitPlaceholder() for _ in range(n)]

    def out(self) -> str:
        """Raise a RuntimeError, as Qubit placeholders are not valid Quil."""
        raise RuntimeError(f"Qubit {self} has not been assigned an index")

    @property
    def index(self) -> NoReturn:
        """Raise a RuntimeError, as Qubit placeholders do not have an index."""
        raise RuntimeError(f"Qubit {self} has not been assigned an index")

    def __str__(self) -> str:
        return f"q{id(self)}"

    def __repr__(self) -> str:
        return f"q{id(self)}"

    def __hash__(self) -> int:
        return hash(self._placeholder)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, quil_rs.QubitPlaceholder):
            return self._placeholder == other
        if isinstance(other, QubitPlaceholder):
            return self._placeholder == other._placeholder
        return False

    def __lt__(self, other: object) -> bool:
        if isinstance(other, quil_rs.QubitPlaceholder):
            return self._placeholder < other
        if isinstance(other, QubitPlaceholder):
            return self._placeholder < other._placeholder
        raise TypeError(f"Comparison between LabelPlaceholder and {type(other)} is not supported.")


QubitDesignator = Union[Qubit, QubitPlaceholder, FormalArgument, int]


def _convert_to_rs_qubit(qubit: Union[QubitDesignator, quil_rs.Qubit, QubitPlaceholder]) -> quil_rs.Qubit:
    if isinstance(qubit, quil_rs.Qubit):
        return qubit
    if isinstance(qubit, Qubit):
        return quil_rs.Qubit.from_fixed(qubit.index)
    if isinstance(qubit, QubitPlaceholder):
        return quil_rs.Qubit.from_placeholder(qubit._placeholder)
    if isinstance(qubit, FormalArgument):
        return quil_rs.Qubit.from_variable(qubit.name)
    if isinstance(qubit, int):
        return quil_rs.Qubit.from_fixed(qubit)
    raise ValueError(f"{type(qubit)} is not a valid QubitDesignator")


def _convert_to_rs_qubits(qubits: Iterable[QubitDesignator]) -> list[quil_rs.Qubit]:
    return [_convert_to_rs_qubit(qubit) for qubit in qubits]


def _convert_to_py_qubit(qubit: Union[QubitDesignator, quil_rs.Qubit, quil_rs.QubitPlaceholder]) -> QubitDesignator:
    if isinstance(qubit, quil_rs.Qubit):
        if qubit.is_fixed():
            return Qubit(qubit.to_fixed())
        if qubit.is_variable():
            return FormalArgument(qubit.to_variable())
        if qubit.is_placeholder():
            return QubitPlaceholder(placeholder=qubit.to_placeholder())
    if isinstance(qubit, (Qubit, QubitPlaceholder, FormalArgument, Parameter, int)):
        return qubit
    raise ValueError(f"{type(qubit)} is not a valid QubitDesignator")


def _convert_to_py_qubits(qubits: Iterable[Union[QubitDesignator, quil_rs.Qubit]]) -> list[QubitDesignator]:
    return [_convert_to_py_qubit(qubit) for qubit in qubits]


def unpack_qubit(qubit: Union[QubitDesignator, FormalArgument]) -> Union[Qubit, QubitPlaceholder, FormalArgument]:
    """Get a qubit from an object.

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
    """Get the index of a QubitDesignator.

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
# list[Any] here.
MemoryReferenceDesignator = Union["MemoryReference", quil_rs.MemoryReference, tuple[str, int], list[Any], str]


def unpack_classical_reg(c: MemoryReferenceDesignator) -> "MemoryReference":
    """Get the address for a classical register.

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
    """Representation of a label.

    :param label_name: The label name.
    """

    def __init__(self, label_name: str):
        """Initialize a new label."""
        self.target = quil_rs.Target.from_fixed(label_name)

    @staticmethod
    def _from_rs_target(target: quil_rs.Target) -> "Label":
        return Label(target.to_fixed())

    def out(self) -> str:
        """Return the label as a valid Quil string."""
        return self.target.to_quil()

    @property
    def name(self) -> str:
        """Return the label name."""
        return self.target.to_fixed()

    @name.setter
    def name(self, label_name: str) -> None:
        self.target = quil_rs.Target.from_fixed(label_name)

    def __str__(self) -> str:
        return self.target.to_quil_or_debug()

    def __repr__(self) -> str:
        return repr(self.target)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Label):
            return self.target == other.target
        return False

    def __hash__(self) -> int:
        return hash(self.target)


class LabelPlaceholder(QuilAtom):
    """A placeholder for a Quil label.

    This is useful for constructing circuits without needing to name them.
    All placeholders must be resolved to actual labels before they can be used in a program.
    """

    def __init__(self, prefix: str = "L", *, placeholder: Optional[quil_rs.TargetPlaceholder] = None):
        """Initialize a new label placeholder."""
        if placeholder:
            self.target = quil_rs.Target.from_placeholder(placeholder)
        else:
            self.target = quil_rs.Target.from_placeholder(quil_rs.TargetPlaceholder(prefix))

    @staticmethod
    def _from_rs_target(target: quil_rs.Target) -> "LabelPlaceholder":
        return LabelPlaceholder(placeholder=target.to_placeholder())

    @property
    def prefix(self) -> str:
        """Get the prefix of the label placeholder."""
        return self.target.to_placeholder().base_label

    def out(self) -> str:
        """Raise a RuntimeError, as label placeholders are not valid Quil."""
        raise RuntimeError("Label has not been assigned a name")

    def __str__(self) -> str:
        return self.target.to_quil_or_debug()

    def __repr__(self) -> str:
        return repr(self.target)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LabelPlaceholder):
            return self.target == other.target
        return False

    def __hash__(self) -> int:
        return hash(self.target)


ParameterDesignator = Union["Expression", "MemoryReference", int, float, complex, np.number]


def _convert_to_rs_expression(
    parameter: Union[ParameterDesignator, quil_rs_expr.Expression],
) -> quil_rs_expr.Expression:
    if isinstance(parameter, quil_rs_expr.Expression):
        return parameter
    elif isinstance(parameter, (int, float, complex, np.number)):
        return quil_rs_expr.Expression.from_number(complex(parameter))
    elif isinstance(parameter, (Expression, MemoryReference)):
        return quil_rs_expr.Expression.parse(str(parameter))
    raise ValueError(f"{type(parameter)} is not a valid ParameterDesignator")


def _convert_to_rs_expressions(
    parameters: Sequence[Union[ParameterDesignator, quil_rs_expr.Expression]],
) -> list[quil_rs_expr.Expression]:
    return [_convert_to_rs_expression(parameter) for parameter in parameters]


@deprecated(version="4.0", reason="This function has been superseded by the `quil` package and will be removed soon.")
def format_parameter(element: ParameterDesignator) -> str:
    """Format a particular parameter.

    Essentially the same as built-in formatting except using 'i' instead of 'j' for the imaginary number.

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
            if not np.isclose(r, 0, atol=1e-14):
                raise ValueError(f"Invalid complex number: {element}")
            out = "i"
        elif i == -1:
            if not np.isclose(r, 0, atol=1e-14):
                raise ValueError(f"Invalid complex number: {element}")
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
    raise AssertionError(f"Invalid parameter: {element}")


ExpressionValueDesignator = Union[int, float, complex]
ExpressionDesignator = Union["Expression", ExpressionValueDesignator]


def _convert_to_py_expression(
    expression: Union[
        ParameterDesignator,
        ExpressionDesignator,
        ExpressionValueDesignator,
        quil_rs_expr.Expression,
        quil_rs.MemoryReference,
    ],
) -> ExpressionDesignator:
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
        if expression.is_function_call():
            fc = expression.to_function_call()
            parameter = _convert_to_py_expression(fc.expression)
            if fc.function == quil_rs_expr.ExpressionFunction.Cis:
                return quil_cis(parameter)
            if fc.function == quil_rs_expr.ExpressionFunction.Cosine:
                return quil_cos(parameter)
            if fc.function == quil_rs_expr.ExpressionFunction.Exponent:
                return quil_exp(parameter)
            if fc.function == quil_rs_expr.ExpressionFunction.Sine:
                return quil_sin(parameter)
            if fc.function == quil_rs_expr.ExpressionFunction.SquareRoot:
                return quil_sqrt(parameter)
        if expression.is_prefix():
            prefix = expression.to_prefix()
            py_expression = _convert_to_py_expression(prefix.expression)
            if prefix == quil_rs_expr.PrefixOperator.Plus:
                return py_expression
            elif isinstance(py_expression, (int, float, complex, Expression)):
                return -py_expression
    raise TypeError(f"{type(expression)} is not a valid ExpressionDesignator")


def _convert_to_py_expressions(
    expressions: Sequence[
        Union[ParameterDesignator, ExpressionDesignator, quil_rs_expr.Expression, quil_rs.MemoryReference]
    ],
) -> Sequence[ExpressionDesignator]:
    return [_convert_to_py_expression(expression) for expression in expressions]


class Expression:
    """Expression involving some unbound parameters.

    Parameters in Quil are represented as a label like '%x' for the parameter named 'x'. An example expression therefore
    may be '%x*(%y/4)'.

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

    def _evaluate(self) -> np.complex128:
        """Attempt to evaluate the expression to by simplifying it to a complex number.

        Expression simplification can be slow, especially for large recursive expressions.
        This method will raise a ValueError if the expression cannot be simplified to a complex
        number.
        """
        expr = quil_rs_expr.Expression.parse(str(self))
        expr.simplify()  # type: ignore[no-untyped-call]
        if not expr.is_number():
            raise ValueError(f"Cannot evaluate expression {self} to a number. Got {expr}.")
        return np.complex128(expr.to_number())

    def __float__(self) -> float:
        """Return a copy of the expression as a float by attempting to simplify the expression.

        Expression simplification can be slow, especially for large recursive expressions.
        This cast will raise a ValueError if simplification doesn't result in a real number.
        """
        value = self._evaluate()
        if value.imag != 0:
            raise ValueError(f"Cannot convert complex value with non-zero imaginary value to float: {value}")
        return float(value.real)

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Implement the numpy array protocol for this expression.

        If the dtype is not object, then there will be an attempt to simplify the expression to a
        complex number. If the expression cannot be simplified to one, then fallback to the
        object representation of the expression.

        Note that expression simplification can be slow for large recursive expressions.
        """
        try:
            if dtype != object:
                return np.asarray(self._evaluate(), dtype=dtype)
            raise ValueError
        except ValueError:
            # np.asarray(self, ...) would cause an infinite recursion error, so we build the array with a
            # placeholder value, then replace it with self after.
            array = np.asarray(None, dtype=object)
            array.flat[0] = self
            return array


ParameterSubstitutionsMapDesignator = Mapping[Union["Parameter", "MemoryReference"], ExpressionValueDesignator]


def substitute(expr: ExpressionDesignator, d: ParameterSubstitutionsMapDesignator) -> ExpressionDesignator:
    r"""Explicitly evaluate as much of ``expr`` as possible, using substitutions from `d`.

    This supports substitution of both parameters and memory references. Each memory reference must be individually
    assigned a value at each memory offset to be substituted.

    :param expr: The expression whose parameters or memory references are to be substituted.
    :param d: Numerical substitutions for parameters or memory references.
    :return: A partially simplified Expression, or a number.
    """
    if isinstance(expr, Expression):
        return expr._substitute(d)
    return expr


def substitute_array(a: Union[Sequence[Expression], np.ndarray], d: ParameterSubstitutionsMapDesignator) -> np.ndarray:
    """Apply ``substitute`` to all elements of an array ``a`` and return the resulting array.

    :param a: The array of expressions whose parameters or memory references are to be substituted.
    :param d: Numerical substitutions for parameters or memory references, for all array elements.
    :return: An array of partially substituted Expressions, or numbers.
    """
    a = np.asarray(a, order="C")
    return np.array([substitute(v, d) for v in a.flat]).reshape(a.shape)


class Parameter(QuilAtom, Expression):
    """Parameters in Quil are represented as a label like '%x' for the parameter named 'x'."""

    def __init__(self, name: str):
        """Initialize a new parameter."""
        self.name = name

    def out(self) -> str:
        """Return the parameter as a valid Quil string."""
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
    """Base class for standard Quil functions."""

    def __init__(
        self,
        name: str,
        expression: ExpressionDesignator,
        fn: Callable[[ExpressionValueDesignator], ExpressionValueDesignator],
    ):
        """Initialize a new function."""
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
    """Quil COS() function."""
    return Function("SIN", expression, np.sin)


def quil_cos(expression: ExpressionDesignator) -> Function:
    """Quil SIN() function."""
    return Function("COS", expression, np.cos)


def quil_sqrt(expression: ExpressionDesignator) -> Function:
    """Quil SQRT() function."""
    return Function("SQRT", expression, np.sqrt)


def quil_exp(expression: ExpressionDesignator) -> Function:
    """Quil EXP() function."""
    return Function("EXP", expression, np.exp)


def quil_cis(expression: ExpressionDesignator) -> Function:
    """Quil CIS() function."""

    def _cis(x: ExpressionValueDesignator) -> complex:
        # numpy doesn't ship with type stubs, so mypy assumes anything coming from numpy has type
        # Any, hence we need to cast the return type to complex here to satisfy the type checker.
        return cast(complex, np.exp(1j * x))

    return Function("CIS", expression, _cis)


class BinaryExp(Expression):
    """A Quil binary expression."""

    operator: ClassVar[str]
    precedence: ClassVar[int]
    associates: ClassVar[str]

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["BinaryExp", ExpressionValueDesignator]:
        """Perform the operation on the two expressions."""
        raise NotImplementedError

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        """Initialize a new binary expression."""
        self.op1 = op1
        self.op2 = op2

    @classmethod
    def _from_rs_infix_expression(
        cls, infix_expression: quil_rs_expr.InfixExpression
    ) -> Union["BinaryExp", ExpressionValueDesignator]:
        left = _convert_to_py_expression(infix_expression.left)
        right = _convert_to_py_expression(infix_expression.right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Plus:
            return Add.fn(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Minus:
            return Sub.fn(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Slash:
            return Div.fn(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Star:
            return Mul.fn(left, right)
        if infix_expression.operator == quil_rs_expr.InfixOperator.Caret:
            return Pow.fn(left, right)
        raise ValueError(f"{type(infix_expression)} is not a valid InfixExpression")

    def _substitute(self, d: ParameterSubstitutionsMapDesignator) -> Union["BinaryExp", ExpressionValueDesignator]:
        sop1, sop2 = substitute(self.op1, d), substitute(self.op2, d)
        return self.fn(sop1, sop2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.op1 == other.op1 and self.op2 == other.op2

    def __neq__(self, other: "BinaryExp") -> bool:
        return not self.__eq__(other)


class Add(BinaryExp):
    """The addition operation."""

    operator = " + "
    precedence = 1
    associates = "both"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Add", ExpressionValueDesignator]:
        """Perform the addition operation."""
        return a + b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        """Initialize a new addition operation between two expressions."""
        super().__init__(op1, op2)


class Sub(BinaryExp):
    """The subtraction operation."""

    operator = " - "
    precedence = 1
    associates = "left"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Sub", ExpressionValueDesignator]:
        """Perform the subtraction operation."""
        return a - b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        """Initialize a new addition operation between two expressions."""
        super().__init__(op1, op2)


class Mul(BinaryExp):
    """The multiplication operation."""

    operator = "*"
    precedence = 2
    associates = "both"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Mul", ExpressionValueDesignator]:
        """Perform the multiplication operation."""
        return a * b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        """Initialize a new multiplication operation between two expressions."""
        super().__init__(op1, op2)


class Div(BinaryExp):
    """The division operation."""

    operator = "/"
    precedence = 2
    associates = "left"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Div", ExpressionValueDesignator]:
        """Perform the division operation."""
        return a / b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        """Initialize a new division operation between two expressions."""
        super().__init__(op1, op2)


class Pow(BinaryExp):
    """The exponentiation operation."""

    operator = "^"
    precedence = 3
    associates = "right"

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union["Pow", ExpressionValueDesignator]:
        """Perform the exponentiation operation."""
        return a**b

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        """Initialize a new exponentiation operation between two expressions."""
        super().__init__(op1, op2)


def _expression_to_string(expression: ExpressionDesignator) -> str:
    """Convert an expression to a string.

    This operation is recursive, and takes into account precedence and associativity when placing parenthesis.

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
        # extra care to insert parens. Similarly, complex numbers need
        # to be wrapped in parens so the imaginary part is captured.
        # See gh-943,1734.
        elif (isinstance(expression.op2, float) and ("pi" in right and right != "pi")) or isinstance(
            expression.op2, complex
        ):
            right = "(" + right + ")"

        return left + expression.operator + right
    elif isinstance(expression, Function):
        return expression.name + "(" + _expression_to_string(expression.expression) + ")"
    elif isinstance(expression, Parameter):
        return str(expression)
    else:
        return format_parameter(expression)


def _contained_parameters(expression: ExpressionDesignator) -> set[Parameter]:
    """Determine which parameters are contained in this expression.

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
    """Return the float as a string, expressing the float as a multiple of pi if possible.

    More specifically, check to see if there exists a rational number r = p/q in reduced form for which the difference
    between element/np.pi and r is small and q <= 8.

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
    """Representation of a reference to a classical memory address.

    :param name: The name of the variable
    :param offset: Everything in Quil is a C-style array, so every memory reference has an offset.
    :param declared_size: The optional size of the named declaration. This can be used for bounds
        checking, but isn't. It is used for pretty-printing to quil by deciding whether to output
        memory references with offset 0 as either e.g. ``ro[0]`` or ``beta`` depending on whether
        the declared variable is of length >1 or 1, resp.
    """

    def __init__(self, name: str, offset: int = 0, declared_size: Optional[int] = None):
        """Initialize a new memory reference."""
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
        """Return the memory reference as a valid Quil string."""
        if self.declared_size is not None and self.declared_size == 1 and self.offset == 0:
            return f"{self.name}"
        else:
            return f"{self.name}[{self.offset}]"

    def __str__(self) -> str:
        if self.declared_size is not None and self.declared_size == 1 and self.offset == 0:
            return f"{self.name}"
        else:
            return f"{self.name}[{self.offset}]"

    def __repr__(self) -> str:
        return f"<MRef {self.name}[{self.offset}]>"

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


def _contained_mrefs(expression: ExpressionDesignator) -> set[MemoryReference]:
    """Determine which memory references are contained in this expression.

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
    """Representation of a frame descriptor."""

    def __new__(cls, qubits: Sequence[QubitDesignator], name: str) -> Self:
        """Initialize a new Frame."""
        return super().__new__(cls, name, _convert_to_rs_qubits(qubits))

    @classmethod
    def _from_rs_frame_identifier(cls, frame: quil_rs.FrameIdentifier) -> "Frame":
        return super().__new__(cls, frame.name, frame.qubits)

    @property  # type: ignore[override]
    def qubits(self) -> tuple[QubitDesignator, ...]:
        """Get the qubits in the frame."""
        return tuple(_convert_to_py_qubits(super().qubits))

    @qubits.setter
    def qubits(self, qubits: tuple[Qubit, FormalArgument]) -> None:
        quil_rs.FrameIdentifier.qubits.__set__(self, _convert_to_rs_qubits(qubits))  # type: ignore[attr-defined]

    def out(self) -> str:
        """Return the frame as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        """Return the frame as a string."""
        return super().to_quil_or_debug()


class WaveformInvocation(quil_rs.WaveformInvocation, QuilAtom):
    """A waveform invocation."""

    def __new__(cls, name: str, parameters: Optional[dict[str, ParameterDesignator]] = None) -> Self:
        """Initialize a new waveform invocation."""
        if parameters is None:
            parameters = {}
        rs_parameters = {key: _convert_to_rs_expression(value) for key, value in parameters.items()}
        return super().__new__(cls, name, rs_parameters)

    @property  # type: ignore[override]
    def parameters(self) -> dict[str, ParameterDesignator]:
        """The parameters in the waveform invocation."""
        return {key: _convert_to_py_expression(value) for key, value in super().parameters.items()}

    @parameters.setter
    def parameters(self, parameters: dict[str, ParameterDesignator]) -> None:
        rs_parameters = {key: _convert_to_rs_expression(value) for key, value in parameters.items()}
        quil_rs.WaveformInvocation.parameters.__set__(self, rs_parameters)  # type: ignore[attr-defined]

    def out(self) -> str:
        """Return the waveform invocation as a valid Quil string."""
        return self.to_quil()

    def __str__(self) -> str:
        """Return the frame as a string."""
        return super().to_quil_or_debug()


@deprecated(
    version="4.0",
    reason="The WaveformReference class will be removed, consider using WaveformInvocation instead.",
)
class WaveformReference(WaveformInvocation):
    """Representation of a Waveform reference."""

    def __new__(cls, name: str) -> Self:
        """Initialize a new waveform reference."""
        return super().__new__(cls, name, {})


def _template_waveform_property(
    name: str, *, dtype: Optional[Union[type[int], type[float]]] = None, doc: Optional[str] = None
) -> property:
    """Initialize a getters, setters, and deleter for a parameter on a ``TemplateWaveform``.

    Should only be used inside of ``TemplateWaveform`` or one its base classes.

    :param name: The name of the property
    :param dtype: An optional parameter that takes the int or float type, and attempts
            to convert the underlying complex value by casting the real part to `dtype`. If set, this function will
            raise an error if the complex number has a non-zero imaginary part.
    :param doc: Docstring for the property.
    """

    def fget(self: "TemplateWaveform") -> Optional[ParameterDesignator]:
        parameter = self.get_parameter(name)
        if parameter is None or dtype is None:
            return parameter

        if dtype is int or dtype is float:
            if isinstance(parameter, dtype):
                return parameter
            if not isinstance(parameter, complex):
                raise TypeError(
                    f"Requested float for parameter {name}, but a non-numeric value of type {type(parameter)} was "
                    "found instead"
                )
            if parameter.imag != 0.0:
                raise ValueError(
                    f"Requested float for parameter {name}, but a complex number with a non-zero imaginary part was "
                    "found"
                )
            return dtype(parameter.real)
        raise TypeError(f"TemplateWaveform is not compatible with dtype {dtype}")

    def fset(self: "TemplateWaveform", value: ParameterDesignator) -> None:
        self.set_parameter(name, value)

    def fdel(self: "TemplateWaveform") -> None:
        self.set_parameter(name, None)

    return property(fget, fset, fdel, doc)


class TemplateWaveform(quil_rs.WaveformInvocation, QuilAtom):
    """Base class for creating waveform templates."""

    NAME: ClassVar[str]

    def __new__(
        cls,
        name: str,
        *,
        duration: float,
        **kwargs: Union[Optional[ParameterDesignator], Optional[ExpressionDesignator]],
    ) -> Self:
        """Initialize a new TemplateWaveform."""
        rs_parameters = {key: _convert_to_rs_expression(value) for key, value in kwargs.items() if value is not None}
        rs_parameters["duration"] = _convert_to_rs_expression(duration)
        return super().__new__(cls, name, rs_parameters)

    def out(self) -> str:
        """Return the waveform as a valid Quil string."""
        return str(self)

    def get_parameter(self, name: str) -> Optional[ParameterDesignator]:
        """Get a parameter in the waveform by name."""
        parameter = super().parameters.get(name, None)
        if parameter is None:
            return None
        return _convert_to_py_expression(parameter)

    def set_parameter(self, name: str, value: Optional[ParameterDesignator]) -> None:
        """Set a parameter with a value."""
        parameters = super().parameters
        if value is None:
            parameters.pop(name, None)
        else:
            parameters[name] = _convert_to_rs_expression(value)
        quil_rs.WaveformInvocation.parameters.__set__(self, parameters)  # type: ignore[attr-defined]

    duration = _template_waveform_property("duration", dtype=float)

    def num_samples(self, rate: float) -> int:
        """Return the number of samples in the reference implementation of the waveform.

        Note: this does not include any hardware-enforced alignment (cf. documentation for `samples`).

        :param rate: The sample rate, in Hz.
        :return: The number of samples.
        """
        duration = self.duration.real
        if self.duration.imag != 0.0:
            raise ValueError("Can't calculate number of samples with a complex duration")
        return int(np.ceil(duration * rate))

    def samples(self, rate: float) -> np.ndarray:
        """Generate samples of the waveform.

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
        """Build a TemplateWaveform from a ``quil`` waveform invocation.

        The ``quil`` package has no equivalent to ``TemplateWaveform``s, this function checks the name and properties of
        a ``quil`` ``WaveformInvocation`` to see if they potentially match a subclass of ``TemplateWaveform``. If a
        match is found and construction succeeds, then that type is returned. Otherwise, a generic
        ``WaveformInvocation`` is returned.
        """
        from pyquil.quiltwaveforms import (
            BoxcarAveragerKernel,
            DragGaussianWaveform,
            ErfSquareWaveform,
            FlatWaveform,
            GaussianWaveform,
            HrmGaussianWaveform,
        )

        template: type[TemplateWaveform]  # mypy needs a type annotation here to understand this.
        for template in [
            FlatWaveform,
            GaussianWaveform,
            DragGaussianWaveform,
            HrmGaussianWaveform,
            ErfSquareWaveform,
            BoxcarAveragerKernel,
        ]:
            if template.NAME != waveform.name:
                continue
            parameter_names = [
                parameter[0] for parameter in inspect.getmembers(template, lambda a: isinstance(a, property))
            ]
            if set(waveform.parameters.keys()).issubset(parameter_names):
                try:
                    parameters = {key: _convert_to_py_expression(value) for key, value in waveform.parameters.items()}
                    return template(**parameters)  # type: ignore[arg-type]
                except TypeError:
                    break

        return super().__new__(cls, waveform.name, waveform.parameters)

    def __str__(self) -> str:
        return super().to_quil_or_debug()


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
