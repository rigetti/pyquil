##############################################################################
# Copyright 2017-2026 Rigetti Computing
#
#    Licensed under the Apache License Version 2.0 (the "License");
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

import abc
from collections.abc import Mapping
from fractions import Fraction
from typing import TypeAlias

import numpy as np
from quil.expression import Address, Expression, ExpressionDesignator, FunctionCallExpression, InfixExpression, Parameter, ParameterDesignator, PrefixExpression
from quil.instructions import (
    FrameIdentifier,
    MemoryReference,
    Qubit,
    FixedQubit,
    FormalArgument,
    QubitPlaceholder,
    QubitDesignator,
    Target,
    TargetPlaceholder,
    WaveformInvocation,
)
from quil.waveform import Waveform
from typing_extensions import deprecated

# Imports for backwards compatibility.
from quil.instructions import (
    MemoryReferenceDesignator,
    unpack_classical_reg as _unpack_classical_reg_rs,
)

from quil.expression import (
    ExpressionValueDesignator,
    quil_sin,
    quil_cos,
    quil_sqrt,
    quil_exp,
    quil_cis,
)


class QuilAtom(metaclass=abc.ABCMeta):  # noqa: B024  # ABC is used for `register`.
    """Abstract class for atomic elements of Quil."""


class TemplateWaveform(metaclass=abc.ABCMeta):  # noqa: B024  # ABC is used for `register`.
    """Abstract class for waveforms from Quil."""


# Aliases for backwards compatibility.
Frame: TypeAlias = FrameIdentifier
ParameterSubstitutionsMapDesignator: TypeAlias = Mapping[str | MemoryReference, complex | list[complex]]
BinaryExp: TypeAlias = InfixExpression
Function: TypeAlias = FunctionCallExpression


class Label(QuilAtom):
    """Representation of a (fixed) Quil label target, e.g. the ``@my-label`` in ``JUMP @my-label``.

    .. note::
        This is distinct from ``quil.instructions.Label``, which represents the ``LABEL @my-label``
        *instruction* (re-exported from this package as ``pyquil.quilbase.JumpTarget``); this class
        represents only the bare ``@my-label`` target referenced by ``JUMP``/``JUMP-WHEN``/``JUMP-UNLESS``.

    :param label_name: The label name.
    """

    def __init__(self, label_name: str):
        """Initialize a new label."""
        self.target = Target.Fixed(label_name)

    @staticmethod
    def _from_rs_target(target: Target) -> "Label":
        match target:
            case Target.Fixed(name):
                return Label(name)
        raise ValueError(f"{target} is not a fixed Target")

    def out(self) -> str:
        """Return the label as a valid Quil string."""
        return self.target.to_quil()

    @property
    def name(self) -> str:
        """Return the label name."""
        match self.target:
            case Target.Fixed(name):
                return name
        raise RuntimeError(f"Label {self} is not a fixed target")

    @name.setter
    def name(self, label_name: str) -> None:
        self.target = Target.Fixed(label_name)

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

    def __init__(self, prefix: str = "L", *, placeholder: TargetPlaceholder | None = None):
        """Initialize a new label placeholder."""
        self.target = Target.Placeholder(placeholder if placeholder is not None else TargetPlaceholder(prefix))

    @staticmethod
    def _from_rs_target(target: Target) -> "LabelPlaceholder":
        match target:
            case Target.Placeholder(placeholder):
                return LabelPlaceholder(placeholder=placeholder)
        raise ValueError(f"{target} is not a placeholder Target")

    @property
    def prefix(self) -> str:
        """Get the prefix of the label placeholder."""
        match self.target:
            case Target.Placeholder(placeholder):
                return placeholder.base_label
        raise RuntimeError(f"Label {self} is not a placeholder target")

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


def _convert_to_py_target(target: Target) -> "Label | LabelPlaceholder":
    """Convert a native ``Target`` (as returned from ``Jump``/``JumpWhen``/``JumpUnless.target``) to a pyQuil atom."""
    match target:
        case Target.Fixed(_):
            return Label._from_rs_target(target)
        case Target.Placeholder(_):
            return LabelPlaceholder._from_rs_target(target)
    raise ValueError(f"{target} is not a valid Target")


def _convert_to_rs_target(label: "Label | LabelPlaceholder | Target | str") -> Target:
    """Convert a pyQuil label atom (or a native ``Target``/``str``) to a native ``Target``."""
    if isinstance(label, Target):
        return label
    if isinstance(label, str):
        return Target.Fixed(label)
    if isinstance(label, (Label, LabelPlaceholder)):
        return label.target
    raise ValueError(f"{type(label)} is not a valid Label designator")


QuilAtom.register(Qubit)
QuilAtom.register(FixedQubit)
QuilAtom.register(FormalArgument)
QuilAtom.register(QubitPlaceholder)
QuilAtom.register(Parameter)
QuilAtom.register(MemoryReference)


# The native ``QubitPlaceholder.__str__``/``__repr__`` renders e.g. ``QubitPlaceholder(0x...)``.
# Pre-v5 pyQuil rendered placeholders without the redundant "Qubit" prefix (e.g. in
# ``PauliTerm.__repr__``, which interpolates the qubit designator directly); restore that
# formatting so string-matching user code (and reprs embedded in other objects) is unaffected.
def _qubit_placeholder_str(self: QubitPlaceholder) -> str:
    return f"Placeholder({hex(id(self))})"


QubitPlaceholder.__str__ = _qubit_placeholder_str  # type: ignore[method-assign]
QubitPlaceholder.__repr__ = _qubit_placeholder_str  # type: ignore[method-assign]
QuilAtom.register(WaveformInvocation)


def unpack_classical_reg(classical_reg: MemoryReferenceDesignator) -> MemoryReference:
    """Get the address for a classical register, from a convenient representation.

    This is a thin wrapper around ``quil.instructions.unpack_classical_reg`` that additionally
    accepts a ``(name, index)`` tuple/list, or a bare ``str`` (interpreted as ``name[0]``), for
    backwards compatibility with common pyQuil idioms for addressing memory, e.g. ``("ro", 0)``
    or just ``"ro"``.

    :param classical_reg: A ``MemoryReference``, a ``(name, index)`` tuple/list, a bare ``str``
        register name, or a Declaration.
    :return: A ``MemoryReference``.
    """
    if isinstance(classical_reg, (tuple, list)):
        if len(classical_reg) == 1:
            (name,) = classical_reg
            return MemoryReference(name, 0)
        name, index = classical_reg
        return MemoryReference(name, index)
    if isinstance(classical_reg, str):
        return MemoryReference(classical_reg, 0)
    return _unpack_classical_reg_rs(classical_reg)


@deprecated("Use `Qubit(qubit)` instead")
def unpack_qubit(qubit: QubitDesignator) -> Qubit:
    """Get a qubit from an object.

    :param qubit: the qubit designator to unpack.
    :return: A Qubit or QubitPlaceholder instance
    """
    return Qubit(qubit)


def qubit_index(qubit: QubitDesignator) -> int:
    """Get the index of a QubitDesignator.

    :param qubit: the qubit designator.
    :return: An int that is the qubit index.
    """
    match qubit:
        case FixedQubit(index) | int(index):
            return index
    raise TypeError(f"No index for Qubit: {qubit}")


@deprecated("Use `expression.evaluate(variables, memory_references, partial=True)` instead")
def substitute(expr: ExpressionDesignator, d: ParameterSubstitutionsMapDesignator) -> ExpressionDesignator:
    r"""Explicitly evaluate as much of ``expr`` as possible, using substitutions from `d`.

    This supports substitution of both parameters and memory references.
    Each memory reference must be individually assigned a value at each memory offset to be substituted.

    :param expr: The expression whose parameters or memory references are to be substituted.
    :param d: Numerical substitutions for parameters or memory references.
    :return: A partially simplified Expression, or a number.
    """
    return _substitute(expr, d)


def _substitute(expr: ExpressionDesignator, d: ParameterSubstitutionsMapDesignator) -> ExpressionDesignator:
    return expr.substitute(d) if isinstance(expr, Expression) else expr


substitute_array = np.vectorize(
    _substitute,
    otypes=[object],
    signature="()->()",
    excluded=[1],
    doc="""Apply ``substitute`` to all elements of an array ``a`` and return the resulting array.

    :param a: The array of expressions whose parameters or memory references are to be substituted.
    :param d: Numerical substitutions for parameters or memory references, for all array elements.
    :return: An array of partially substituted Expressions, or numbers.
    """,
)


def _contained_mrefs(expression: ExpressionDesignator, current: set[MemoryReference] | None=None) -> set[MemoryReference]:
    """Determine which memory references are contained in this expression.

    :param expression: expression involving parameters
    :param current: set of memory references already found in the expression
    :return: set of parameters contained in this expression
    """
    if current is None:
        current = set()

    match expression:
        case Address(mref):
            current.add(mref)
        case InfixExpression(left=left, right=right):
            _contained_mrefs(left, current)
            _contained_mrefs(right, current)
        case PrefixExpression(expression=e) | FunctionCallExpression(expression=e):
            _contained_mrefs(e, current)

    return current


@deprecated("Use `Expression(a) + Expression(b)` instead")
def Add(a: ExpressionDesignator, b: ExpressionDesignator) -> Expression:
    """Return an expression representing the sum of two expressions."""
    return Expression(a) + Expression(b)


@deprecated("Use `Expression(a) - Expression(b)` instead")
def Sub(a: ExpressionDesignator, b: ExpressionDesignator) -> Expression:
    """Return an expression representing the difference of two expressions."""
    return Expression(a) - Expression(b)


@deprecated("Use `Expression(a) * Expression(b)` instead")
def Mul(a: ExpressionDesignator, b: ExpressionDesignator) -> Expression:
    """Return an expression representing the product of two expressions."""
    return Expression(a) * Expression(b)


@deprecated("Use `Expression(a) / Expression(b)` instead")
def Div(a: ExpressionDesignator, b: ExpressionDesignator) -> Expression:
    """Return an expression representing the quotient of two expressions."""
    return Expression(a) / Expression(b)


@deprecated("Use `Expression(a) ** Expression(b)` instead")
def Pow(a: ExpressionDesignator, b: ExpressionDesignator) -> Expression:
    """Return an expression representing the power of two expressions."""
    return Expression(a) ** Expression(b)


def _check_for_pi(element: float) -> str:
    """Return the float as a string, expressing the float as a multiple of pi if possible.

    More specifically, check to see if there exists a rational number r = p/q in reduced form for which the
    difference between element/np.pi and r is small and q <= 8.

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


@deprecated("This function has been superseded by the `quil` package and will be removed soon.")
def format_parameter(element: ParameterDesignator) -> str:
    """Format a particular parameter.

    Essentially the same as built-in formatting except using 'i' instead of 'j' for the imaginary number.

    :param element: The parameter to format for Quil output.
    """
    if isinstance(element, (int, np.integer)):
        return repr(int(element))
    elif isinstance(element, (float, np.floating)):
        return _check_for_pi(float(element))
    elif isinstance(element, (complex, np.complexfloating)):
        element = complex(element.real, element.imag)
        out = ""
        r = element.real
        i = element.imag
        if i == 0:
            return repr(r)

        if r != 0:
            out += repr(r)

        if i == 1:
            if np.isclose(r, 0, atol=1e-14):
                out = "i"
            else:
                out += "+i"
        elif i == -1:
            if np.isclose(r, 0, atol=1e-14):
                out = "-i"
            else:
                out += "-i"
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
        return str(element)
    raise AssertionError(f"Invalid parameter: {element}")
