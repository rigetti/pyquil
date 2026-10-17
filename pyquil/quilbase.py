##############################################################################
# Copyright 2016-2026 Rigetti Computing
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
"""Contains the core pyQuil objects that correspond to Quil instructions."""

import abc
import math
from collections.abc import Callable, Container, Sequence
from typing import TYPE_CHECKING, TypeAlias
from warnings import deprecated
import warnings

import numpy as np
from quil.expression import Expression, ExpressionDesignator, Parameter, ParameterDesignator
from quil.instructions import (
    Arithmetic,
    ArithmeticOperator,
    AttributeValue,
    BinaryLogic,
    BinaryOperator,
    CalibrationDefinition,
    CircuitDefinition,
    Comparison,
    ComparisonOperator,
    Convert,
    Declaration,
    Delay,
    Exchange,
    Fence,
    FrameDefinition,
    FrameIdentifier,
    GateDefinition,
    GateModifier,
    GateSpecification,
    HaltType,
    Instruction,
    Jump,
    JumpWhen,
    JumpUnless,
    Label,
    Load,
    MeasureCalibrationDefinition,
    MemoryReference,
    Move,
    NopType,
    PauliSum,
    PauliTargetDesignator,
    Qubit,
    QubitDesignator,
    Store,
    UnaryLogic,
    UnaryOperator,
    WaitType,
    WaveformDefinition,
)

# ``FrameDefinition`` (``DefFrame``) no longer exposes the named convenience properties
# (``direction``, ``initial_frequency``, ``hardware_object``, ``sample_rate``, ``center_frequency``,
# ``channel_delay``) that pre-v5 pyQuil provided; it only exposes the raw ``attributes`` dict /
# ``get_attribute``/``set_attribute``. Restore these as thin wrappers around the well-known,
# canonically-cased (kebab-case, uppercase) Quil attribute names.
# Note: there is no equivalent way to restore a settable ``.frame``/``.identifier`` property, since
# the native type has no mutable backing storage reachable from Python for it (see MIGRATION-NOTES.md).
_DEF_FRAME_ATTRIBUTES = {
    "direction": "DIRECTION",
    "initial_frequency": "INITIAL-FREQUENCY",
    "hardware_object": "HARDWARE-OBJECT",
    "sample_rate": "SAMPLE-RATE",
    "center_frequency": "CENTER-FREQUENCY",
    "channel_delay": "CHANNEL-DELAY",
}


def _def_frame_get_attribute(self: FrameDefinition, name: str) -> ParameterDesignator | str | None:
    value = self.get_attribute(name)
    match value:
        case None:
            return None
        case AttributeValue.String(s):
            return s
        case AttributeValue.Expression(e):
            return e.evaluate(partial=True)
    raise ValueError(f"Unexpected attribute value type: {type(value)}")


def _def_frame_set_attribute(self: FrameDefinition, name: str, value: ParameterDesignator | str | None) -> None:
    if value is None:
        return
    self.set_attribute(name, value)


def _make_def_frame_property(attribute_name: str) -> property:
    return property(
        lambda self: _def_frame_get_attribute(self, attribute_name),
        lambda self, value: _def_frame_set_attribute(self, attribute_name, value),
    )


for _snake_name, _canonical_name in _DEF_FRAME_ATTRIBUTES.items():
    setattr(FrameDefinition, _snake_name, _make_def_frame_property(_canonical_name))

# ``WaveformDefinition`` (``DefWaveform``) doesn't directly expose ``.parameters`` (only via
# ``.definition.parameters``); restore it as a convenience property for parity with ``.entries``/``.name``.
WaveformDefinition.parameters = property(lambda self: self.definition.parameters)  # type: ignore[method-assign]

# ``CalibrationDefinition`` (``DefCalibration``) exposes ``.name``/``.parameters``/``.qubits`` as
# read-only properties (only ``.identifier``/``.instructions`` are settable); restore settability by
# reconstructing the identifier, matching pre-v5 pyQuil ergonomics.


def _make_calibration_identifier_property(field_name: str) -> property:
    def _setter(self: CalibrationDefinition, value: object) -> None:
        identifier = self.identifier
        if field_name == "parameters":
            value = [Expression(v) for v in value]  # type: ignore[union-attr]
        setattr(identifier, field_name, value)
        self.identifier = identifier

    return property(lambda self: getattr(self.identifier, field_name), _setter)


for _field_name in ("name", "parameters", "qubits"):
    setattr(CalibrationDefinition, _field_name, _make_calibration_identifier_property(_field_name))

CalibrationDefinition.instrs = property(  # type: ignore[method-assign]
    lambda self: self.instructions,
    lambda self, value: setattr(self, "instructions", value),
)

# ``MeasureCalibrationDefinition`` (``DefMeasureCalibration``) similarly exposes ``.qubit``/
# ``.memory_reference`` as read-only properties (only ``.identifier``/``.instructions`` are
# settable); restore settability the same way, by reconstructing the identifier.


def _measure_calibration_qubit_setter(self: MeasureCalibrationDefinition, value: object) -> None:
    identifier = self.identifier
    identifier.qubit = value  # type: ignore[assignment]
    self.identifier = identifier


def _measure_calibration_memory_reference_setter(
    self: MeasureCalibrationDefinition, value: "MemoryReference | None"
) -> None:
    identifier = self.identifier
    identifier.target = value.name if value is not None else None
    self.identifier = identifier


MeasureCalibrationDefinition.qubit = property(  # type: ignore[method-assign]
    lambda self: self.identifier.qubit, _measure_calibration_qubit_setter
)
MeasureCalibrationDefinition.memory_reference = property(  # type: ignore[method-assign]
    lambda self: MemoryReference(self.identifier.target, 0) if self.identifier.target is not None else None,
    _measure_calibration_memory_reference_setter,
)
MeasureCalibrationDefinition.instrs = property(  # type: ignore[method-assign]
    lambda self: self.instructions,
    lambda self, value: setattr(self, "instructions", value),
)

# Re-exported for backwards compatibility.
from quil.instructions import (
    Call,
    Capture,
    Gate,
    Halt,
    Include,
    Measurement,
    Nop,
    Pragma,
    Pulse,
    RawCapture,
    Reset,
    SetFrequency,
    SetPhase,
    SetScale,
    ShiftFrequency,
    ShiftPhase,
    SwapPhases,
    Wait,
)

# Primes used to peel perfect powers apart in :func:`_integer_base_and_exponent`. An exponent
# whose prime factors all lie in this list is fully reducible; matrix dimensions never approach
# ``2 ** 61``, so this covers every realistic qudit decomposition.
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)


def _int_n_root(x: int, n: int) -> int:
    """Return the integer ``n``-th root of ``x`` (the largest ``m`` with ``m ** n <= x``)."""
    low, high = 1, x
    while low <= high:
        mid = (low + high) // 2
        mid_pow_n = mid**n
        if mid_pow_n == x:
            return mid
        elif mid_pow_n < x:
            low = mid + 1
        else:
            high = mid - 1
    return low - 1


def _is_prime(n: int) -> bool:
    """Return whether ``n`` is prime, via trial division seeded by :data:`_SMALL_PRIMES`."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    for d in range(_SMALL_PRIMES[-1] + 2, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def _integer_base_and_exponent(n: int) -> tuple[int, int] | None:
    """Decompose n as ``base ** exponent`` for a prime ``base``.

    Returns the prime ``d`` and the exponent ``k`` such that ``n == d ** k``, or ``None`` if
    ``n`` is not a prime power. This is the natural qudit decomposition: a matrix dimension
    ``d ** k`` describes ``k`` qudits of dimension ``d``.
    """
    if n < 2:
        return None

    base, exponent = n, 1
    for p in _SMALL_PRIMES:
        while (root := _int_n_root(base, p)) >= 2 and root**p == base:
            base, exponent = root, exponent * p
        if root < 2:
            break

    if not _is_prime(base):
        return None
    return base, exponent


def gate_definition_matrix(dg: GateDefinition) -> list[list[Expression]]:
    """Return the rows of a ``GATE ... AS MATRIX`` :py:class:`GateDefinition`'s unitary.

    :raises TypeError: If ``dg``'s specification is not a matrix (e.g. a permutation or
        Pauli sum gate definition).
    """
    if not isinstance(dg.specification, GateSpecification.Matrix):
        raise TypeError(f"GateDefinition {dg.name!r} is not defined by a matrix: {dg.specification=}.")
    return dg.specification._0


def _strip_modifiers(gate: Gate, limit: int | None = None) -> Gate:
    """Remove modifiers from :py:class:`Gate`.

    This function removes up to ``limit`` gate modifiers from the given gate,
    starting from the leftmost gate modifier.

    :param gate: A gate.
    :param limit: An upper bound on how many modifiers to remove.
    """
    if limit is None:
        limit = len(gate.modifiers)

    # We walk the modifiers from left-to-right, tracking indices to identify
    # qubits/params introduced by gate modifiers.
    #
    # Invariants:
    #   - gate.qubits[0:qubit_index] are qubits introduced by gate modifiers
    #   - gate.parameters[param_index:] are parameters introduced by gate modifiers
    qubit_index = 0
    param_index = len(gate.parameters)
    for m in gate.modifiers[:limit]:
        if m == GateModifier.CONTROLLED:
            qubit_index += 1
        elif m == GateModifier.FORKED:
            if param_index % 2 != 0:
                raise ValueError("FORKED gate has an invalid number of parameters.")
            param_index //= 2
            qubit_index += 1
        elif m == GateModifier.DAGGER:
            pass
        else:
            raise TypeError(f"Unsupported gate modifier {m}")

    stripped = Gate(
        gate.name,
        gate.parameters[:param_index],
        gate.qubits[qubit_index:],
        gate.modifiers[limit:],
    )
    return stripped


if TYPE_CHECKING:
    from pyquil.paulis import PauliSum


class AbstractInstruction(metaclass=abc.ABCMeta):  # noqa: B024  # ABC is used for `register`.
    """Abstract class for representing single instructions."""


class SimpleInstruction(AbstractInstruction):
    """Abstract class for representing 'simple' instructions."""


AbstractInstruction.register(Instruction)
for cls in Instruction.__subclasses__():
    AbstractInstruction.register(cls)

SimpleInstruction.register(NopType)
SimpleInstruction.register(HaltType)
SimpleInstruction.register(WaitType)

RESERVED_WORDS: Container[str] = [
    "DEFGATE",
    "DEFCIRCUIT",
    "MEASURE",
    "LABEL",
    "HALT",
    "JUMP",
    "JUMP-WHEN",
    "JUMP-UNLESS",
    "RESET",
    "WAIT",
    "NOP",
    "INCLUDE",
    "PRAGMA",
    "DECLARE",
    "NEG",
    "NOT",
    "AND",
    "IOR",
    "XOR",
    "TRUE",
    "FALSE",
    "OR",
]


class DeprecatedSubclass(type):
    def __new__(mcls, name, bases, namespace, *,
                base_cls: type,
                deprecation_msg: str | None = None,
                base_class_name: str | None = None,
                constructor: type | None = None,
                constructor_name: str | None = None,
                constructor_args: tuple[str, ...] | None = None,
                constructor_msg: str | None = None,
                instance_check_msg: str | None = None,
                subclass_check_msg: str | None = None,
                matches: Callable[[object], bool] | None = None,
    ):
        cls = super().__new__(mcls, name, bases, namespace)

        if deprecation_msg is None:
            deprecation_msg = f"`pyquil.quilbase.{name}` is deprecated"
        if base_class_name is None:
            module = base_cls.__module__.replace("quil._quil", "quil")
            base_class_name = f"{module}.{base_cls.__name__}"
        if constructor_name is None:
            constructor_name = base_class_name

        if constructor is None:
            # Partial-application classes define `__new__` to build the underlying `quil` object.
            constructor = namespace.get("__new__", base_cls.__new__)

        if constructor_msg is None:
            constructor_args_str = ", ".join(constructor_args or ())
            constructor_msg = f"use `{constructor_name}({constructor_args_str})` instead."
        if instance_check_msg is None:
            instance_check_msg = f"use `isinstance(obj, {base_class_name})` instead."
        if subclass_check_msg is None:
            subclass_check_msg = f"use `issubclass(cls, {base_class_name})` instead."

        cls._base_cls = base_cls
        cls._matches = matches
        cls._constructor = constructor
        cls._constructor_dep = f"{deprecation_msg}; {constructor_msg}"
        cls._isinstance_dep = f"{deprecation_msg}; {instance_check_msg}"
        cls._issubclass_dep = f"{deprecation_msg}; {subclass_check_msg}"
        return cls

    def __call__(cls, *args, **kwargs):
        warnings.warn(cls._constructor_dep, category=DeprecationWarning, stacklevel=2)
        return cls._constructor(cls._base_cls, *args, **kwargs)

    def __instancecheck__(cls, instance) -> bool:
        warnings.warn(cls._isinstance_dep, category=DeprecationWarning, stacklevel=2)
        return isinstance(instance, cls._base_cls) and (cls._matches is None or cls._matches(instance))

    def __subclasscheck__(cls, subclass) -> bool:
        warnings.warn(cls._issubclass_dep, category=DeprecationWarning, stacklevel=2)
        return issubclass(subclass, cls._base_cls)


def _dep_args[T](base: type[T], *consts: str, **kwargs: str):
    """Return a dict of arguments for DeprecatedSubclass metaclass."""

    base_name = base.__name__
    constructor_args = ", ".join(f"{k}={v}" for k, v in kwargs.items())

    isinstance_check_msg = f"(isinstance(obj, {base_name})"
    if consts:
        isinstance_check_msg += "".join(f" and obj.{c} == {kwargs[c]}" for c in consts)
    isinstance_check_msg += "\n".join((
        ")",
        "--or--",
        "match obj:",
        f"    case {base_name}({constructor_args}): ...",
    ))

    return dict(
        constructor_args=tuple(f"{k}={v}" for k, v in kwargs.items()),
        instance_check_msg=isinstance_check_msg,
    )


# These are essentially aliases for backwards compatibility.
@deprecated("`quil` classes should not be subclassed; use them directly instead")
class ArithmeticBinaryOp(
    metaclass=DeprecatedSubclass, base_cls=Arithmetic, constructor_args=("operator", "destination", "source")
): ...


@deprecated("`ClassicalComparison` is deprecated; use `Comparison` instead")
class ClassicalComparison(
    metaclass=DeprecatedSubclass,
    base_cls=Comparison,
    constructor_args=("operator", "destination", "lhs", "rhs"),
): ...


@deprecated("`ClassicalConvert` is deprecated; use `Convert` instead")
class ClassicalConvert(metaclass=DeprecatedSubclass, base_cls=Convert, constructor_args=("destination", "source")): ...


@deprecated("`ClassicalExchange` is deprecated; use `Exchange` instead")
class ClassicalExchange(metaclass=DeprecatedSubclass, base_cls=Exchange, constructor_args=("left", "right")): ...


@deprecated("`ClassicalLoad` is deprecated; use `Load` instead")
class ClassicalLoad(
    metaclass=DeprecatedSubclass, base_cls=Load, constructor_args=("destination", "source", "offset")
): ...


@deprecated("`ClassicalMove` is deprecated; use `Move` instead")
class ClassicalMove(metaclass=DeprecatedSubclass, base_cls=Move, constructor_args=("destination", "source")): ...


@deprecated("`ClassicalStore` is deprecated; use `Store` instead")
class ClassicalStore(
    metaclass=DeprecatedSubclass, base_cls=Store, constructor_args=("destination", "offset", "source")
): ...


@deprecated("`ClasssicalExchange` is deprecated; use `Exchange` instead")
class ClasssicalExchange(metaclass=DeprecatedSubclass, base_cls=Exchange, constructor_args=("left", "right")): ...


@deprecated("`Declare` is deprecated; use `Declaration` instead")
class Declare(
    metaclass=DeprecatedSubclass,
    base_cls=Declaration,
    constructor_args=("name", "memory_type", "memory_size", "shared_region", "offsets"),
): ...


@deprecated("`DefCalibration` is deprecated; use `CalibrationDefinition` instead")
class DefCalibration(
    metaclass=DeprecatedSubclass,
    base_cls=CalibrationDefinition,
    constructor_args=("identifier", "instructions", "qubits", "instrs", "modifiers"),
): ...


@deprecated("`DefCircuit` is deprecated; use `CircuitDefinition` instead")
class DefCircuit(
    metaclass=DeprecatedSubclass,
    base_cls=CircuitDefinition,
    constructor_args=("name", "parameters", "qubit_variables", "instructions"),
): ...


@deprecated("`DefFrame` is deprecated; use `FrameDefinition` instead")
class DefFrame(
    metaclass=DeprecatedSubclass, base_cls=FrameDefinition, constructor_args=("identifier", "attributes")
): ...


@deprecated("`DefMeasureCalibration` is deprecated; use `MeasureCalibrationDefinition` instead")
class DefMeasureCalibration(
    metaclass=DeprecatedSubclass,
    base_cls=MeasureCalibrationDefinition,
    constructor_args=("qubit", "memory_reference", "instructions"),
): ...


@deprecated("`DefWaveform` is deprecated; use `WaveformDefinition` instead")
class DefWaveform(
    metaclass=DeprecatedSubclass, base_cls=WaveformDefinition, constructor_args=("name", "parameters", "entries")
): ...


@deprecated("`Frame` is deprecated; use `FrameIdentifier` instead")
class Frame(metaclass=DeprecatedSubclass, base_cls=FrameIdentifier, constructor_args=("qubits", "name")): ...


@deprecated("`JumpTarget` is deprecated; use `Label` instead")
class JumpTarget(metaclass=DeprecatedSubclass, base_cls=Label, constructor_args=("target",)): ...


@deprecated("`LogicalBinaryOp` is deprecated; use `BinaryLogic` instead")
class LogicalBinaryOp(
    metaclass=DeprecatedSubclass, base_cls=BinaryLogic, constructor_args=("operator", "destination", "source")
): ...


@deprecated("`UnaryClassicalInstruction` is deprecated; use `UnaryLogic` instead")
class UnaryClassicalInstruction(
    metaclass=DeprecatedSubclass, base_cls=UnaryLogic, constructor_args=("operator", "operand")
): ...


# These are also for backwards compatibility with older classes,
# but in these cases, the original subclasses were essentially partial functions.
# The older classes were removed because they can't round-trip through quil-based functions.
@deprecated("UnaryLogic subclasses are deprecated")
class ClassicalNeg(
    metaclass=DeprecatedSubclass,
    base_cls=UnaryLogic,
    matches=lambda obj: obj.operator == UnaryOperator.NEG,
    **_dep_args(UnaryLogic, "operator", operator="UnaryOperator.NEG", operand="target"),
):
    def __new__(cls, target: MemoryReference) -> UnaryLogic:
        """Return a NEG instruction."""
        return UnaryLogic(UnaryOperator.NEG, target)


@deprecated("UnaryLogic subclasses are deprecated")
class ClassicalNot(
    metaclass=DeprecatedSubclass,
    base_cls=UnaryLogic,
    matches=lambda obj: obj.operator == UnaryOperator.NOT,
    **_dep_args(UnaryLogic, "operator", operator="UnaryOperator.NOT", operand="target"),
):
    def __new__(cls, target: MemoryReference) -> UnaryLogic:
        """Return a NOT instruction."""
        return UnaryLogic(UnaryOperator.NOT, target)


@deprecated("BinaryLogic subclasses are deprecated")
class ClassicalAnd(
    metaclass=DeprecatedSubclass,
    base_cls=BinaryLogic,
    matches=lambda obj: obj.operator == BinaryOperator.AND,
    **_dep_args(BinaryLogic, "operator", operator="BinaryOperator.AND", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int) -> BinaryLogic:
        """Return an AND instruction."""
        return BinaryLogic(BinaryOperator.AND, left, right)


@deprecated("BinaryLogic subclasses are deprecated")
class ClassicalInclusiveOr(
    metaclass=DeprecatedSubclass,
    base_cls=BinaryLogic,
    matches=lambda obj: obj.operator == BinaryOperator.IOR,
    **_dep_args(BinaryLogic, "operator", operator="BinaryOperator.IOR", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int) -> BinaryLogic:
        """Return an IOR instruction."""
        return BinaryLogic(BinaryOperator.IOR, left, right)


@deprecated("BinaryLogic subclasses are deprecated")
class ClassicalExclusiveOr(
    metaclass=DeprecatedSubclass,
    base_cls=BinaryLogic,
    matches=lambda obj: obj.operator == BinaryOperator.XOR,
    **_dep_args(BinaryLogic, "operator", operator="BinaryOperator.XOR", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int) -> BinaryLogic:
        """Return an XOR instruction."""
        return BinaryLogic(BinaryOperator.XOR, left, right)


@deprecated("BinaryLogic subclasses are deprecated")
class ClassicalArithmeticShiftRight(
    metaclass=DeprecatedSubclass,
    base_cls=BinaryLogic,
    matches=lambda obj: obj.operator == BinaryOperator.ASHR,
    **_dep_args(BinaryLogic, "operator", operator="BinaryOperator.ASHR", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int) -> BinaryLogic:
        """Return an arithmetic shift-right instruction."""
        return BinaryLogic(BinaryOperator.ASHR, left, right)


@deprecated("BinaryLogic subclasses are deprecated")
class ClassicalShiftLeft(
    metaclass=DeprecatedSubclass,
    base_cls=BinaryLogic,
    matches=lambda obj: obj.operator == BinaryOperator.SHL,
    **_dep_args(BinaryLogic, "operator", operator="BinaryOperator.SHL", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int) -> BinaryLogic:
        """Return a shift-left instruction."""
        return BinaryLogic(BinaryOperator.SHL, left, right)


@deprecated("BinaryLogic subclasses are deprecated")
class ClassicalShiftRight(
    metaclass=DeprecatedSubclass,
    base_cls=BinaryLogic,
    matches=lambda obj: obj.operator == BinaryOperator.SHR,
    **_dep_args(BinaryLogic, "operator", operator="BinaryOperator.SHR", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int) -> BinaryLogic:
        """Return a shift-right instruction."""
        return BinaryLogic(BinaryOperator.SHR, left, right)


@deprecated("Arithmetic subclasses are deprecated")
class ClassicalAdd(
    metaclass=DeprecatedSubclass,
    base_cls=Arithmetic,
    matches=lambda obj: obj.operator == ArithmeticOperator.ADD,
    **_dep_args(Arithmetic, "operator", operator="ArithmeticOperator.ADD", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int | float) -> Arithmetic:
        """Return an ADD instruction."""
        return Arithmetic(ArithmeticOperator.ADD, left, right)


@deprecated("Arithmetic subclasses are deprecated")
class ClassicalSub(
    metaclass=DeprecatedSubclass,
    base_cls=Arithmetic,
    matches=lambda obj: obj.operator == ArithmeticOperator.SUBTRACT,
    **_dep_args(Arithmetic, "operator", operator="ArithmeticOperator.SUBTRACT", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int | float) -> Arithmetic:
        """Return a SUB instruction."""
        return Arithmetic(ArithmeticOperator.SUBTRACT, left, right)


@deprecated("Arithmetic subclasses are deprecated")
class ClassicalMul(
    metaclass=DeprecatedSubclass,
    base_cls=Arithmetic,
    matches=lambda obj: obj.operator == ArithmeticOperator.MULTIPLY,
    **_dep_args(Arithmetic, "operator", operator="ArithmeticOperator.MULTIPLY", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int | float) -> Arithmetic:
        """Return a MUL instruction."""
        return Arithmetic(ArithmeticOperator.MULTIPLY, left, right)


@deprecated("Arithmetic subclasses are deprecated")
class ClassicalDiv(
    metaclass=DeprecatedSubclass,
    base_cls=Arithmetic,
    matches=lambda obj: obj.operator == ArithmeticOperator.DIVIDE,
    **_dep_args(Arithmetic, "operator", operator="ArithmeticOperator.DIVIDE", destination="left", source="right"),
):
    def __new__(cls, left: MemoryReference, right: MemoryReference | int | float) -> Arithmetic:
        """Return a DIV instruction."""
        return Arithmetic(ArithmeticOperator.DIVIDE, left, right)


@deprecated("Comparison subclasses are deprecated")
class ClassicalEqual(
    metaclass=DeprecatedSubclass,
    base_cls=Comparison,
    matches=lambda obj: obj.operator == ComparisonOperator.EQUAL,
    **_dep_args(Comparison, "operator", operator="ComparisonOperator.EQUAL", target="target", left="left", right="right"),
):
    def __new__(cls, target: MemoryReference, left: MemoryReference, right: MemoryReference | int | float) -> Comparison:
        """Return an EQ comparison instruction."""
        return Comparison(ComparisonOperator.EQUAL, target, left, right)


@deprecated("Comparison subclasses are deprecated")
class ClassicalLessThan(
    metaclass=DeprecatedSubclass,
    base_cls=Comparison,
    matches=lambda obj: obj.operator == ComparisonOperator.LESS_THAN,
    **_dep_args(
        Comparison, "operator", operator="ComparisonOperator.LESS_THAN", target="target", left="left", right="right"
    ),
):
    def __new__(cls, target: MemoryReference, left: MemoryReference, right: MemoryReference | int | float) -> Comparison:
        """Return a LT comparison instruction."""
        return Comparison(ComparisonOperator.LESS_THAN, target, left, right)


@deprecated("Comparison subclasses are deprecated")
class ClassicalLessEqual(
    metaclass=DeprecatedSubclass,
    base_cls=Comparison,
    matches=lambda obj: obj.operator == ComparisonOperator.LESS_THAN_OR_EQUAL,
    **_dep_args(
        Comparison,
        "operator",
        operator="ComparisonOperator.LESS_THAN_OR_EQUAL",
        target="target",
        left="left",
        right="right",
    ),
):
    def __new__(cls, target: MemoryReference, left: MemoryReference, right: MemoryReference | int | float) -> Comparison:
        """Return a LE comparison instruction."""
        return Comparison(ComparisonOperator.LESS_THAN_OR_EQUAL, target, left, right)


@deprecated("Comparison subclasses are deprecated")
class ClassicalGreaterThan(
    metaclass=DeprecatedSubclass,
    base_cls=Comparison,
    matches=lambda obj: obj.operator == ComparisonOperator.GREATER_THAN,
    **_dep_args(
        Comparison, "operator", operator="ComparisonOperator.GREATER_THAN", target="target", left="left", right="right"
    ),
):
    def __new__(cls, target: MemoryReference, left: MemoryReference, right: MemoryReference | int | float) -> Comparison:
        """Return a GT comparison instruction."""
        return Comparison(ComparisonOperator.GREATER_THAN, target, left, right)


@deprecated("Comparison subclasses are deprecated")
class ClassicalGreaterEqual(
    metaclass=DeprecatedSubclass,
    base_cls=Comparison,
    matches=lambda obj: obj.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL,
    **_dep_args(
        Comparison,
        "operator",
        operator="ComparisonOperator.GREATER_THAN_OR_EQUAL",
        target="target",
        left="left",
        right="right",
    ),
):
    def __new__(cls, target: MemoryReference, left: MemoryReference, right: MemoryReference | int | float) -> Comparison:
        """Return a GE comparison instruction."""
        return Comparison(ComparisonOperator.GREATER_THAN_OR_EQUAL, target, left, right)


@deprecated("DelayFrames is deprecated, use Delay.from_frames instead.")
class DelayFrames(
    metaclass=DeprecatedSubclass,
    base_cls=Delay,
    constructor_name="quil.instructions.Delay.from_frames",
    constructor_args=("duration", "frames"),
    instance_check_msg="use `isinstance(obj, quil.instructions.Delay) and obj.has_frames()` instead.",
    matches=lambda obj: obj.has_frames(),
):
    def __new__(cls, frames: list[FrameIdentifier], duration: float) -> Delay:
        """Return a DELAY instruction operating on frames.

        The `DelayFrames` class was removed because there is no corresponding Quil instruction,
        meaning when added to a `Program`, it must be converted into a `Delay` instruction,
        and hence is not returned from a `Program` as its original class.
        To avoid confusion, the class was removed.
        To replace creating instances, use the static `Delay.from_frames` alternate constructor;
        to replace `isinstance` checks, use `isinstance(x, Delay)` and `x.has_frames()`
        to check whether or not the `Delay` includes frames.
        """
        return Delay.from_frames(duration, frames)


@deprecated("DelayQubits is deprecated, use Delay instead.")
class DelayQubits(
    metaclass=DeprecatedSubclass,
    base_cls=Delay,
    constructor_args=("duration", "qubits"),
    instance_check_msg="use `isinstance(obj, quil.instructions.Delay) and not obj.has_frames()` instead.",
    matches=lambda obj: not obj.has_frames(),
):
    def __new__(cls, qubits: Sequence[QubitDesignator], duration: float) -> Delay:
        """Return a DELAY instruction operating on qubits.

        The `DelayQubits` class was removed because there is no corresponding Quil instruction,
        meaning when added to a `Program`, it must be converted into a `Delay` instruction,
        and hence is not returned from a `Program` as its original class.
        To avoid confusion, the class was removed.
        To replace creating instances, use the `Delay(duration, qubits)` constructor;
        to replace `isinstance` checks, use `isinstance(x, Delay)` and `not x.has_frames()`
        to check whether or not the `Delay` includes frames.
        """
        return Delay(duration, qubits)


@deprecated("FenceAll is deprecated, use Fence([]) instead.")
class FenceAll(
    metaclass=DeprecatedSubclass,
    base_cls=Fence,
    constructor_args=("[]",),
    instance_check_msg="use `isinstance(obj, quil.instructions.Fence) and not obj.qubits` instead.",
    matches=lambda obj: not obj.qubits,
):
    def __new__(cls) -> Fence:
        """Return a FENCE instruction that operates on all qubits."""
        return Fence([])


@deprecated("ResetQubit is deprecated, use Reset(qubit) instead.")
class ResetQubit(
    metaclass=DeprecatedSubclass,
    base_cls=Reset,
    constructor_args=("qubit",),
    instance_check_msg="use `isinstance(obj, quil.instructions.Reset) and obj.qubit is not None` instead.",
    matches=lambda obj: obj.qubit is not None,
):
    def __new__(cls, qubit: QubitDesignator) -> Reset:
        """Return a RESET instruction that operates on a single qubit."""
        if qubit is None:
            raise TypeError("qubit should not be None")
        return Reset(qubit)


@deprecated("GateDefinition subclasses are deprecated -- provide a GateSpecification directly")
class DefGate(
    metaclass=DeprecatedSubclass,
    base_cls=GateDefinition,
    constructor_args=("name", "parameters", "specification=GateSpecification.Matrix(matrix)"),
    instance_check_msg=(
        "use `isinstance(obj, quil.instructions.GateDefinition)`"
        " and `isinstance(obj.specification, GateSpecification.Matrix)` instead."
    ),
    matches=lambda obj: isinstance(obj.specification, GateSpecification.Matrix),
):
    def __new__(
        cls,
        name: str,
        matrix: Sequence[Sequence[ExpressionDesignator]] | np.ndarray,
        parameters: list[Parameter] | None = None,
    ) -> GateDefinition:
        """Initialize a new gate definition from a unitary matrix."""
        return GateDefinition(name, parameters or [], GateSpecification.Matrix(matrix))


@deprecated("GateDefinition subclasses are deprecated -- provide a GateSpecification directly")
class DefPermutationGate(
    metaclass=DeprecatedSubclass,
    base_cls=GateDefinition,
    constructor_args=("name", "parameters=[]", "specification=GateSpecification.Permutation(permutation)"),
    instance_check_msg=(
        "use `isinstance(obj, quil.instructions.GateDefinition)`"
        " and `isinstance(obj.specification, GateSpecification.Permutation)` instead."
    ),
    matches=lambda obj: isinstance(obj.specification, GateSpecification.Permutation),
):
    def __new__(cls, name: str, permutation: list[int] | np.ndarray) -> GateDefinition:
        """Initialize a new gate definition with a permutation."""
        return GateDefinition(name, [], GateSpecification.Permutation(permutation))

if TYPE_CHECKING:
    from pyquil.paulis import (PauliSum as _PauliSum, PauliTerm as _PauliTerm)

from quil.instructions import PauliSum, PauliTerm

@deprecated("GateDefinition subclasses are deprecated -- provide a GateSpecification directly")
class DefGateByPaulis(
    metaclass=DeprecatedSubclass,
    base_cls=GateDefinition,
    constructor_args=(
        "name=gate_name",
        "parameters=parameters",
        "specification=GateSpecification.PauliSum(PauliSum(body))",
    ),
    instance_check_msg=(
        "use `isinstance(obj, quil.instructions.GateDefinition)`"
        " and `isinstance(obj.specification, GateSpecification.PauliSum)` instead."
    ),
    matches=lambda obj: isinstance(obj.specification, GateSpecification.PauliSum),
):
    def __new__(
        cls, gate_name: str, parameters: list[Parameter], arguments: list[PauliTargetDesignator],
        body: PauliSum | _PauliSum | Sequence[PauliTerm] | Sequence[_PauliTerm]
    ) -> GateDefinition:
        """Return a DEFGATE instruction with parameters."""
        # A valid call to the original `PauliSum.__init__` means `body` is a `Sequence[PauliTerm]`.

        terms = list(body) or [PauliTerm("I", None, 0)]

        for i, term in enumerate(terms):
            if isinstance(term, _PauliTerm):
                terms[i] = PauliTerm.from_list(
                    [(gate, arg) for arg, gate in term],
                    coefficient=term.coefficient
                )
            elif not isinstance(term, PauliTerm):
                raise TypeError(f"All terms in the body must be instances of PauliTerm, but term {i} is {type(term)}.")

        specification = GateSpecification.PauliSum(PauliSum(terms=terms, arguments=arguments))
        return GateDefinition(gate_name, parameters, specification)


