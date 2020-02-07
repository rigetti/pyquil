##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
from warnings import warn
from typing import Callable, Mapping, Optional, Tuple, Union

import numpy as np

from pyquil.quilatom import (
    Addr,
    Expression,
    MemoryReference,
    MemoryReferenceDesignator,
    ParameterDesignator,
    QubitDesignator,
    unpack_classical_reg,
    unpack_qubit,
)
from pyquil.quilbase import (
    AbstractInstruction,
    Gate,
    Halt,
    Reset,
    ResetQubit,
    Measurement,
    Nop,
    Wait,
    ClassicalNeg,
    ClassicalNot,
    ClassicalAnd,
    ClassicalInclusiveOr,
    ClassicalExclusiveOr,
    ClassicalEqual,
    ClassicalGreaterEqual,
    ClassicalGreaterThan,
    ClassicalLessEqual,
    ClassicalLessThan,
    ClassicalMove,
    ClassicalExchange,
    ClassicalConvert,
    ClassicalLoad,
    ClassicalStore,
    ClassicalAdd,
    ClassicalSub,
    ClassicalMul,
    ClassicalDiv,
)


def unpack_reg_val_pair(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: Union[MemoryReferenceDesignator, int, float],
) -> Tuple[MemoryReference, Union[MemoryReference, int, float]]:
    """
    Helper function for typechecking / type-coercing arguments to constructors for binary classical
    operators.

    :param classical_reg1: Specifier for the classical memory address to be modified.
    :param classical_reg2: Specifier for the second argument: a classical memory address or an
        immediate value.
    :return: A pair of pyQuil objects suitable for use as operands.
    """
    left = unpack_classical_reg(classical_reg1)
    if isinstance(classical_reg2, (float, int)):
        return left, classical_reg2
    return left, unpack_classical_reg(classical_reg2)


def prepare_ternary_operands(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: MemoryReferenceDesignator,
    classical_reg3: Union[MemoryReferenceDesignator, int, float],
) -> Tuple[MemoryReference, MemoryReference, Union[MemoryReference, int, float]]:
    """
    Helper function for typechecking / type-coercing arguments to constructors for ternary
    classical operators.

    :param classical_reg1: Specifier for the classical memory address to be modified.
    :param classical_reg2: Specifier for the left operand: a classical memory address.
    :param classical_reg3: Specifier for the right operand: a classical memory address or an
        immediate value.
    :return: A triple of pyQuil objects suitable for use as operands.
    """
    if isinstance(classical_reg1, int):
        raise TypeError("Target operand of comparison must be a memory address")
    classical_reg1 = unpack_classical_reg(classical_reg1)
    if isinstance(classical_reg2, int):
        raise TypeError("Left operand of comparison must be a memory address")
    classical_reg2 = unpack_classical_reg(classical_reg2)
    if not isinstance(classical_reg3, (float, int)):
        classical_reg3 = unpack_classical_reg(classical_reg3)

    return classical_reg1, classical_reg2, classical_reg3


def I(qubit: QubitDesignator) -> Gate:
    """Produces the I identity gate::

        I = [1, 0]
            [0, 1]

    This gate is a single qubit identity gate.
    Note that this gate is different that the NOP instruction as noise channels
    are typically still applied during the duration of identity gates. Identities will
    also block parallelization like any other gate.

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="I", params=[], qubits=[unpack_qubit(qubit)])


def X(qubit: QubitDesignator) -> Gate:
    """Produces the X ("NOT") gate::

        X = [[0, 1],
             [1, 0]]

    This gate is a single qubit X-gate.

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="X", params=[], qubits=[unpack_qubit(qubit)])


def Y(qubit: QubitDesignator) -> Gate:
    """Produces the Y gate::

        Y = [[0, 0 - 1j],
             [0 + 1j, 0]]

    This gate is a single qubit Y-gate.

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="Y", params=[], qubits=[unpack_qubit(qubit)])


def Z(qubit: QubitDesignator) -> Gate:
    """Produces the Z gate::

        Z = [[1,  0],
             [0, -1]]

    This gate is a single qubit Z-gate.

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="Z", params=[], qubits=[unpack_qubit(qubit)])


def H(qubit: QubitDesignator) -> Gate:
    """Produces the Hadamard gate::

        H = (1 / sqrt(2)) * [[1,  1],
                             [1, -1]]

    Produces the H instruction. This gate is a single qubit Hadamard gate.

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="H", params=[], qubits=[unpack_qubit(qubit)])


def S(qubit: QubitDesignator) -> Gate:
    """Produces the S gate::

        S = [[1, 0],
             [0, 1j]]

    This gate is a single qubit S-gate.

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="S", params=[], qubits=[unpack_qubit(qubit)])


def T(qubit: QubitDesignator) -> Gate:
    """Produces the T gate::

        T = [[1, 0],
             [0, exp(1j * pi / 4)]]

    This gate is a single qubit T-gate. It is the same as RZ(pi/4).

    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="T", params=[], qubits=[unpack_qubit(qubit)])


def RX(angle: ParameterDesignator, qubit: QubitDesignator) -> Gate:
    """Produces the RX gate::

        RX(phi) = [[cos(phi / 2), -1j * sin(phi / 2)],
                   [-1j * sin(phi / 2), cos(phi / 2)]]

    This gate is a single qubit X-rotation.

    :param angle: The angle to rotate around the x-axis on the bloch sphere.
    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="RX", params=[angle], qubits=[unpack_qubit(qubit)])


def RY(angle: ParameterDesignator, qubit: QubitDesignator) -> Gate:
    """Produces the RY gate::

        RY(phi) = [[cos(phi / 2), -sin(phi / 2)],
                   [sin(phi / 2), cos(phi / 2)]]

    This gate is a single qubit Y-rotation.

    :param angle: The angle to rotate around the y-axis on the bloch sphere.
    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="RY", params=[angle], qubits=[unpack_qubit(qubit)])


def RZ(angle: ParameterDesignator, qubit: QubitDesignator) -> Gate:
    """Produces the RZ gate::

        RZ(phi) = [[cos(phi / 2) - 1j * sin(phi / 2), 0]
                   [0, cos(phi / 2) + 1j * sin(phi / 2)]]

    This gate is a single qubit Z-rotation.

    :param angle: The angle to rotate around the z-axis on the bloch sphere.
    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="RZ", params=[angle], qubits=[unpack_qubit(qubit)])


def PHASE(angle: ParameterDesignator, qubit: QubitDesignator) -> Gate:
    """Produces the PHASE gate::

        PHASE(phi) = [[1, 0],
                      [0, exp(1j * phi)]]

    This is the same as the RZ gate.

    :param angle: The angle to rotate around the z-axis on the bloch sphere.
    :param qubit: The qubit apply the gate to.
    :returns: A Gate object.
    """
    return Gate(name="PHASE", params=[angle], qubits=[unpack_qubit(qubit)])


def CZ(control: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a controlled-Z gate::

        CZ = [[1, 0, 0,  0],
              [0, 1, 0,  0],
              [0, 0, 1,  0],
              [0, 0, 0, -1]]


    This gate applies to two qubit arguments to produce the controlled-Z gate instruction.

    :param control: The control qubit.
    :param target: The target qubit. The target qubit has an Z-gate applied to it if the control
        qubit is in the excited state.
    :returns: A Gate object.
    """
    return Gate(name="CZ", params=[], qubits=[unpack_qubit(q) for q in (control, target)])


def CNOT(control: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a controlled-NOT (controlled-X) gate::

        CNOT = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]]

    This gate applies to two qubit arguments to produce the controlled-not gate instruction.

    :param control: The control qubit.
    :param target: The target qubit. The target qubit has an X-gate applied to it if the control
        qubit is in the ``|1>`` state.
    :returns: A Gate object.
    """
    return Gate(name="CNOT", params=[], qubits=[unpack_qubit(q) for q in (control, target)])


def CCNOT(control1: QubitDesignator, control2: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a doubly-controlled NOT gate::

        CCNOT = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0]]

    This gate applies to three qubit arguments to produce the controlled-controlled-not gate
    instruction.

    :param control1: The first control qubit.
    :param control2: The second control qubit.
    :param target: The target qubit. The target qubit has an X-gate applied to it if both control
        qubits are in the excited state.
    :returns: A Gate object.
    """
    qubits = [unpack_qubit(q) for q in (control1, control2, target)]
    return Gate(name="CCNOT", params=[], qubits=qubits)


def CPHASE00(angle: ParameterDesignator, control: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a controlled-phase gate that phases the ``|00>`` state::

        CPHASE00(phi) = diag([exp(1j * phi), 1, 1, 1])

    This gate applies to two qubit arguments to produce the variant of the controlled phase
    instruction that affects the state 00.

    :param angle: The input phase angle to apply when both qubits are in the ``|0>`` state.
    :param control: Qubit 1.
    :param target: Qubit 2.
    :returns: A Gate object.
    """
    qubits = [unpack_qubit(q) for q in (control, target)]
    return Gate(name="CPHASE00", params=[angle], qubits=qubits)


def CPHASE01(angle: ParameterDesignator, control: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a controlled-phase gate that phases the ``|01>`` state::

        CPHASE01(phi) = diag([1.0, exp(1j * phi), 1.0, 1.0])

    This gate applies to two qubit arguments to produce the variant of the controlled phase
    instruction that affects the state 01.

    :param angle: The input phase angle to apply when q1 is in the ``|1>`` state and q2 is in
        the ``|0>`` state.
    :param control: Qubit 1.
    :param target: Qubit 2.
    :returns: A Gate object.
    """
    qubits = [unpack_qubit(q) for q in (control, target)]
    return Gate(name="CPHASE01", params=[angle], qubits=qubits)


def CPHASE10(angle: ParameterDesignator, control: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a controlled-phase gate that phases the ``|10>`` state::

        CPHASE10(phi) = diag([1, 1, exp(1j * phi), 1])

    This gate applies to two qubit arguments to produce the variant of the controlled phase
    instruction that affects the state 10.

    :param angle: The input phase angle to apply when q2 is in the ``|1>`` state and q1 is in
        the ``|0>`` state.
    :param control: Qubit 1.
    :param target: Qubit 2.
    :returns: A Gate object.
    """
    qubits = [unpack_qubit(q) for q in (control, target)]
    return Gate(name="CPHASE10", params=[angle], qubits=qubits)


# NOTE: We don't use ParameterDesignator here because of the following Sphinx error. This error
# can be resolved by importing Expression, but then flake8 complains about an unused import:
#   Cannot resolve forward reference in type annotations of "pyquil.gates.CPHASE":
#   name 'Expression' is not defined
def CPHASE(
    angle: Union[Expression, MemoryReference, np.int_, int, float, complex],
    control: QubitDesignator,
    target: QubitDesignator,
) -> Gate:
    """Produces a controlled-phase instruction::

        CPHASE(phi) = diag([1, 1, 1, exp(1j * phi)])

    This gate applies to two qubit arguments to produce the variant of the controlled phase
    instruction that affects the state 11.

    Compare with the ``CPHASExx`` variants. This variant is the most common and does
    not have a suffix, although you can think of it as ``CPHASE11``.

    :param angle: The input phase angle to apply when both qubits are in the ``|1>`` state.
    :param control: Qubit 1.
    :param target: Qubit 2.
    :returns: A Gate object.
    """
    qubits = [unpack_qubit(q) for q in (control, target)]
    return Gate(name="CPHASE", params=[angle], qubits=qubits)


def SWAP(q1: QubitDesignator, q2: QubitDesignator) -> Gate:
    """Produces a SWAP gate which swaps the state of two qubits::

        SWAP = [[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]


    :param q1: Qubit 1.
    :param q2: Qubit 2.
    :returns: A Gate object.
    """
    return Gate(name="SWAP", params=[], qubits=[unpack_qubit(q) for q in (q1, q2)])


def CSWAP(control: QubitDesignator, target_1: QubitDesignator, target_2: QubitDesignator) -> Gate:
    """Produces a controlled-SWAP gate. This gate conditionally swaps the state of two qubits::

        CSWAP = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]


    :param control: The control qubit.
    :param target_1: The first target qubit.
    :param target_2: The second target qubit. The two target states are swapped if the control is
        in the ``|1>`` state.
    """
    qubits = [unpack_qubit(q) for q in (control, target_1, target_2)]
    return Gate(name="CSWAP", params=[], qubits=qubits)


def ISWAP(q1: QubitDesignator, q2: QubitDesignator) -> Gate:
    """Produces an ISWAP gate::

        ISWAP = [[1, 0,  0,  0],
                 [0, 0,  1j, 0],
                 [0, 1j, 0,  0],
                 [0, 0,  0,  1]]

    This gate swaps the state of two qubits, applying a -i phase to q1 when it
    is in the 1 state and a -i phase to q2 when it is in the 0 state.

    :param q1: Qubit 1.
    :param q2: Qubit 2.
    :returns: A Gate object.
    """
    return Gate(name="ISWAP", params=[], qubits=[unpack_qubit(q) for q in (q1, q2)])


def PSWAP(angle: ParameterDesignator, q1: QubitDesignator, q2: QubitDesignator) -> Gate:
    """Produces a parameterized SWAP gate::

        PSWAP(phi) = [[1, 0,             0,             0],
                      [0, 0,             exp(1j * phi), 0],
                      [0, exp(1j * phi), 0,             0],
                      [0, 0,             0,             1]]


    :param angle: The angle of the phase to apply to the swapped states. This phase is applied to
        q1 when it is in the 1 state and to q2 when it is in the 0 state.
    :param q1: Qubit 1.
    :param q2: Qubit 2.
    :returns: A Gate object.
    """
    return Gate(name="PSWAP", params=[angle], qubits=[unpack_qubit(q) for q in (q1, q2)])


def XY(angle: ParameterDesignator, q1: QubitDesignator, q2: QubitDesignator) -> Gate:
    """Produces a parameterized ISWAP gate::

        XY(phi) = [[1,               0,               0, 0],
                   [0,      cos(phi/2), 1j * sin(phi/2), 0],
                   [0, 1j * sin(phi/2),      cos(phi/2), 0],
                   [0,               0,               0, 1]

    :param angle: The angle of the rotation to apply to the population 1 subspace.
    :param q1: Qubit 1.
    :param q2: Qubit 2.
    :returns: A Gate object.
    """
    return Gate(name="XY", params=[angle], qubits=[unpack_qubit(q) for q in (q1, q2)])


WAIT = Wait()
"""
This instruction tells the quantum computation to halt. Typically these is used while classical
memory is being manipulated by a CPU in a hybrid classical/quantum algorithm.

:returns: A Wait object.
"""


def RESET(qubit_index: Optional[QubitDesignator] = None) -> Union[Reset, ResetQubit]:
    """
    Reset all qubits or just one specific qubit.

    :param qubit_index: The qubit to reset.
        This can be a qubit's index, a Qubit, or a QubitPlaceholder.
        If None, reset all qubits.
    :returns: A Reset or ResetQubit Quil AST expression corresponding to a global or targeted
        reset, respectively.
    """
    if qubit_index is not None:
        return ResetQubit(unpack_qubit(qubit_index))
    else:
        return Reset()


NOP = Nop()
"""
This instruction applies no operation at that timestep. Typically these are ignored in error-models.

:returns: A Nop object.
"""

HALT = Halt()
"""
This instruction ends the program.

:returns: A Halt object.
"""


def MEASURE(
    qubit: QubitDesignator, classical_reg: Optional[MemoryReferenceDesignator]
) -> Measurement:
    """
    Produce a MEASURE instruction.

    :param qubit: The qubit to measure.
    :param classical_reg: The classical register to measure into, or None.
    :return: A Measurement instance.
    """
    qubit = unpack_qubit(qubit)
    if classical_reg is None:
        address = None
    elif isinstance(classical_reg, int):
        warn(
            "Indexing measurement addresses by integers is deprecated. "
            "Replacing this with the MemoryReference ro[i] instead."
        )
        address = MemoryReference("ro", classical_reg)
    else:
        address = unpack_classical_reg(classical_reg)
    return Measurement(qubit, address)


def TRUE(classical_reg: Union[MemoryReference, int]) -> ClassicalMove:
    """
    Produce a TRUE instruction.

    :param classical_reg: A classical register to modify.
    :return: An instruction object representing the equivalent MOVE.
    """
    warn("`TRUE a` has been deprecated. Use `MOVE a 1` instead.")
    if isinstance(classical_reg, int):
        classical_reg = Addr(classical_reg)
    return MOVE(classical_reg, 1)


def FALSE(classical_reg: Union[MemoryReference, int]) -> ClassicalMove:
    """
    Produce a FALSE instruction.

    :param classical_reg: A classical register to modify.
    :return: An instruction object representing the equivalent MOVE.
    """
    warn("`FALSE a` has been deprecated. Use `MOVE a 0` instead.")
    if isinstance(classical_reg, int):
        classical_reg = Addr(classical_reg)
    return MOVE(classical_reg, 0)


def NEG(classical_reg: MemoryReferenceDesignator) -> ClassicalNeg:
    """
    Produce a NEG instruction.

    :param classical_reg: A classical memory address to modify.
    :return: A ClassicalNeg instance.
    """
    return ClassicalNeg(unpack_classical_reg(classical_reg))


def NOT(classical_reg: MemoryReferenceDesignator) -> ClassicalNot:
    """
    Produce a NOT instruction.

    :param classical_reg: A classical register to modify.
    :return: A ClassicalNot instance.
    """
    return ClassicalNot(unpack_classical_reg(classical_reg))


def AND(
    classical_reg1: MemoryReferenceDesignator, classical_reg2: Union[MemoryReferenceDesignator, int]
) -> ClassicalAnd:
    """
    Produce an AND instruction.

    NOTE: The order of operands was reversed in pyQuil <=1.9 .

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register or immediate value.
    :return: A ClassicalAnd instance.
    """
    left, right = unpack_reg_val_pair(classical_reg1, classical_reg2)
    assert isinstance(right, (MemoryReference, int))  # placate mypy
    return ClassicalAnd(left, right)


def OR(
    classical_reg1: MemoryReferenceDesignator, classical_reg2: MemoryReferenceDesignator
) -> ClassicalInclusiveOr:
    """
    Produce an OR instruction.

    NOTE: Deprecated. Use IOR instead.

    :param classical_reg1: The first classical register.
    :param classical_reg2: The second classical register, which gets modified.
    :return: A ClassicalOr instance.
    """
    warn("OR has been deprecated. Use IOR, inclusive OR, instead.")
    return IOR(classical_reg2, classical_reg1)


def IOR(
    classical_reg1: MemoryReferenceDesignator, classical_reg2: Union[MemoryReferenceDesignator, int]
) -> ClassicalInclusiveOr:
    """
    Produce an inclusive OR instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register or immediate value.
    :return: A ClassicalInclusiveOr instance.
    """
    left, right = unpack_reg_val_pair(classical_reg1, classical_reg2)
    assert isinstance(right, (MemoryReference, int))  # placate mypy
    return ClassicalInclusiveOr(left, right)


def XOR(
    classical_reg1: MemoryReferenceDesignator, classical_reg2: Union[MemoryReferenceDesignator, int]
) -> ClassicalExclusiveOr:
    """
    Produce an exclusive OR instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register or immediate value.
    :return: A ClassicalExclusiveOr instance.
    """
    left, right = unpack_reg_val_pair(classical_reg1, classical_reg2)
    assert isinstance(right, (MemoryReference, int))  # placate mypy
    return ClassicalExclusiveOr(left, right)


def MOVE(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalMove:
    """
    Produce a MOVE instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register or immediate value.
    :return: A ClassicalMove instance.
    """
    left, right = unpack_reg_val_pair(classical_reg1, classical_reg2)
    return ClassicalMove(left, right)


def EXCHANGE(
    classical_reg1: MemoryReferenceDesignator, classical_reg2: MemoryReferenceDesignator
) -> ClassicalExchange:
    """
    Produce an EXCHANGE instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register, which gets modified.
    :return: A ClassicalExchange instance.
    """
    left = unpack_classical_reg(classical_reg1)
    right = unpack_classical_reg(classical_reg2)
    return ClassicalExchange(left, right)


def LOAD(
    target_reg: MemoryReferenceDesignator, region_name: str, offset_reg: MemoryReferenceDesignator
) -> ClassicalLoad:
    """
    Produce a LOAD instruction.

    :param target_reg: LOAD storage target.
    :param region_name: Named region of memory to load from.
    :param offset_reg: Offset into region of memory to load from. Must be a MemoryReference.
    :return: A ClassicalLoad instance.
    """
    return ClassicalLoad(
        unpack_classical_reg(target_reg), region_name, unpack_classical_reg(offset_reg)
    )


def STORE(
    region_name: str,
    offset_reg: MemoryReferenceDesignator,
    source: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalStore:
    """
    Produce a STORE instruction.

    :param region_name: Named region of memory to store to.
    :param offset_reg: Offset into memory region. Must be a MemoryReference.
    :param source: Source data. Can be either a MemoryReference or a constant.
    :return: A ClassicalStore instance.
    """
    if not isinstance(source, int) and not isinstance(source, float):
        source = unpack_classical_reg(source)
    return ClassicalStore(region_name, unpack_classical_reg(offset_reg), source)


def CONVERT(
    classical_reg1: MemoryReferenceDesignator, classical_reg2: MemoryReferenceDesignator
) -> ClassicalConvert:
    """
    Produce a CONVERT instruction.

    :param classical_reg1: MemoryReference to store to.
    :param classical_reg2: MemoryReference to read from.
    :return: A ClassicalConvert instance.
    """
    return ClassicalConvert(
        unpack_classical_reg(classical_reg1), unpack_classical_reg(classical_reg2)
    )


def ADD(
    classical_reg: MemoryReferenceDesignator, right: Union[MemoryReferenceDesignator, int, float]
) -> ClassicalAdd:
    """
    Produce an ADD instruction.

    :param classical_reg: Left operand for the arithmetic operation. Also serves as the store
        target.
    :param right: Right operand for the arithmetic operation.
    :return: A ClassicalAdd instance.
    """
    left, right = unpack_reg_val_pair(classical_reg, right)
    return ClassicalAdd(left, right)


def SUB(
    classical_reg: MemoryReferenceDesignator, right: Union[MemoryReferenceDesignator, int, float]
) -> ClassicalSub:
    """
    Produce a SUB instruction.

    :param classical_reg: Left operand for the arithmetic operation. Also serves as the store
        target.
    :param right: Right operand for the arithmetic operation.
    :return: A ClassicalSub instance.
    """
    left, right = unpack_reg_val_pair(classical_reg, right)
    return ClassicalSub(left, right)


def MUL(
    classical_reg: MemoryReferenceDesignator, right: Union[MemoryReferenceDesignator, int, float]
) -> ClassicalMul:
    """
    Produce a MUL instruction.

    :param classical_reg: Left operand for the arithmetic operation. Also serves as the store
        target.
    :param right: Right operand for the arithmetic operation.
    :return: A ClassicalMul instance.
    """
    left, right = unpack_reg_val_pair(classical_reg, right)
    return ClassicalMul(left, right)


def DIV(
    classical_reg: MemoryReferenceDesignator, right: Union[MemoryReferenceDesignator, int, float]
) -> ClassicalDiv:
    """
    Produce an DIV instruction.

    :param classical_reg: Left operand for the arithmetic operation. Also serves as the store
        target.
    :param right: Right operand for the arithmetic operation.
    :return: A ClassicalDiv instance.
    """
    left, right = unpack_reg_val_pair(classical_reg, right)
    return ClassicalDiv(left, right)


def EQ(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: MemoryReferenceDesignator,
    classical_reg3: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalEqual:
    """
    Produce an EQ instruction.

    :param classical_reg1: Memory address to which to store the comparison result.
    :param classical_reg2: Left comparison operand.
    :param classical_reg3: Right comparison operand.
    :return: A ClassicalEqual instance.
    """
    classical_reg1, classical_reg2, classical_reg3 = prepare_ternary_operands(
        classical_reg1, classical_reg2, classical_reg3
    )

    return ClassicalEqual(classical_reg1, classical_reg2, classical_reg3)


def LT(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: MemoryReferenceDesignator,
    classical_reg3: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalLessThan:
    """
    Produce an LT instruction.

    :param classical_reg1: Memory address to which to store the comparison result.
    :param classical_reg2: Left comparison operand.
    :param classical_reg3: Right comparison operand.
    :return: A ClassicalLessThan instance.
    """
    classical_reg1, classical_reg2, classical_reg3 = prepare_ternary_operands(
        classical_reg1, classical_reg2, classical_reg3
    )
    return ClassicalLessThan(classical_reg1, classical_reg2, classical_reg3)


def LE(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: MemoryReferenceDesignator,
    classical_reg3: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalLessEqual:
    """
    Produce an LE instruction.

    :param classical_reg1: Memory address to which to store the comparison result.
    :param classical_reg2: Left comparison operand.
    :param classical_reg3: Right comparison operand.
    :return: A ClassicalLessEqual instance.
    """
    classical_reg1, classical_reg2, classical_reg3 = prepare_ternary_operands(
        classical_reg1, classical_reg2, classical_reg3
    )
    return ClassicalLessEqual(classical_reg1, classical_reg2, classical_reg3)


def GT(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: MemoryReferenceDesignator,
    classical_reg3: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalGreaterThan:
    """
    Produce an GT instruction.

    :param classical_reg1: Memory address to which to store the comparison result.
    :param classical_reg2: Left comparison operand.
    :param classical_reg3: Right comparison operand.
    :return: A ClassicalGreaterThan instance.
    """
    classical_reg1, classical_reg2, classical_reg3 = prepare_ternary_operands(
        classical_reg1, classical_reg2, classical_reg3
    )
    return ClassicalGreaterThan(classical_reg1, classical_reg2, classical_reg3)


def GE(
    classical_reg1: MemoryReferenceDesignator,
    classical_reg2: MemoryReferenceDesignator,
    classical_reg3: Union[MemoryReferenceDesignator, int, float],
) -> ClassicalGreaterEqual:
    """
    Produce an GE instruction.

    :param classical_reg1: Memory address to which to store the comparison result.
    :param classical_reg2: Left comparison operand.
    :param classical_reg3: Right comparison operand.
    :return: A ClassicalGreaterEqual instance.
    """
    classical_reg1, classical_reg2, classical_reg3 = prepare_ternary_operands(
        classical_reg1, classical_reg2, classical_reg3
    )
    return ClassicalGreaterEqual(classical_reg1, classical_reg2, classical_reg3)


QUANTUM_GATES: Mapping[str, Callable[..., Gate]] = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "PHASE": PHASE,
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "CZ": CZ,
    "CNOT": CNOT,
    "CCNOT": CCNOT,
    "CPHASE00": CPHASE00,
    "CPHASE01": CPHASE01,
    "CPHASE10": CPHASE10,
    "CPHASE": CPHASE,
    "SWAP": SWAP,
    "CSWAP": CSWAP,
    "ISWAP": ISWAP,
    "PSWAP": PSWAP,
    "XY": XY,
}
"""
Dictionary of quantum gate functions keyed by gate names.
"""

STANDARD_GATES = QUANTUM_GATES
"""
Alias for the above dictionary of quantum gates.
"""

STANDARD_INSTRUCTIONS: Mapping[
    str, Union[AbstractInstruction, Callable[..., AbstractInstruction]]
] = {
    "WAIT": WAIT,
    "RESET": RESET,
    "NOP": NOP,
    "HALT": HALT,
    "MEASURE": MEASURE,
    "TRUE": TRUE,
    "FALSE": FALSE,
    "NOT": NOT,
    "AND": AND,
    "OR": OR,
    "MOVE": MOVE,
    "EXCHANGE": EXCHANGE,
    "IOR": IOR,
    "XOR": XOR,
    "NEG": NEG,
    "ADD": ADD,
    "SUB": SUB,
    "MUL": MUL,
    "DIV": DIV,
    "EQ": EQ,
    "GT": GT,
    "GE": GE,
    "LE": LE,
    "LT": LT,
    "LOAD": LOAD,
    "STORE": STORE,
    "CONVERT": CONVERT,
}
"""
Dictionary of standard instruction functions keyed by instruction names.
"""

__all__ = (
    list(QUANTUM_GATES.keys())
    + list(STANDARD_INSTRUCTIONS.keys())
    + ["Gate", "QUANTUM_GATES", "STANDARD_GATES", "STANDARD_INSTRUCTIONS"]
)
