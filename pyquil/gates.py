#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
"""
A lovely bunch of gates and instructions for programming with.  This module is used to provide
Pythonic sugar for Quil instructions.
"""

from .quilbase import (Measurement, Gate, Addr, Wait, Reset, Halt, Nop, ClassicalTrue,
                       ClassicalFalse, ClassicalNot, ClassicalAnd, ClassicalOr, ClassicalMove,
                       ClassicalExchange, DirectQubit, AbstractQubit, issubinstance)
from six import integer_types

def unpack_classical_reg(c):
    """
    Get the address for a classical register.

    :param c: A list of length 1 or an int or an Addr.
    :return: The address as an Addr.
    """
    if not (isinstance(c, integer_types) or isinstance(c,(list, Addr))):
        raise TypeError("c should be an int or list or Addr")
    if isinstance(c, list) and (len(c) != 1 or not isinstance(c[0], int)):
        raise ValueError("if c is a list, it should be of 1 int")
    if isinstance(c, Addr):
        return c
    elif isinstance(c, list):
        return Addr(c[0])
    else:
        return Addr(c)


def unpack_qubit(qubit):
    """
    Get a qubit from an object.

    :param qubit: An int or AbstractQubit.
    :return: An AbstractQubit instance
    """
    if isinstance(qubit, integer_types):
        return DirectQubit(qubit)
    elif not isinstance(qubit, AbstractQubit):
        raise TypeError("qubit should be an int or AbstractQubit instance")
    else:
        return qubit


def _make_gate(name, num_qubits, num_params=0):
    def gate_function(*params):
        params = list(params)
        stray_qubits = []
        if len(params) < num_params:
            raise ValueError(
                "Wrong number of params for {}. {} given, require {}."
                    .format(name, len(params), num_params)
            )
        elif len(params) > num_params:
            stray_qubits = params[num_params:]
            params = params[0:num_params]

        def ctor(*qubits):
            qubits = stray_qubits + list(qubits)
            if len(qubits) != num_qubits:
                raise ValueError(
                    "Wrong number of qubits for {}. {} given, require {}."
                        .format(name, len(qubits), num_qubits)
                )
            return Gate(name, params, [unpack_qubit(q) for q in qubits])

        if len(stray_qubits) == num_qubits:
            return Gate(name, params, [unpack_qubit(q) for q in stray_qubits])
        else:
            return ctor

    return gate_function


I = _make_gate("I", 1)()
"""
Produces the I instruction. This gate is a single qubit identity gate.
Note that this gate is different that the NOP instruction as noise channels
are typically still applied during the duration of identity gates. Identities will
also block parallelization like any other gate.

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
X = _make_gate("X", 1)()
"""
Produces the X instruction. This gate is a single qubit X-gate.

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
Y = _make_gate("Y", 1)()
"""
Produces the Y instruction. This gate is a single qubit Y-gate.

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
Z = _make_gate("Z", 1)()
"""
Produces the Z instruction. This gate is a single qubit Z-gate.

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
H = _make_gate("H", 1)()
"""
Produces the H instruction. This gate is a single qubit Hadamard gate.

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
S = _make_gate("S", 1)()
"""
Produces the S instruction. This gate is a single qubit S-gate.

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
T = _make_gate("T", 1)()
"""
Produces the T instruction. This gate is a single qubit T-gate. It is the same
as RZ(pi/4).

:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""

RX = _make_gate("RX", 1, 1)
"""
Produces the RX instruction. This gate is a single qubit X-rotation.

:param angle: The angle to rotate around the x-axis on the bloch sphere.
:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
RY = _make_gate("RY", 1, 1)
"""
Produces the RY instruction. This gate is a single qubit Y-rotation.

:param angle: The angle to rotate around the y-axis on the bloch sphere.
:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
RZ = _make_gate("RZ", 1, 1)
"""
Produces the RZ instruction. This gate is a single qubit Z-rotation.

:param angle: The angle to rotate around the z-axis on the bloch sphere.
:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""
PHASE = _make_gate("PHASE", 1, 1)
"""
Produces a PHASE instruction. This is the same as the RZ gate.

:param angle: The angle to rotate around the z-axis on the bloch sphere.
:param qubit: The qubit apply the gate to.
:returns: A Gate object.
"""

CNOT = _make_gate("CNOT", 2)()
"""
Produces a CNOT instruction.
This gate applies to two qubit arguments to produce the controlled-not gate instruction.

:param control: The control qubit.
:param target: The target qubit. The target qubit has an X-gate applied to it if the control qubit is in
               the excited state.
:returns: A Gate object.
"""
CCNOT = _make_gate("CCNOT", 3)()
"""
Produces a CCNOT instruction.
This gate applies to three qubit arguments to produce the controlled-controlled-not gate instruction.

:param control-1: The first control qubit.
:param control-2: The second control qubit.
:param target: The target qubit. The target qubit has an X-gate applied to it if both control qubits are in
               the excited state.
:returns: A Gate object.
"""

CPHASE00 = _make_gate("CPHASE00", 2, 1)
"""
Produces a CPHASE00 instruction.
This gate applies to two qubit arguments to produce one of the controlled phase instructions.

:param angle: The input phase angle to apply when both qubits are in the ground state.
:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""
CPHASE01 = _make_gate("CPHASE01", 2, 1)
"""
Produces a CPHASE01 instruction.
This gate applies to two qubit arguments to produce one of the controlled phase instructions.

:param angle: The input phase angle to apply when q1 is in the excited state and q2 is in the ground state.
:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""
CPHASE10 = _make_gate("CPHASE10", 2, 1)
"""
Produces a CPHASE10 instruction.
This gate applies to two qubit arguments to produce one of the controlled phase instructions.

:param angle: The input phase angle to apply when q2 is in the excited state and q1 is in the ground state.
:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""
CPHASE = _make_gate("CPHASE", 2, 1)
"""
Produces a CPHASE00 instruction.
This gate applies to two qubit arguments to produce one of the controlled phase instructions.

:param angle: The input phase angle to apply when both qubits are in the excited state.
:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""

SWAP = _make_gate("SWAP", 2)()
"""
Produces a SWAP instruction. This gate swaps the state of two qubits.

:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""
CSWAP = _make_gate("CSWAP", 3)()
"""
Produces a SWAP instruction. This gate swaps the state of two qubits.

:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""
ISWAP = _make_gate("ISWAP", 2)()
"""
Produces an ISWAP instruction. This gate swaps the state of two qubits, applying a -i phase to q1 when it
is in the excited state and a -i phase to q2 when it is in the ground state.

:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""
PSWAP = _make_gate("PSWAP", 2, 1)
"""
Produces a PSWAP instruction. This is a parameterized swap gate.

:param angle: The angle of the phase to apply to the swapped states. This phase is applied to q1 when it is in
              the excited state and to q2 when it is in the ground state.
:param q1: Qubit 1.
:param q2: Qubit 2.
:returns: A Gate object.
"""

WAIT = Wait()
"""
This instruction tells the quantum computation to halt. Typically these is used while classical memory is being manipulated by a CPU in a hybrid classical/quantum algorithm.

:returns: A Wait object.
"""
RESET = Reset()
"""
This instruction resets all the qubits to the ground state.

:returns: A Reset object.
"""
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


def MEASURE(qubit, classical_reg=None):
    """
    Produce a MEASURE instruction.

    :param qubit: The qubit to measure.
    :param classical_reg: The classical register to measure into, or None.
    :return: A Measurement instance.
    """
    qubit = unpack_qubit(qubit)
    address = None if classical_reg is None else unpack_classical_reg(classical_reg)
    return Measurement(qubit, address)


def TRUE(classical_reg):
    """
    Produce a TRUE instruction.

    :param classical_reg: A classical register to modify.
    :return: A ClassicalTrue instance.
    """
    return ClassicalTrue(unpack_classical_reg(classical_reg))


def FALSE(classical_reg):
    """
    Produce a FALSE instruction.

    :param classical_reg: A classical register to modify.
    :return: A ClassicalFalse instance.
    """
    return ClassicalFalse(unpack_classical_reg(classical_reg))


def NOT(classical_reg):
    """
    Produce a NOT instruction.

    :param classical_reg: A classical register to modify.
    :return: A ClassicalNot instance.
    """
    return ClassicalNot(unpack_classical_reg(classical_reg))


def AND(classical_reg1, classical_reg2):
    """
    Produce an AND instruction.

    :param classical_reg1: The first classical register.
    :param classical_reg2: The second classical register, which gets modified.
    :return: A ClassicalAnd instance.
    """
    left = unpack_classical_reg(classical_reg1)
    right = unpack_classical_reg(classical_reg2)
    return ClassicalAnd(left, right)


def OR(classical_reg1, classical_reg2):
    """
    Produce an OR instruction.

    :param classical_reg1: The first classical register.
    :param classical_reg2: The second classical register, which gets modified.
    :return: A ClassicalOr instance.
    """
    left = unpack_classical_reg(classical_reg1)
    right = unpack_classical_reg(classical_reg2)
    return ClassicalOr(left, right)


def MOVE(classical_reg1, classical_reg2):
    """
    Produce a MOVE instruction.

    :param classical_reg1: The first classical register.
    :param classical_reg2: The second classical register, which gets modified.
    :return: A ClassicalMove instance.
    """
    left = unpack_classical_reg(classical_reg1)
    right = unpack_classical_reg(classical_reg2)
    return ClassicalMove(left, right)


def EXCHANGE(classical_reg1, classical_reg2):
    """
    Produce an EXCHANGE instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register, which gets modified.
    :return: A ClassicalExchange instance.
    """
    left = unpack_classical_reg(classical_reg1)
    right = unpack_classical_reg(classical_reg2)
    return ClassicalExchange(left, right)


STANDARD_GATES = {'I': I,
                  'X': X,
                  'Y': Y,
                  'Z': Z,
                  'H': H,
                  'S': S,
                  'T': T,
                  'PHASE': PHASE,
                  'RX': RX,
                  'RY': RY,
                  'RZ': RZ,
                  'CNOT': CNOT,
                  'CCNOT': CCNOT,
                  'CPHASE00': CPHASE00,
                  'CPHASE01': CPHASE01,
                  'CPHASE10': CPHASE10,
                  'CPHASE': CPHASE,
                  'SWAP': SWAP,
                  'CSWAP': CSWAP,
                  'ISWAP': ISWAP,
                  'PSWAP': PSWAP
                  }
"""
Dictionary of standard gates. Keys are gate names, values are gate functions.
"""
