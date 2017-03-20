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

import pyquil.forest as qvm_endpoint
from pyquil.quil import Program
from pyquil.quilbase import DirectQubit
from pyquil.gates import I, X, Y, Z, H, T, S, RX, RY, RZ, CNOT, CCNOT, PHASE, CPHASE00, CPHASE01, \
    CPHASE10, CPHASE, SWAP, CSWAP, ISWAP, PSWAP, MEASURE, HALT, WAIT, NOP, RESET, \
    TRUE, FALSE, NOT, AND, OR, MOVE, EXCHANGE
from pyquil.quilbase import InstructionGroup, DefGate, Gate, reset_label_counter, RawInstr, Addr

import pytest

import numpy as np
from math import pi, sqrt


def test_make_connection():
    qvm_endpoint.Connection()


def test_gate():
    tg = Gate("TEST", qubits=(DirectQubit(1), DirectQubit(2)), params=[])
    assert tg.out() == "TEST 1 2"


def test_defgate():
    dg = DefGate("TEST", np.array([[1., 0.],
                                   [0., 1.]]))
    assert dg.out() == "DEFGATE TEST:\n    1.0, 0.0\n    0.0, 1.0\n"
    test = dg.get_constructor()
    tg = test(DirectQubit(1), DirectQubit(2))
    assert tg.out() == "TEST 1 2"


def test_defgate_non_square_should_throw_error():
    with pytest.raises(AssertionError) as error_info:
        DefGate("TEST", np.array([[0 + 0.5j, 0.5, 1],
                                  [0.5, 0 - 0.5j, 1]]))
    assert str(error_info.value) == "Matrix must be square."


def test_defgate_non_unitary_should_throw_error():
    with pytest.raises(AssertionError) as error_info:
        DefGate("TEST", np.array([[0, 1],
                                  [2, 3]]))
    assert str(error_info.value) == "Matrix must be unitary."


def test_defgate_param():
    dgp = DefGate("TEST", [[1., 0.], [0., 1.]])
    assert dgp.out() == "DEFGATE TEST:\n    1.0, 0.0\n    0.0, 1.0\n"
    test = dgp.get_constructor()
    tg = test(DirectQubit(1))
    assert tg.out() == "TEST 1"


def test_instruction_group_gates():
    ig = InstructionGroup()
    ig.inst(H(0), X(1))
    assert len(ig.actions) == 2
    assert ig.out() == "H 0\nX 1\n"


def test_instruction_group_tuple():
    ig = InstructionGroup()
    ig.inst(("Y", 0),
            ("X", 1))
    assert len(ig.actions) == 2
    assert ig.out() == "Y 0\nX 1\n"


def test_instruction_group_string():
    ig = InstructionGroup()
    ig.inst("Y 0",
            "X 1", )
    assert len(ig.actions) == 2
    assert ig.out() == "Y 0\nX 1\n"


def test_program_gates():
    ig = Program()
    ig.inst(H(0), X(1))
    assert len(ig.actions) == 2
    assert ig.out() == "H 0\nX 1\n"


def test_program_pop():
    prog = Program(X(0), X(1))
    _, instruction = prog.pop()
    assert prog.out() == "X 0\n"
    assert Program(instruction).out() == "X 1\n"


def test_plus_operator():
    p = Program()
    p += H(0)
    p += [X(0), Y(0), Z(0)]
    assert len(p.actions) == 4
    assert p.out() == "H 0\nX 0\nY 0\nZ 0\n"


def test_program_plus_program():
    p = Program().inst(X(0))
    q = Program().inst(Y(0))
    r = p + q
    assert len(p.actions) == 1
    assert len(q.actions) == 1
    assert len(r.actions) == 2
    assert p.out() == "X 0\n"
    assert q.out() == "Y 0\n"
    assert r.out() == "X 0\nY 0\n"


def test_program_tuple():
    ig = Program()
    ig.inst(("Y", 0),
            ("X", 1))
    assert len(ig.actions) == 2
    assert ig.out() == "Y 0\nX 1\n"


def test_program_string():
    ig = Program()
    ig.inst("Y 0",
            "X 1", )
    assert len(ig.actions) == 2
    assert all(isinstance(i[1], RawInstr) for i in ig.actions)
    assert ig.out() == "Y 0\nX 1\n"


def test_prog_init():
    p = Program()
    p.inst(X(0)).measure(0, 0)
    assert p.out() == 'X 0\nMEASURE 0 [0]\n'


def test_classical_regs():
    p = Program()
    p.inst(X(0)).measure(0, 1)
    assert p.out() == 'X 0\nMEASURE 0 [1]\n'


def test_simple_instructions():
    p = Program().inst(HALT, WAIT, RESET, NOP)
    assert p.out() == 'HALT\nWAIT\nRESET\nNOP\n'


def test_unary_classicals():
    p = Program()
    p.inst(TRUE(0),
           FALSE(Addr(1)),
           NOT(2))
    assert p.out() == 'TRUE [0]\n' \
                      'FALSE [1]\n' \
                      'NOT [2]\n'


def test_binary_classicals():
    p = Program()
    p.inst(AND(0, 1),
           OR(Addr(0), Addr(1)),
           MOVE(0, 1),
           EXCHANGE(0, Addr(1)))
    assert p.out() == 'AND [0] [1]\n' \
                      'OR [0] [1]\n' \
                      'MOVE [0] [1]\n' \
                      'EXCHANGE [0] [1]\n'


def test_measurement_calls():
    p = Program()
    p.inst(MEASURE(0, 1),
           MEASURE(0, Addr(1)))
    assert p.out() == 'MEASURE 0 [1]\n' * 2


def test_construction_syntax():
    p = Program().inst(X(0), Y(1), Z(0)).measure(0, 1)
    assert p.out() == 'X 0\nY 1\nZ 0\nMEASURE 0 [1]\n'
    p = Program().inst(X(0)).inst(Y(1)).measure(0, 1).inst(MEASURE(1, 2))
    assert p.out() == 'X 0\nY 1\nMEASURE 0 [1]\nMEASURE 1 [2]\n'
    p = Program().inst(X(0)).measure(0, 1).inst(Y(1), X(0)).measure(0, 0)
    assert p.out() == 'X 0\nMEASURE 0 [1]\nY 1\nX 0\nMEASURE 0 [0]\n'


def test_singles():
    p = Program(I(0), X(0), Y(1), Z(1), H(2), T(2), S(1))
    assert p.out() == 'I 0\nX 0\nY 1\nZ 1\nH 2\nT 2\nS 1\n'


def test_rotations():
    p = Program(RX(0.5)(0), RY(0.1)(1), RZ(1.4)(2))
    assert p.out() == 'RX(0.5) 0\nRY(0.1) 1\nRZ(1.4) 2\n'


def test_controlled_gates():
    p = Program(CNOT(0, 1), CCNOT(0, 1, 2))
    assert p.out() == 'CNOT 0 1\nCCNOT 0 1 2\n'


def test_phases():
    p = Program(PHASE(np.pi)(1), CPHASE00(np.pi)(0, 1), CPHASE01(np.pi)(0, 1),
                CPHASE10(np.pi)(0, 1),
                CPHASE(np.pi)(0, 1))
    assert p.out() == 'PHASE(3.141592653589793) 1\nCPHASE00(3.141592653589793) 0 1\n' \
                      'CPHASE01(3.141592653589793) 0 1\nCPHASE10(3.141592653589793) 0 1\n' \
                      'CPHASE(3.141592653589793) 0 1\n'


def test_swaps():
    p = Program(SWAP(0, 1), CSWAP(0, 1, 2), ISWAP(0, 1), PSWAP(np.pi)(0, 1))
    assert p.out() == 'SWAP 0 1\nCSWAP 0 1 2\nISWAP 0 1\nPSWAP(3.141592653589793) 0 1\n'


def test_def_gate():
    # First we define the new gate from a matrix
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                       [ 0.5-0.5j,  0.5+0.5j]])
    p = Program().defgate("SQRT-X", sqrt_x)

    # Then we can use the new gate
    p.inst(("SQRT-X", 0))
    assert p.out() == 'DEFGATE SQRT-X:\n    0.5+0.5i, 0.5-0.5i\n    0.5-0.5i, 0.5+0.5i\n\nSQRT-X 0\n'

def test_multiqubit_gate():
    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                       [ 0.5-0.5j,  0.5+0.5j]])
    x_sqrt_x = np.kron(sqrt_x, x_gate_matrix)
    p = Program().defgate("X-SQRT-X", x_sqrt_x)

    # Then we can use the new gate
    p.inst(("X-SQRT-X", 0, 1))

    assert p.out() == 'DEFGATE X-SQRT-X:\n    0.0+0.0i, 0.5+0.5i, 0.0+0.0i, 0.5-0.5i\n    ' \
                      '0.5+0.5i, 0.0+0.0i, 0.5-0.5i, 0.0+0.0i\n    ' \
                      '0.0+0.0i, 0.5-0.5i, 0.0+0.0i, 0.5+0.5i\n    ' \
                      '0.5-0.5i, 0.0+0.0i, 0.5+0.5i, 0.0+0.0i\n\nX-SQRT-X 0 1\n'


def test_define_qft():
    def qft3(q0, q1, q2):
        p = Program()
        p.inst(H(q2),
               CPHASE(pi / 2.0)(q1, q2),
               H(1),
               CPHASE(pi / 4.0)(q0, q2),
               CPHASE(pi / 2.0)(q0, q1),
               H(q0),
               SWAP(q0, q2))
        return p

    # I(2) is to force 3 qubits in state prep program.
    state_prep = Program().inst(X(0))

    prog = state_prep + qft3(0, 1, 2)
    output = prog.out()
    print output
    assert output == 'X 0\nH 2\nCPHASE(1.5707963267948966) 1 2\nH 1\nCPHASE(0.7853981633974483) 0 ' \
                     '2\nCPHASE(1.5707963267948966) 0 1\nH 0\nSWAP 0 2\n'


def test_control_flows():
    reset_label_counter()
    classical_flag_register = 2
    p = Program(X(0), H(0)).measure(0, classical_flag_register)

    # run p in a loop until classical_flag_register is 0
    loop_prog = Program(X(0)).measure(0, classical_flag_register)
    loop_prog.while_do(classical_flag_register, p)
    assert loop_prog.out() == 'X 0\nMEASURE 0 [2]\nLABEL @START1\nJUMP-UNLESS @END2 [2]\nX ' \
                              '0\nH 0\nMEASURE 0 [2]\nJUMP @START1\nLABEL @END2\n'

    # create a program that branches based on the value of a classical register
    x_prog = Program(X(0))
    z_prog = Program()
    branch = Program(H(1)).measure(1, 1).if_then(1, x_prog, z_prog).measure(0, 0)
    assert branch.out() == 'H 1\nMEASURE 1 [1]\nJUMP-WHEN @THEN3 [1]\nJUMP @END4\nLABEL ' \
                           '@THEN3\nX 0\nLABEL @END4\nMEASURE 0 [0]\n'


def test_if_option():
    reset_label_counter()
    p = Program(X(0)).measure(0, 0).if_then(0, Program(X(1)))
    assert p.out() == 'X 0\nMEASURE 0 [0]\nJUMP-WHEN @THEN1 [0]\nJUMP @END2\n' \
                      'LABEL @THEN1\nX 1\nLABEL @END2\n'


def test_alloc_free():
    p = Program()
    print p.resource_manager.in_use
    q1 = p.alloc()
    p.inst(H(q1))
    p.free(q1)
    q2 = p.alloc()
    p.inst(H(q2))
    p.free(q2)
    assert p.resource_manager.live_qubits == []
    assert p.out() == "H 0\nH 0\n"


def test_alloc_free():
    p = Program()

    p.inst(H(0))  # H 0

    q1 = p.alloc()  # q1 = 1
    q2 = p.alloc()  # q2 = 3

    p.inst(CNOT(q1, q2))  # CNOT 1 3

    p.inst(H(2))

    q3 = p.alloc()  # q3 = 4

    p.inst(X(q3))  # X 4

    p.free(q1)  # remove 1

    q4 = p.alloc()  # q4 = 1

    p.inst(Y(q4))  # Y 1

    p.free(q2)
    p.free(q3)
    p.free(q4)

    assert p.resource_manager.live_qubits == []
    assert p.out() == "H 0\n" \
                      "CNOT 1 3\n" \
                      "H 2\n" \
                      "X 4\n" \
                      "Y 1\n"


def test_multiple_instantiate():
    p = Program()
    q = p.alloc()
    p.inst(H(q))
    p.free(q)
    assert p.out() == 'H 0\n'
    assert p.out() == 'H 0\n'


def test_alloc_no_free():
    p = Program()
    q1 = p.alloc()
    q2 = p.alloc()
    p.inst(H(q1))
    p.inst(H(q2))
    assert p.out() == 'H 0\nH 1\n'
    assert p.out() == 'H 0\nH 1\n'


def test_extract_qubits():
    p = Program(RX(0.5)(0), RY(0.1)(1), RZ(1.4)(2))
    assert p.extract_qubits() == set([0, 1, 2])
    p.if_then(0, X(4), H(5)).measure(6, 2)
    assert p.extract_qubits() == set([0, 1, 2, 4, 5, 6])
    p.while_do(0, Program(X(3)).measure(3, 0))
    assert p.extract_qubits() == set([0, 1, 2, 3, 4, 5, 6])
    new_qubit = p.alloc()
    p.inst(X(new_qubit))
    p.synthesize()
    assert p.extract_qubits() == set([0, 1, 2, 3, 4, 5, 6, new_qubit.index()])
