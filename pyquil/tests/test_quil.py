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
import re
from math import pi

import numpy as np
import pytest

from pyquil.gates import I, X, Y, Z, H, T, S, RX, RY, RZ, CNOT, CCNOT, PHASE, CPHASE00, CPHASE01, \
    CPHASE10, CPHASE, SWAP, CSWAP, ISWAP, PSWAP, MEASURE, HALT, WAIT, NOP, RESET, \
    TRUE, FALSE, NOT, AND, OR, MOVE, EXCHANGE, \
    LOAD, CONVERT, STORE, XOR, IOR, NEG, ADD, SUB, MUL, DIV, EQ, GT, GE, LT, LE
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.paulis import exponential_map, sZ
from pyquil.quil import Program, merge_programs, merge_with_pauli_noise, address_qubits, \
    get_classical_addresses_from_program, Pragma, validate_protoquil
from pyquil.quilatom import QubitPlaceholder, Addr, MemoryReference
from pyquil.quilbase import DefGate, Gate, Qubit, JumpWhen, Declare
from pyquil.tests.utils import parse_equals


def test_gate():
    tg = Gate("TEST", qubits=[Qubit(1), Qubit(2)], params=[])
    assert tg.out() == "TEST 1 2"


def test_defgate():
    dg = DefGate("TEST", np.array([[1., 0.],
                                   [0., 1.]]))
    assert dg.out() == "DEFGATE TEST:\n    1.0, 0\n    0, 1.0\n"
    test = dg.get_constructor()
    tg = test(Qubit(1), Qubit(2))
    assert tg.out() == "TEST 1 2"
    tg = test(1, 2)
    assert tg.out() == "TEST 1 2"


def test_defgate_non_square_should_throw_error():
    with pytest.raises(ValueError) as error_info:
        DefGate("TEST", np.array([[0 + 0.5j, 0.5, 1],
                                  [0.5, 0 - 0.5j, 1]]))
    assert str(error_info.value) == "Matrix must be square."


def test_defgate_non_unitary_should_throw_error():
    with pytest.raises(ValueError) as error_info:
        DefGate("TEST", np.array([[0, 1],
                                  [2, 3]]))
    assert str(error_info.value) == "Matrix must be unitary."


def test_defgate_param():
    dgp = DefGate("TEST", [[1., 0.], [0., 1.]])
    assert dgp.out() == "DEFGATE TEST:\n    1.0, 0\n    0, 1.0\n"
    test = dgp.get_constructor()
    tg = test(Qubit(1))
    assert tg.out() == "TEST 1"


def test_inst_gates():
    p = Program()
    p.inst(H(0), X(1))
    assert len(p) == 2
    assert p.out() == "H 0\nX 1\n"


def test_inst_tuple():
    p = Program()
    p.inst(("Y", 0),
           ("X", 1))
    assert len(p) == 2
    assert p.out() == "Y 0\nX 1\n"


def test_inst_string():
    p = Program()
    p.inst("Y 0",
           "X 1", )
    assert len(p) == 2
    assert p.out() == "Y 0\nX 1\n"


def test_program_pop():
    prog = Program(X(0), X(1))
    instruction = prog.pop()
    assert prog.out() == "X 0\n"
    assert Program(instruction).out() == "X 1\n"


def test_len_zero():
    prog = Program()
    assert len(prog) == 0


def test_len_one():
    prog = Program(X(0))
    assert len(prog) == 1


def test_len_nested():
    p = Program(H(0)).measure(0, 0)
    q = Program(H(0), CNOT(0, 1))
    p.if_then(MemoryReference("ro", 0), q)
    assert len(p) == 9


def test_plus_operator():
    p = Program()
    p += H(0)
    p += [X(0), Y(0), Z(0)]
    assert len(p) == 4
    assert p.out() == "H 0\nX 0\nY 0\nZ 0\n"


def test_indexing():
    program = Program(Declare('ro', 'BIT'), H(0), Y(1), CNOT(0, 1)) \
        .measure(0, 0) \
        .if_then(MemoryReference("ro", 0), Program(X(0)), Program())
    assert program[1] == H(0)
    for ii, instr in enumerate(program.instructions):
        assert program[ii] == instr


def test_iteration():
    gate_list = [H(0), Y(1), CNOT(0, 1)]
    program = Program(gate_list)
    for ii, instruction in enumerate(program):
        assert instruction == gate_list[ii]

    # https://github.com/rigetti/pyquil/issues/265
    gate_generator = (gate_list[ii] for ii in range(3))
    program = Program(gate_generator)
    for ii, instruction in enumerate(program):
        assert instruction == gate_list[ii]


def test_program_plus_program():
    p = Program().inst(X(0))
    q = Program().inst(Y(0))
    r = p + q
    assert len(p.instructions) == 1
    assert len(q.instructions) == 1
    assert len(r.instructions) == 2
    assert p.out() == "X 0\n"
    assert q.out() == "Y 0\n"
    assert r.out() == "X 0\nY 0\n"


def test_program_tuple():
    p = Program()
    p.inst(("Y", 0),
           ("X", 1))
    assert len(p) == 2
    assert p.out() == "Y 0\nX 1\n"


def test_program_string():
    p = Program()
    p.inst("Y 0",
           "X 1", )
    assert len(p.instructions) == 2
    assert p.instructions == [Y(0), X(1)]
    assert p.out() == "Y 0\nX 1\n"


def test_prog_init():
    p = Program()
    p.inst(X(0)).measure(0, 0)
    assert p.out() == ('DECLARE ro BIT[1]\n'
                       'X 0\n'
                       'MEASURE 0 ro[0]\n')


def test_classical_regs():
    p = Program()
    p.inst(X(0)).measure(0, 1)
    assert p.out() == ('DECLARE ro BIT[2]\n'
                       'X 0\n'
                       'MEASURE 0 ro[1]\n')


def test_simple_instructions():
    p = Program().inst(HALT, WAIT, RESET(), NOP)
    assert p.out() == 'HALT\nWAIT\nRESET\nNOP\n'


def test_unary_classicals():
    p = Program()
    p.inst(TRUE(0),
           FALSE(Addr(1)),
           NOT(Addr(2)),
           NEG(Addr(3)))
    assert p.out() == 'MOVE ro[0] 1\n' \
                      'MOVE ro[1] 0\n' \
                      'NOT ro[2]\n' \
                      'NEG ro[3]\n'


def test_binary_classicals():
    p = Program()
    p.inst(AND(Addr(0), Addr(1)),
           OR(Addr(1), Addr(0)),
           MOVE(Addr(0), Addr(1)),
           CONVERT(Addr(0), Addr(1)),
           IOR(Addr(0), Addr(1)),
           XOR(Addr(0), Addr(1)),
           ADD(Addr(0), Addr(1)),
           SUB(Addr(0), Addr(1)),
           MUL(Addr(0), Addr(1)),
           DIV(Addr(0), Addr(1)),
           EXCHANGE(Addr(0), Addr(1)))
    assert p.out() == 'AND ro[0] ro[1]\n' \
                      'IOR ro[0] ro[1]\n' \
                      'MOVE ro[0] ro[1]\n' \
                      'CONVERT ro[0] ro[1]\n' \
                      'IOR ro[0] ro[1]\n' \
                      'XOR ro[0] ro[1]\n' \
                      'ADD ro[0] ro[1]\n' \
                      'SUB ro[0] ro[1]\n'\
                      'MUL ro[0] ro[1]\n' \
                      'DIV ro[0] ro[1]\n' \
                      'EXCHANGE ro[0] ro[1]\n'


def test_ternary_classicals():
    p = Program()
    p.inst(LOAD(MemoryReference("ro", 0), "ro", MemoryReference("n", 0)),
           STORE("ro", MemoryReference("n", 0), MemoryReference("ro", 0)),
           EQ(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 2)),
           GT(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 2)),
           GE(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 2)),
           LE(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 2)),
           LT(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 2)))
    assert p.out() == 'LOAD ro[0] ro n[0]\n' \
                      'STORE ro n[0] ro[0]\n' \
                      'EQ ro[0] ro[1] ro[2]\n' \
                      'GT ro[0] ro[1] ro[2]\n' \
                      'GE ro[0] ro[1] ro[2]\n' \
                      'LE ro[0] ro[1] ro[2]\n' \
                      'LT ro[0] ro[1] ro[2]\n'


def test_measurement_calls():
    p = Program()
    p.inst(MEASURE(0, 1),
           MEASURE(0, Addr(1)))
    assert p.out() == ('DECLARE ro BIT[2]\n'
                       'MEASURE 0 ro[1]\n'
                       'MEASURE 0 ro[1]\n')


def test_measure_all():
    p = Program()
    p.measure_all((0, 0), (1, 1), (2, 3))
    assert p.out() == 'DECLARE ro BIT[4]\n' \
                      'MEASURE 0 ro[0]\n' \
                      'MEASURE 1 ro[1]\n' \
                      'MEASURE 2 ro[3]\n'

    p = Program([H(idx) for idx in range(4)])
    p.measure_all()
    for idx in range(4):
        assert p[idx + 5] == MEASURE(idx, idx)

    p = Program()
    p.measure_all()
    assert p.out() == ''


def test_dagger():
    # these gates are their own inverses
    p = Program().inst(I(0), X(0), Y(0), Z(0),
                       H(0), CNOT(0, 1), CCNOT(0, 1, 2),
                       SWAP(0, 1), CSWAP(0, 1, 2))
    assert p.dagger().out() == 'CSWAP 0 1 2\nSWAP 0 1\n' \
                               'CCNOT 0 1 2\nCNOT 0 1\nH 0\n' \
                               'Z 0\nY 0\nX 0\nI 0\n'

    # these gates require negating a parameter
    p = Program().inst(PHASE(pi, 0), RX(pi, 0), RY(pi, 0),
                       RZ(pi, 0), CPHASE(pi, 0, 1),
                       CPHASE00(pi, 0, 1), CPHASE01(pi, 0, 1),
                       CPHASE10(pi, 0, 1), PSWAP(pi, 0, 1))
    assert p.dagger().out() == 'PSWAP(-pi) 0 1\n' \
                               'CPHASE10(-pi) 0 1\n' \
                               'CPHASE01(-pi) 0 1\n' \
                               'CPHASE00(-pi) 0 1\n' \
                               'CPHASE(-pi) 0 1\n' \
                               'RZ(-pi) 0\n' \
                               'RY(-pi) 0\n' \
                               'RX(-pi) 0\n' \
                               'PHASE(-pi) 0\n'

    # these gates are special cases
    p = Program().inst(S(0), T(0), ISWAP(0, 1))
    assert p.dagger().out() == 'PSWAP(pi/2) 0 1\n' \
                               'RZ(pi/4) 0\n' \
                               'PHASE(-pi/2) 0\n'

    # must invert defined gates
    G = np.array([[0, 1], [0 + 1j, 0]])
    p = Program().defgate("G", G).inst(("G", 0))
    assert p.dagger().out() == 'DEFGATE G-INV:\n' \
                               '    0.0, -i\n' \
                               '    1.0, 0.0\n\n' \
                               'G-INV 0\n'

    # can also pass in a list of inverses
    inv_dict = {"G": "J"}
    p = Program().defgate("G", G).inst(("G", 0))
    assert p.dagger(inv_dict=inv_dict).out() == 'J 0\n'

    # defined parameterized gates cannot auto generate daggered version https://github.com/rigetti/pyquil/issues/304
    theta = Parameter('theta')
    gparam_matrix = np.array([[quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
                             [-1j * quil_sin(theta / 2), quil_cos(theta / 2)]])
    g_param_def = DefGate('GPARAM', gparam_matrix, [theta])
    p = Program(g_param_def)
    with pytest.raises(TypeError):
        p.dagger()

    # defined parameterized gates should passback parameters https://github.com/rigetti/pyquil/issues/304
    GPARAM = g_param_def.get_constructor()
    p = Program(GPARAM(pi)(1, 2))
    assert p.dagger().out() == 'GPARAM-INV(pi) 1 2\n'


def test_construction_syntax():
    p = Program().inst(X(0), Y(1), Z(0)).measure(0, 1)
    assert p.out() == ('DECLARE ro BIT[2]\n'
                       'X 0\n'
                       'Y 1\n'
                       'Z 0\n'
                       'MEASURE 0 ro[1]\n')
    p = Program().inst(X(0)).inst(Y(1)).measure(0, 1).inst(MEASURE(1, 2))
    assert p.out() == ('DECLARE ro BIT[3]\n'
                       'X 0\n'
                       'Y 1\n'
                       'MEASURE 0 ro[1]\n'
                       'MEASURE 1 ro[2]\n')
    p = Program().inst(X(0)).measure(0, 1).inst(Y(1), X(0)).measure(0, 0)
    assert p.out() == ('DECLARE ro BIT[2]\n'
                       'X 0\n'
                       'MEASURE 0 ro[1]\n'
                       'Y 1\n'
                       'X 0\n'
                       'MEASURE 0 ro[0]\n')


def test_singles():
    p = Program(I(0), X(0), Y(1), Z(1), H(2), T(2), S(1))
    assert p.out() == 'I 0\nX 0\nY 1\nZ 1\nH 2\nT 2\nS 1\n'


def test_rotations():
    p = Program(RX(0.5, 0), RY(0.1, 1), RZ(1.4, 2))
    assert p.out() == 'RX(0.5) 0\nRY(0.1) 1\nRZ(1.4) 2\n'


def test_controlled_gates():
    p = Program(CNOT(0, 1), CCNOT(0, 1, 2))
    assert p.out() == 'CNOT 0 1\nCCNOT 0 1 2\n'


def test_phases():
    p = Program(PHASE(np.pi, 1), CPHASE00(np.pi, 0, 1), CPHASE01(np.pi, 0, 1),
                CPHASE10(np.pi, 0, 1),
                CPHASE(np.pi, 0, 1))
    assert p.out() == 'PHASE(pi) 1\nCPHASE00(pi) 0 1\n' \
                      'CPHASE01(pi) 0 1\nCPHASE10(pi) 0 1\n' \
                      'CPHASE(pi) 0 1\n'


def test_swaps():
    p = Program(SWAP(0, 1), CSWAP(0, 1, 2), ISWAP(0, 1), PSWAP(np.pi, 0, 1))
    assert p.out() == 'SWAP 0 1\nCSWAP 0 1 2\nISWAP 0 1\nPSWAP(pi) 0 1\n'


def test_def_gate():
    # First we define the new gate from a matrix
    sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                       [0.5 - 0.5j, 0.5 + 0.5j]])
    p = Program().defgate("SQRT-X", sqrt_x)

    # Then we can use the new gate
    p.inst(("SQRT-X", 0))
    assert p.out() == 'DEFGATE SQRT-X:\n    0.5+0.5i, 0.5-0.5i\n    0.5-0.5i, 0.5+0.5i\n\nSQRT-X 0\n'


def test_def_gate_with_parameters():
    theta = Parameter('theta')
    rx = np.array([[quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
                   [-1j * quil_sin(theta / 2), quil_cos(theta / 2)]])

    p = Program().defgate("RX", rx, [theta])
    assert p.out() == 'DEFGATE RX(%theta):\n' \
                      '    cos(%theta/2), -i*sin(%theta/2)\n' \
                      '    -i*sin(%theta/2), cos(%theta/2)\n\n'

    dg = DefGate('MY_RX', rx, [theta])
    MY_RX = dg.get_constructor()
    p = Program().inst(MY_RX(np.pi)(0))
    assert p.out() == 'MY_RX(pi) 0\n'


def test_multiqubit_gate():
    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                       [0.5 - 0.5j, 0.5 + 0.5j]])
    x_sqrt_x = np.kron(sqrt_x, x_gate_matrix)
    p = Program().defgate("X-SQRT-X", x_sqrt_x)

    # Then we can use the new gate
    p.inst(("X-SQRT-X", 0, 1))

    assert p.out() == 'DEFGATE X-SQRT-X:\n    0.0, 0.5+0.5i, 0.0, 0.5-0.5i\n    ' \
                      '0.5+0.5i, 0.0, 0.5-0.5i, 0.0\n    ' \
                      '0.0, 0.5-0.5i, 0.0, 0.5+0.5i\n    ' \
                      '0.5-0.5i, 0.0, 0.5+0.5i, 0.0\n\nX-SQRT-X 0 1\n'


def test_define_qft():
    def qft3(q0, q1, q2):
        p = Program()
        p.inst(H(q2),
               CPHASE(pi / 2.0, q1, q2),
               H(1),
               CPHASE(pi / 4.0, q0, q2),
               CPHASE(pi / 2.0, q0, q1),
               H(q0),
               SWAP(q0, q2))
        return p

    # I(2) is to force 3 qubits in state prep program.
    state_prep = Program().inst(X(0))

    prog = state_prep + qft3(0, 1, 2)
    output = prog.out()
    assert output == 'X 0\nH 2\nCPHASE(pi/2) 1 2\nH 1\nCPHASE(pi/4) 0 ' \
                     '2\nCPHASE(pi/2) 0 1\nH 0\nSWAP 0 2\n'


def test_control_flows():
    outer_loop = Program()
    classical_flag_register = outer_loop.declare('classical_flag_register', 'BIT')
    outer_loop += MOVE(classical_flag_register, 1)  # initialize

    inner_loop = Program()
    inner_loop += Program(X(0), H(0))
    inner_loop += MEASURE(0, classical_flag_register)

    # run inner_loop in a loop until classical_flag_register is 0
    outer_loop.while_do(classical_flag_register, inner_loop)
    assert outer_loop.out() == '\n'.join([
        "DECLARE classical_flag_register BIT[1]",
        "MOVE classical_flag_register 1",
        "LABEL @START1",
        "JUMP-UNLESS @END2 classical_flag_register",
        "X 0",
        "H 0",
        "MEASURE 0 classical_flag_register",
        "JUMP @START1",
        "LABEL @END2",
        ""
    ])


def test_control_flows_2():
    # create a program that branches based on the value of a classical register
    x_prog = Program(X(0))
    z_prog = Program()
    branch = Program(H(1)).measure(1, 1) \
        .if_then(MemoryReference("ro", 1), x_prog, z_prog) \
        .measure(0, 0)
    assert branch.out() == ('DECLARE ro BIT[2]\n'
                            'H 1\n'
                            'MEASURE 1 ro[1]\n'
                            'JUMP-WHEN @THEN1 ro[1]\n'
                            'JUMP @END2\n'
                            'LABEL @THEN1\n'
                            'X 0\n'
                            'LABEL @END2\n'
                            'MEASURE 0 ro[0]\n')


def test_if_option():
    p = Program(X(0)).measure(0, 0).if_then(MemoryReference("ro", 0), Program(X(1)))
    assert p.out() == ('DECLARE ro BIT[1]\n'
                       'X 0\n'
                       'MEASURE 0 ro[0]\n'
                       'JUMP-WHEN @THEN1 ro[0]\n'
                       'JUMP @END2\n'
                       'LABEL @THEN1\n'
                       'X 1\n'
                       'LABEL @END2\n')

    assert isinstance(p.instructions[3], JumpWhen)


def test_alloc():
    p = Program()

    p.inst(H(0))  # H 0

    q1 = p.alloc()  # q1 = 1
    q2 = p.alloc()  # q2 = 3

    p.inst(CNOT(q1, q2))  # CNOT 1 3

    p.inst(H(2))

    q3 = p.alloc()  # q3 = 4

    p.inst(X(q3))  # X 4

    with pytest.raises(RuntimeError) as e:
        _ = p.out()
    assert e.match(r'Qubit q\d+ has not been assigned an index')


def test_alloc_2():
    p = Program()

    p.inst(H(0))  # H 0

    q1 = p.alloc()  # q1 = 1
    q2 = p.alloc()  # q2 = 3

    p.inst(CNOT(q1, q2))  # CNOT 1 3

    p.inst(H(2))

    q3 = p.alloc()  # q3 = 4

    p.inst(X(q3))  # X 4
    with pytest.raises(ValueError) as e:
        _ = address_qubits(p, {
            q1: 1,
            q2: 3,
            q3: 4,
        })

    assert e.match('Your program mixes instantiated qubits with placeholders')


def test_alloc_new():
    p = Program()

    q0 = QubitPlaceholder()
    p.inst(H(q0))  # H 0

    q1 = QubitPlaceholder()
    q2 = QubitPlaceholder()

    p.inst(CNOT(q1, q2))  # CNOT 1 3

    qxxx = QubitPlaceholder()
    p.inst(H(qxxx))

    q3 = QubitPlaceholder()

    p.inst(X(q3))  # X 4
    p = address_qubits(p, {
        q1: 1,
        q2: 3,
        q3: 4,
        q0: 0,
        qxxx: 2,
    })

    assert p.out() == "H 0\n" \
                      "CNOT 1 3\n" \
                      "H 2\n" \
                      "X 4\n"


def test_multiaddress():
    p = Program()
    q0, q1 = [QubitPlaceholder() for _ in range(2)]
    p += exponential_map(sZ(q0) * sZ(q1))(0.5)

    map1 = {q0: 0, q1: 1}
    map2 = {q0: 9, q1: 10}

    p1 = address_qubits(p, map1)

    with pytest.raises(RuntimeError):
        _ = p.out()  # make sure the original isn't affected

    assert p1.out() == "CNOT 0 1\n" \
                       "RZ(1.0) 1\n" \
                       "CNOT 0 1\n"

    p2 = address_qubits(p, map2)
    assert p1.out() == "CNOT 0 1\n" \
                       "RZ(1.0) 1\n" \
                       "CNOT 0 1\n"
    assert p2.out() == "CNOT 9 10\n" \
                       "RZ(1.0) 10\n" \
                       "CNOT 9 10\n"


def test_multiple_instantiate():
    p = Program()
    q = p.alloc()
    p.inst(H(q))
    p = address_qubits(p)
    assert p.out() == 'H 0\n'
    assert p.out() == 'H 0\n'


def test_reuse_alloc():
    p = Program()
    q1 = p.alloc()
    q2 = p.alloc()
    p.inst(H(q1))
    p.inst(H(q2))
    p.inst(CNOT(q1, q2))
    p = address_qubits(p)
    assert p.out() == 'H 0\nH 1\nCNOT 0 1\n'


def test_prog_merge():
    prog_0 = Program(X(0))
    prog_1 = Program(Y(0))
    assert merge_programs([prog_0, prog_1]).out() == (prog_0 + prog_1).out()
    prog_0.defgate("test", np.eye(2))
    prog_0.inst(("test", 0))
    prog_1.defgate("test", np.eye(2))
    prog_1.inst(("test", 0))
    assert merge_programs([prog_0, prog_1]).out() == """DEFGATE test:
    1.0, 0
    0, 1.0

X 0
test 0
Y 0
test 0
"""


def test_merge_with_pauli_noise():
    p = Program(X(0)).inst(Z(0))
    probs = [0., 1., 0., 0.]
    merged = merge_with_pauli_noise(p, probs, [0])
    assert merged.out() == """DEFGATE pauli_noise:
    1.0, 0
    0, 1.0

PRAGMA ADD-KRAUS pauli_noise 0 "(0.0 0.0 0.0 0.0)"
PRAGMA ADD-KRAUS pauli_noise 0 "(0.0 1.0 1.0 0.0)"
PRAGMA ADD-KRAUS pauli_noise 0 "(0.0 0.0 0.0 0.0)"
PRAGMA ADD-KRAUS pauli_noise 0 "(0.0 0.0 0.0 -0.0)"
X 0
pauli_noise 0
Z 0
pauli_noise 0
"""


def test_get_qubits():
    pq = Program(X(0), CNOT(0, 4), MEASURE(5, 5))
    assert pq.get_qubits() == {0, 4, 5}

    q = [QubitPlaceholder() for _ in range(6)]
    pq = Program(X(q[0]), CNOT(q[0], q[4]), MEASURE(q[5], 5))
    qq = pq.alloc()
    pq.inst(Y(q[2]), X(qq))
    assert address_qubits(pq).get_qubits() == {0, 1, 2, 3, 4}

    qubit_index = 1
    p = Program(("H", qubit_index))
    assert p.get_qubits() == {qubit_index}
    q1 = p.alloc()
    q2 = p.alloc()
    p.inst(("CNOT", q1, q2))
    with pytest.raises(ValueError) as e:
        _ = address_qubits(p).get_qubits()
    assert e.match('Your program mixes instantiated qubits with placeholders')


def test_get_qubit_placeholders():
    qs = QubitPlaceholder.register(8)
    pq = Program(X(qs[0]), CNOT(qs[0], qs[4]), MEASURE(qs[5], 5))
    assert pq.get_qubits() == {qs[i] for i in [0, 4, 5]}


def test_get_qubits_not_as_indices():
    pq = Program(X(0), CNOT(0, 4), MEASURE(5, 5))
    assert pq.get_qubits(indices=False) == {Qubit(i) for i in [0, 4, 5]}


def test_eq():
    p1 = Program()
    q1 = p1.alloc()
    q2 = p1.alloc()
    p1.inst([H(q1), CNOT(q1, q2)])
    p1 = address_qubits(p1)

    p2 = Program()
    p2.inst([H(0), CNOT(0, 1)])

    assert p1 == p2
    assert not p1 != p2


def test_kraus():
    pq = Program(X(0))
    pq.define_noisy_gate("X", (0,), [
        [[0., 1.],
         [1., 0.]],
        [[0., 0.],
         [0., 0.]]
    ])
    pq.inst(X(1))
    pq.define_noisy_gate("X", (1,), [
        [[0., 1.],
         [1., 0.]],
        [[0., 0.],
         [0., 0.]]
    ])

    ret = pq.out()
    assert ret == """X 0
PRAGMA ADD-KRAUS X 0 "(0.0 1.0 1.0 0.0)"
PRAGMA ADD-KRAUS X 0 "(0.0 0.0 0.0 0.0)"
X 1
PRAGMA ADD-KRAUS X 1 "(0.0 1.0 1.0 0.0)"
PRAGMA ADD-KRAUS X 1 "(0.0 0.0 0.0 0.0)"
"""
    # test error due to bad normalization
    with pytest.raises(ValueError):
        pq.define_noisy_gate("X", (0,), [
            [[0., 1.],
             [1., 0.]],
            [[0., 1.],
             [1., 0.]]
        ])
    # test error due to bad shape of kraus op
    with pytest.raises(ValueError):
        pq.define_noisy_gate("X", (0,), [
            [[0., 1., 0.],
             [1., 0., 0.]],
            [[0., 1.],
             [1., 0.]]
        ])

    pq1 = Program(X(0))
    pq1.define_noisy_gate("X", (0,), [
        [[0., 1.],
         [1., 0.]],
        [[0., 0.],
         [0., 0.]]
    ])
    pq2 = Program(X(1))
    pq2.define_noisy_gate("X", (1,), [
        [[0., 1.],
         [1., 0.]],
        [[0., 0.],
         [0., 0.]]
    ])

    assert pq1 + pq2 == pq

    pq_nn = Program(X(0))
    pq_nn.no_noise()
    pq_nn.inst(X(1))

    assert pq_nn.out() == """X 0
PRAGMA NO-NOISE
X 1
"""


def test_define_noisy_readout():
    pq = Program(X(0))
    pq.define_noisy_readout(0, .8, .9)

    pq.inst(X(1))
    pq.define_noisy_readout(1, .9, .8)

    ret = pq.out()
    assert ret == """X 0
PRAGMA READOUT-POVM 0 "(0.8 0.09999999999999998 0.19999999999999996 0.9)"
X 1
PRAGMA READOUT-POVM 1 "(0.9 0.19999999999999996 0.09999999999999998 0.8)"
"""
    # test error due to bad normalization
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, 1.1, .5)
    # test error due to bad normalization
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, .5, 1.5)
    # test error due to negative probability
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, -0.1, .5)
    # test error due to negative probability
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, .5, -1.)
    # test error due to bad qubit_index value
    with pytest.raises(ValueError):
        pq.define_noisy_readout(-1, .5, .5)
    # test error due to bad qubit_index type
    with pytest.raises(TypeError):
        pq.define_noisy_readout(1., .5, .5)


# https://github.com/rigetti/pyquil/issues/72
def test_if_then_inherits_defined_gates():
    p1 = Program()
    p1.inst(H(0))
    p1.measure(0, 0)

    p2 = Program()
    p2.defgate("A", np.array([[1., 0.], [0., 1.]]))
    p2.inst(("A", 0))

    p3 = Program()
    p3.defgate("B", np.array([[0., 1.], [1., 0.]]))
    p3.inst(("B", 0))

    p1.if_then(MemoryReference("ro", 0), p2, p3)
    assert p2.defined_gates[0] in p1.defined_gates
    assert p3.defined_gates[0] in p1.defined_gates


# https://github.com/rigetti/pyquil/issues/124
def test_allocating_qubits_on_multiple_programs():
    p = Program()
    qubit0 = p.alloc()
    p.inst(X(qubit0))

    q = Program()
    qubit1 = q.alloc()
    q.inst(X(qubit1))

    assert address_qubits(p + q).out() == "X 0\nX 1\n"


# https://github.com/rigetti/pyquil/issues/163
def test_installing_programs_inside_other_programs():
    p = Program()
    q = Program()
    p.inst(q)
    assert len(p) == 0


# https://github.com/rigetti/pyquil/issues/168
def test_nesting_a_program_inside_itself():
    p = Program(H(0)).measure(0, 0)
    with pytest.raises(ValueError):
        p.if_then(MemoryReference("ro", 0), p)


# https://github.com/rigetti/pyquil/issues/170
def test_inline_alloc():
    p = Program()
    p += H(p.alloc())
    assert address_qubits(p).out() == "H 0\n"


# https://github.com/rigetti/pyquil/issues/138
def test_defgate_integer_input():
    dg = DefGate("TEST", np.array([[1, 0],
                                   [0, 1]]))
    assert dg.out() == "DEFGATE TEST:\n    1, 0\n    0, 1\n"


def test_out_vs_str():
    qs = QubitPlaceholder.register(6)
    pq = Program(X(qs[0]), CNOT(qs[0], qs[4]), MEASURE(qs[5], 5))

    with pytest.raises(RuntimeError) as e:
        pq.out()
    assert e.match(r'Qubit q\d+ has not been assigned an index')

    string_version = str(pq)
    should_be_re = (r'DECLARE ro BIT\[6\]\nX \{q\d+\}\nCNOT \{q\d+\} \{q\d+\}\nMEASURE \{q\d+\} ro\[5\]\n')
    assert re.fullmatch(should_be_re, string_version, flags=re.MULTILINE)


def test_get_classical_addresses_from_program():
    p = Program([H(i) for i in range(4)])
    assert get_classical_addresses_from_program(p) == {}

    p += [MEASURE(i, i) for i in [0, 3, 1]]
    assert get_classical_addresses_from_program(p) == {"ro": [0, 1, 3]}


def test_get_classical_addresses_from_quil_program():
    """
    Tests that can get_classical_addresses_from_program can handle both MEASURE
    quil instructions with and without explicit classical registers.
    """

    p = Program('\n'.join([
        'X 0',
        'MEASURE 0'
    ]))
    assert get_classical_addresses_from_program(p) == {}

    p = Program('\n'.join([
        'X 0',
        'MEASURE 0 ro[1]'
    ]))
    assert get_classical_addresses_from_program(p) == {"ro": [1]}


def test_pragma_with_placeholders():
    q = QubitPlaceholder()
    q2 = QubitPlaceholder()
    p = Program()
    p.inst(Pragma('FENCE', [q, q2]))
    address_map = {q: 0, q2: 1}
    addressed_pragma = address_qubits(p, address_map)[0]
    parse_equals('PRAGMA FENCE 0 1\n', addressed_pragma)

    pq = Program(X(q))
    pq.define_noisy_readout(q, .8, .9)

    pq.inst(X(q2))
    pq.define_noisy_readout(q2, .9, .8)

    ret = address_qubits(pq, address_map).out()
    assert ret == """X 0
PRAGMA READOUT-POVM 0 "(0.8 0.09999999999999998 0.19999999999999996 0.9)"
X 1
PRAGMA READOUT-POVM 1 "(0.9 0.19999999999999996 0.09999999999999998 0.8)"
"""


def test_implicit_declare():
    program = Program(MEASURE(0, 0))
    assert program.out() == ('DECLARE ro BIT[1]\n'
                             'MEASURE 0 ro[0]\n')


def test_no_implicit_declare():
    program = Program(
        Declare("read_out", "BIT", 5),
        MEASURE(0, MemoryReference("read_out", 4)))
    assert program.out() == ('DECLARE read_out BIT[5]\n'
                             'MEASURE 0 read_out[4]\n')


def test_no_implicit_declare_2():
    program = Program(
        MEASURE(0, MemoryReference("asdf", 4)))
    assert program.out() == 'MEASURE 0 asdf[4]\n'


def test_reset():
    p = Program()
    p.reset(0)
    p.reset()
    assert p.out() == "RESET 0\nRESET\n"


def test_copy():
    prog1 = Program(
        H(0),
        CNOT(0, 1),
    )
    prog2 = prog1.copy().measure_all()
    assert prog1.out() == '\n'.join([
        'H 0',
        'CNOT 0 1',
        ''
    ])
    assert prog2.out() == '\n'.join([
        'H 0',
        'CNOT 0 1',
        'DECLARE ro BIT[2]',
        'MEASURE 0 ro[0]',
        'MEASURE 1 ro[1]',
        '',
    ])


def test_measure_all_noncontig():
    prog = Program(
        H(0),
        H(10),
    )
    prog.measure_all()
    assert prog.out() == '\n'.join([
        'H 0',
        'H 10',
        'DECLARE ro BIT[11]',
        'MEASURE 0 ro[0]',
        'MEASURE 10 ro[10]',
        '',
    ])


def test_validate_protoquil_reset_first():
    prog = Program(
        H(0),
        RESET(),
    )
    with pytest.raises(ValueError):
        validate_protoquil(prog)
    assert not prog.is_protoquil()


def test_validate_protoquil_reset_qubit():
    prog = Program(
        RESET(2),
    )
    with pytest.raises(ValueError):
        validate_protoquil(prog)
    assert not prog.is_protoquil()


def test_validate_protoquil_measure_last():
    prog = Program(
        MEASURE(0),
        H(0),
    )
    with pytest.raises(ValueError):
        validate_protoquil(prog)
    assert not prog.is_protoquil()


def test_validate_protoquil_with_pragma():
    prog = Program(
        RESET(),
        H(1),
        Pragma('DELAY'),
        MEASURE(1)
    )
    assert prog.is_protoquil()


def test_validate_protoquil_suite():
    validate_protoquil(Program("""
RESET
DECLARE ro BIT[3]
RX(-pi/4) 2
RZ(4*pi) 3
I 0
CZ 2 3
MEASURE 2 ro[2]
MEASURE 3 ro[3]
"""))

    validate_protoquil(Program("""
RESET
DECLARE ro BIT[3]
RX(-pi/4) 2
RZ(4*pi) 3
I 0
CZ 2 3
MEASURE 2 ro[2]
MEASURE 3 ro[3]
HALT
"""))
    validate_protoquil(Program("""
RESET
DECLARE ro BIT[3]
RX(-pi/4) 2
RZ(4*pi) 3
I 0
MEASURE 0
CZ 2 3
MEASURE 2 ro[2]
X 3
MEASURE 3 ro[3]
HALT
"""))

    with pytest.raises(ValueError):
        validate_protoquil(Program("""
RESET
DECLARE ro BIT[3]
RX(-pi/4) 2
RZ(4*pi) 3
RESET
I 0
CZ 2 3
MEASURE 2 ro[2]
MEASURE 3 ro[3]
"""))

    with pytest.raises(ValueError):
        validate_protoquil(Program("""
RESET
DECLARE ro BIT[3]
RX(-pi/4) 2
RZ(4*pi) 3
MEASURE 2
I 0
CZ 2 3
MEASURE 2 ro[2]
MEASURE 3 ro[3]
"""))

    with pytest.raises(ValueError):
        validate_protoquil(Program("""
RESET
DECLARE ro BIT[3]
RX(-pi/4) 2
RZ(4*pi) 3
HALT
I 0
CZ 2 3
MEASURE 2 ro[2]
MEASURE 3 ro[3]
"""))


def test_validate_protoquil_multiple_measures():
    prog = Program(
        RESET(),
        H(1),
        Pragma('DELAY'),
        MEASURE(1),
        MEASURE(1)
    )
    with pytest.raises(ValueError):
        validate_protoquil(prog)
