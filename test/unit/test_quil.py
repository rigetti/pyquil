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
from syrupy.assertion import SnapshotAssertion
import quil.instructions as quil_rs

from pyquil.gates import (
    I,
    X,
    Y,
    Z,
    H,
    T,
    S,
    RX,
    RY,
    RZ,
    CNOT,
    CCNOT,
    PHASE,
    CPHASE00,
    CPHASE01,
    CPHASE10,
    CPHASE,
    SWAP,
    CSWAP,
    ISWAP,
    PSWAP,
    MEASURE,
    HALT,
    WAIT,
    NOP,
    RESET,
    NOT,
    AND,
    MOVE,
    EXCHANGE,
    LOAD,
    CONVERT,
    STORE,
    XOR,
    IOR,
    NEG,
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    GT,
    GE,
    LT,
    LE,
)
from pyquil.paulis import exponential_map, sZ
from pyquil.quil import (
    Program,
    merge_programs,
    merge_with_pauli_noise,
    address_qubits,
    get_classical_addresses_from_program,
    Pragma,
    validate_supported_quil,
)
from pyquil.quilatom import Frame, MemoryReference, Parameter, QubitPlaceholder, Sub, quil_cos, quil_sin
from pyquil.quilbase import (
    DefGate,
    DefFrame,
    Gate,
    Qubit,
    JumpWhen,
    Declare,
    DefCalibration,
    DefMeasureCalibration,
    DefPermutationGate,
)
from test.unit.utils import parse_equals


def test_gate(snapshot):
    tg = Gate("TEST", qubits=[Qubit(1), Qubit(2)], params=[])
    assert tg.out() == snapshot


def test_defgate(snapshot):
    dg = DefGate("TEST", np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert dg.out() == snapshot
    test = dg.get_constructor()
    tg = test(Qubit(1), Qubit(2))
    assert tg.out() == snapshot
    tg = test(1, 2)
    assert tg.out() == snapshot


def test_defgate_non_square_should_throw_error():
    with pytest.raises(ValueError) as error_info:
        DefGate("TEST", np.array([[0 + 0.5j, 0.5, 1], [0.5, 0 - 0.5j, 1]]))
    assert str(error_info.value) == "Matrix must be square."


def test_defgate_non_unitary_should_throw_error():
    with pytest.raises(ValueError) as error_info:
        DefGate("TEST", np.array([[0, 1], [2, 3]]))
    assert str(error_info.value) == "Matrix must be unitary."


def test_defgate_param(snapshot):
    dgp = DefGate("TEST", [[1.0, 0.0], [0.0, 1.0]])
    assert dgp.out() == snapshot
    test = dgp.get_constructor()
    tg = test(Qubit(1))
    assert tg.out() == snapshot


def test_defgate_redefintion():
    """Test that adding a defgate with the same name updates the definition."""
    program = Program()
    mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    dgp = DefGate("TEST", mat)
    program += dgp

    assert program.defined_gates[0].name == "TEST"
    assert np.all(program.defined_gates[0].matrix == mat)

    new_mat = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    dgp = DefGate("TEST", new_mat)
    program += dgp

    assert program.defined_gates[0].name == "TEST"
    assert len(program.defined_gates) == 1
    assert np.all(program.defined_gates[0].matrix == new_mat)


def test_defcal_redefinition(snapshot: SnapshotAssertion):
    """Test that adding a DefCalibration with the same name updates the definition."""
    program = Program()
    defcal = DefCalibration("TEST", [], [Qubit(1)], instrs=[RX(np.pi, 1)])
    program += defcal

    assert len(program.calibrations) == 1
    assert program.calibrations[0].instrs[0].out() == snapshot

    program += defcal

    assert len(program.calibrations) == 1

    new_defcal = DefCalibration("TEST", [], [Qubit(1)], instrs=[RX(np.pi / 2, 1)])

    program += new_defcal
    assert len(program.calibrations) == 1
    assert program.calibrations[0].instrs[0].out() == snapshot


def test_defcalmeasure_redefinition(snapshot: SnapshotAssertion):
    """Test that adding a DefMeasureCalibration with the same name updates the definition."""
    program = Program()
    defmeasure = DefMeasureCalibration(Qubit(1), MemoryReference("ro"), [RX(np.pi, 1)])
    program += defmeasure

    assert len(program.measure_calibrations) == 1
    assert program.measure_calibrations[0].instrs[0].out() == snapshot

    program += defmeasure

    assert len(program.measure_calibrations) == 1

    new_defmeasure = DefMeasureCalibration(Qubit(1), MemoryReference("ro"), [RX(np.pi / 2, 1)])

    program += new_defmeasure
    assert len(program.measure_calibrations) == 1
    assert program.measure_calibrations[0].instrs[0].out() == snapshot


def test_inst_gates(snapshot):
    p = Program()
    p.inst(H(0), X(1))
    assert len(p) == 2
    assert p.out() == snapshot


def test_inst_tuple(snapshot):
    p = Program()
    p.inst(("Y", 0), ("X", 1))
    assert len(p) == 2
    assert p.out() == snapshot


def test_inst_rs_gate(snapshot):
    p = Program()
    q = quil_rs.Qubit.from_fixed(0)
    p.inst(quil_rs.Gate("X", [], [q], []))
    assert p.out() == snapshot


def test_inst_string(snapshot):
    p = Program()
    p.inst("Y 0", "X 1")
    assert len(p) == 2
    assert p.out() == snapshot


def test_len_zero():
    prog = Program()
    assert len(prog) == 0


def test_len_one():
    prog = Program(X(0))
    assert len(prog) == 1


def test_len_nested():
    p = Program(Declare("ro", "BIT"), H(0)).measure(0, MemoryReference("ro", 0))
    q = Program(H(0), CNOT(0, 1))
    p.if_then(MemoryReference("ro", 0), q)
    assert len(p) == 9


def test_plus_operator(snapshot):
    p = Program()
    p += H(0)
    p += [X(0), Y(0), Z(0)]
    assert len(p) == 4
    assert p.out() == snapshot


def test_indexing():
    program = (
        Program(Declare("ro", "BIT"), H(0), Y(1), CNOT(0, 1))
        .measure(0, MemoryReference("ro", 0))
        .if_then(MemoryReference("ro", 0), Program(X(0)), Program())
    )
    assert program[1] == H(0)
    for ii, instr in enumerate(program.instructions):
        assert program[ii] == instr


def test_iteration():
    gate_list = [H(0), Y(1), CNOT(0, 1)]
    program = Program(gate_list)
    for ii, instruction in enumerate(program):
        assert str(instruction) == gate_list[ii].out()

    # https://github.com/rigetti/pyquil/issues/265
    gate_generator = (gate_list[ii] for ii in range(3))
    program = Program(gate_generator)
    for ii, instruction in enumerate(program):
        assert str(instruction) == gate_list[ii].out()


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


def test_program_tuple(snapshot):
    p = Program()
    p.inst(("Y", 0), ("X", 1))
    assert len(p) == 2
    assert p.out() == snapshot


def test_program_string(snapshot):
    p = Program()
    p.inst("Y 0", "X 1")
    assert len(p.instructions) == 2
    assert p.out() == snapshot


def test_program_slice():
    p = Program(H(0), CNOT(0, 1), H(1))
    assert isinstance(p[1:], Program)


def test_prog_init(snapshot):
    p = Program()
    p.inst(Declare("ro", "BIT"), X(0)).measure(0, MemoryReference("ro", 0))
    assert p.out() == snapshot


def test_classical_regs():
    p = Program()
    p.inst(
        Declare("ro", "BIT", 2),
        Declare("reg", "BIT", 2),
        X(0),
    ).measure(0, MemoryReference("reg", 1))
    assert p.out() == "DECLARE reg BIT[2]\nDECLARE ro BIT[2]\nX 0\nMEASURE 0 reg[1]\n"
    assert p.declarations == {
        "reg": Declare("reg", "BIT", 2),
        "ro": Declare("ro", "BIT", 2),
    }


def test_simple_instructions(snapshot):
    p = Program().inst(HALT, WAIT, RESET(), NOP)
    assert p.out() == snapshot


def test_unary_classicals(snapshot):
    p = Program()
    p.inst(
        MOVE(MemoryReference("ro", 0), 1),
        MOVE(MemoryReference("ro", 1), 0),
        NOT(MemoryReference("ro", 2)),
        NEG(MemoryReference("ro", 3)),
    )
    assert p.out() == snapshot


def test_binary_classicals(snapshot):
    p = Program()

    p.inst(
        AND(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        MOVE(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        CONVERT(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        IOR(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        XOR(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        ADD(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        SUB(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        MUL(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        DIV(MemoryReference("ro", 0), MemoryReference("ro", 1)),
        EXCHANGE(MemoryReference("ro", 0), MemoryReference("ro", 1)),
    )

    assert p.out() == snapshot


def test_memory_reference_unpacking(snapshot):
    p = Program()

    p.inst(
        AND("ro", ("ro", 1)),
        MOVE("ro", ("ro", 1)),
        CONVERT("ro", ("ro", 1)),
        IOR("ro", ("ro", 1)),
        XOR("ro", ("ro", 1)),
        ADD("ro", ("ro", 1)),
        SUB("ro", ("ro", 1)),
        MUL("ro", ("ro", 1)),
        DIV("ro", ("ro", 1)),
        EXCHANGE("ro", ("ro", 1)),
    )

    assert p.out() == snapshot


def test_ternary_classicals(snapshot):
    p = Program()
    p.inst(
        LOAD(MemoryReference("ro", 0), "ro", MemoryReference("n", 0)),
        STORE("ro", MemoryReference("n", 0), MemoryReference("ro", 0)),
        STORE("ro", MemoryReference("n", 0), 0),
        STORE("ro", MemoryReference("n", 0), 0.1),
        EQ(MemoryReference("ro", 0), MemoryReference("ro", 1), 0),
        EQ(MemoryReference("ro", 0), MemoryReference("ro", 1), 0.0),
        EQ(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 0)),
        GE(MemoryReference("ro", 0), MemoryReference("ro", 1), 1),
        GE(MemoryReference("ro", 0), MemoryReference("ro", 1), 1.1),
        GE(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 1)),
        GT(MemoryReference("ro", 0), MemoryReference("ro", 1), 2),
        GT(MemoryReference("ro", 0), MemoryReference("ro", 1), 2.2),
        GT(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 2)),
        LE(MemoryReference("ro", 0), MemoryReference("ro", 1), 3),
        LE(MemoryReference("ro", 0), MemoryReference("ro", 1), 3.3),
        LE(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 3)),
        LT(MemoryReference("ro", 0), MemoryReference("ro", 1), 4),
        LT(MemoryReference("ro", 0), MemoryReference("ro", 1), 4.4),
        LT(MemoryReference("ro", 0), MemoryReference("ro", 1), MemoryReference("ro", 4)),
    )
    assert p.out() == snapshot


def test_measurement_calls(snapshot):
    p = Program()
    p.inst(
        Declare("ro", "BIT", 2),
        MEASURE(0, MemoryReference("ro", 1)),
        MEASURE(0, MemoryReference("ro", 1)),
    )
    assert p.out() == snapshot


def test_measure_all(snapshot):
    p = Program()
    mem = p.declare("ro", memory_size=4)
    p.measure_all((0, mem[0]), (1, mem[1]), (2, mem[3]))
    assert p.out() == snapshot

    p = Program([H(idx) for idx in range(4)])
    p.measure_all()
    for idx in range(4):
        assert str(p[idx + 5]) == MEASURE(idx, MemoryReference("ro", idx)).out()

    p = Program()
    p.measure_all()
    assert p.out() == ""


def test_dagger():
    p = Program(X(0), H(0))
    assert p.dagger().out() == "DAGGER H 0\nDAGGER X 0\n"

    p = Program(X(0), MEASURE(0, MemoryReference("ro", 0)))
    with pytest.raises(ValueError):
        p.dagger().out()

    # ensure that modifiers are preserved https://github.com/rigetti/pyquil/pull/914
    p = Program()
    control = 0
    target = 1
    cnot_control = 2
    p += X(target).controlled(control)
    p += Y(target).controlled(control)
    p += Z(target).controlled(control)
    p += H(target).controlled(control)
    p += S(target).controlled(control)
    p += T(target).controlled(control)
    p += PHASE(pi, target).controlled(control)
    p += CNOT(cnot_control, target).controlled(control)

    for instr, instr_dagger in zip(reversed(p.instructions), p.dagger().instructions):
        assert "DAGGER " + str(instr) == str(instr_dagger)


def test_construction_syntax(snapshot):
    p = Program().inst(Declare("ro", "BIT", 2), X(0), Y(1), Z(0)).measure(0, MemoryReference("ro", 1))
    assert p.out() == snapshot
    p = (
        Program()
        .inst(Declare("ro", "BIT", 3), X(0))
        .inst(Y(1))
        .measure(0, MemoryReference("ro", 1))
        .inst(MEASURE(1, MemoryReference("ro", 2)))
    )
    assert p.out() == snapshot
    p = (
        Program()
        .inst(Declare("ro", "BIT", 2), X(0))
        .measure(0, MemoryReference("ro", 1))
        .inst(Y(1), X(0))
        .measure(0, MemoryReference("ro", 0))
    )
    assert p.out() == snapshot


def test_singles(snapshot):
    p = Program(I(0), X(0), Y(1), Z(1), H(2), T(2), S(1))
    assert p.out() == snapshot


def test_rotations(snapshot):
    p = Program(RX(0.5, 0), RY(0.1, 1), RZ(1.4, 2))
    assert p.out() == snapshot


def test_controlled_gates(snapshot):
    p = Program(CNOT(0, 1), CCNOT(0, 1, 2))
    assert p.out() == snapshot


def test_phases(snapshot):
    p = Program(
        PHASE(np.pi, 1),
        CPHASE00(np.pi, 0, 1),
        CPHASE01(np.pi, 0, 1),
        CPHASE10(np.pi, 0, 1),
        CPHASE(np.pi, 0, 1),
    )
    assert p.out() == snapshot


def test_swaps(snapshot):
    p = Program(SWAP(0, 1), CSWAP(0, 1, 2), ISWAP(0, 1), PSWAP(np.pi, 0, 1))
    assert p.out() == snapshot


def test_def_gate(snapshot):
    # First we define the new gate from a matrix
    sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
    p = Program().defgate("SQRT-X", sqrt_x)

    # Then we can use the new gate
    p.inst(("SQRT-X", 0))
    assert p.out() == snapshot


def test_def_gate_with_parameters(snapshot: SnapshotAssertion):
    theta = Parameter("theta")
    rx = np.array(
        [
            [quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
            [-1j * quil_sin(theta / 2), quil_cos(theta / 2)],
        ]
    )

    p = Program().defgate("MY_RX", rx, [theta])
    assert p.out() == snapshot

    dg = DefGate("MY_RX", rx, [theta])
    MY_RX = dg.get_constructor()
    p = Program().inst(MY_RX(np.pi)(0))
    assert p.out() == snapshot


def test_multiqubit_gate(snapshot):
    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
    x_sqrt_x = np.kron(sqrt_x, x_gate_matrix)
    p = Program().defgate("X-SQRT-X", x_sqrt_x)

    # Then we can use the new gate
    p.inst(("X-SQRT-X", 0, 1))

    assert p.out() == snapshot


def test_define_qft(snapshot):
    def qft3(q0, q1, q2):
        p = Program()
        p.inst(
            H(q2),
            CPHASE(pi / 2.0, q1, q2),
            H(1),
            CPHASE(pi / 4.0, q0, q2),
            CPHASE(pi / 2.0, q0, q1),
            H(q0),
            SWAP(q0, q2),
        )
        return p

    # I(2) is to force 3 qubits in state prep program.
    state_prep = Program().inst(X(0))

    prog = state_prep + qft3(0, 1, 2)
    output = prog.out()
    assert output == snapshot


def test_control_flows():
    outer_loop = Program()
    classical_flag_register = outer_loop.declare("classical_flag_register", "BIT")
    outer_loop += MOVE(classical_flag_register, 1)  # initialize

    inner_loop = Program()
    inner_loop += Program(X(0), H(0))
    inner_loop += MEASURE(0, classical_flag_register)

    # run inner_loop in a loop until classical_flag_register is 0
    outer_loop.while_do(classical_flag_register, inner_loop)
    outer_loop.resolve_label_placeholders()
    assert outer_loop.out() == "\n".join(
        [
            "DECLARE classical_flag_register BIT[1]",
            "MOVE classical_flag_register[0] 1",
            "LABEL @START_0",
            "JUMP-UNLESS @END_0 classical_flag_register[0]",
            "X 0",
            "H 0",
            "MEASURE 0 classical_flag_register[0]",
            "JUMP @START_0",
            "LABEL @END_0",
            "",
        ]
    )


def test_control_flows_2():
    # create a program that branches based on the value of a classical register
    x_prog = Program(X(0))
    z_prog = Program()
    branch = (
        Program(Declare("ro", "BIT", 2), H(1))
        .measure(1, MemoryReference("ro", 1))
        .if_then(MemoryReference("ro", 1), x_prog, z_prog)
        .measure(0, MemoryReference("ro", 0))
    )
    branch.resolve_label_placeholders()
    assert branch.out() == (
        "DECLARE ro BIT[2]\n"
        "H 1\n"
        "MEASURE 1 ro[1]\n"
        "JUMP-WHEN @THEN_0 ro[1]\n"
        "JUMP @END_0\n"
        "LABEL @THEN_0\n"
        "X 0\n"
        "LABEL @END_0\n"
        "MEASURE 0 ro[0]\n"
    )


def test_if_option():
    p = (
        Program(Declare("ro", "BIT", 1), X(0))
        .measure(0, MemoryReference("ro", 0))
        .if_then(MemoryReference("ro", 0), Program(X(1)))
    )
    p.resolve_label_placeholders()
    assert p.out() == (
        "DECLARE ro BIT[1]\n"
        "X 0\n"
        "MEASURE 0 ro[0]\n"
        "JUMP-WHEN @THEN_0 ro[0]\n"
        "JUMP @END_0\n"
        "LABEL @THEN_0\n"
        "X 1\n"
        "LABEL @END_0\n"
    )

    assert isinstance(p.instructions[3], JumpWhen)


def test_qubit_placeholder():
    p = Program()

    p.inst(H(0))  # H 0

    q1 = QubitPlaceholder()  # q1 = 1
    q2 = QubitPlaceholder()  # q2 = 3

    p.inst(CNOT(q1, q2))  # CNOT 1 3

    p.inst(H(2))

    q3 = QubitPlaceholder()  # q3 = 4

    p.inst(X(q3))  # X 4

    with pytest.raises(ValueError) as e:
        _ = p.out()
    assert e.match("Qubit has not yet been resolved")


def test_qubit_placeholder_2():
    p = Program()

    p.inst(H(0))  # H 0

    q1 = QubitPlaceholder()  # q1 = 1
    q2 = QubitPlaceholder()  # q2 = 3

    p.inst(CNOT(q1, q2))  # CNOT 1 3

    p.inst(H(2))

    q3 = QubitPlaceholder()  # q3 = 4

    p.inst(X(q3))  # X 4


def test_qubit_placeholder_new():
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
    p = address_qubits(p, {q1: 1, q2: 3, q3: 4, q0: 0, qxxx: 2})

    assert p.out() == "H 0\nCNOT 1 3\nH 2\nX 4\n"


def test_multiaddress():
    p = Program()
    q0, q1 = [QubitPlaceholder() for _ in range(2)]
    p += exponential_map(sZ(q0) * sZ(q1))(0.5)

    map1 = {q0: 0, q1: 1}
    map2 = {q0: 9, q1: 10}

    p1 = address_qubits(p, map1)

    with pytest.raises(ValueError):
        _ = p.out()  # make sure the original isn't affected

    assert p1.out() == "CNOT 0 1\nRZ(1) 1\nCNOT 0 1\n"

    p2 = address_qubits(p, map2)
    assert p1.out() == "CNOT 0 1\nRZ(1) 1\nCNOT 0 1\n"
    assert p2.out() == "CNOT 9 10\nRZ(1) 10\nCNOT 9 10\n"


def test_multiple_instantiate():
    p = Program()
    q = QubitPlaceholder()
    p.inst(H(q))
    p = address_qubits(p)
    assert p.out() == "H 0\n"
    assert p.out() == "H 0\n"


def test_reuse_placeholder():
    p = Program()
    q1 = QubitPlaceholder()
    q2 = QubitPlaceholder()
    p.inst(H(q1))
    p.inst(H(q2))
    p.inst(CNOT(q1, q2))
    p = address_qubits(p)
    assert p.out() == "H 0\nH 1\nCNOT 0 1\n"


def test_pauli_gate(snapshot: SnapshotAssertion):
    s = """DEFGATE U(%beta) p q AS PAULI-SUM:
    ZZ(-1*%beta/4) p q
    Z(%beta/4) p
    Z(%beta/4) q

DEFGATE V:
    1.0, 0
    0, 1.0

U(2.0) 1 0
"""
    p = Program(s)
    assert p.out() == snapshot


def test_prog_merge(snapshot: SnapshotAssertion):
    prog_0 = Program(X(0))
    prog_1 = Program(Y(0))
    assert merge_programs([prog_0, prog_1]).out() == (prog_0 + prog_1).out()
    test_def = DefGate("test", np.eye(2))
    TEST = test_def.get_constructor()
    prog_0.inst(test_def)
    prog_0.inst(TEST(0))
    prog_1.inst(test_def)
    prog_1.inst(TEST(0))
    assert merge_programs([prog_0, prog_1]).out() == snapshot
    perm_def = DefPermutationGate("PERM", [0, 1, 3, 2])
    PERM = perm_def.get_constructor()
    prog_0.inst(perm_def)
    prog_0.inst(PERM(0, 1))
    prog_1.inst(perm_def)
    prog_1.inst(PERM(1, 0))
    assert merge_programs([prog_0, prog_1]).out() == snapshot
    assert merge_programs([Program("DECLARE ro BIT[1]"), Program("H 0"), Program("MEASURE 0 ro[0]")]).out() == snapshot


def test_merge_with_pauli_noise(snapshot):
    p = Program(X(0)).inst(Z(0))
    probs = [0.0, 1.0, 0.0, 0.0]
    merged = merge_with_pauli_noise(p, probs, [0])
    assert merged.out() == snapshot


def test_get_qubits():
    pq = Program(Declare("ro", "BIT"), X(0), CNOT(0, 4), MEASURE(5, MemoryReference("ro", 0)))
    assert pq.get_qubits() == {0, 4, 5}

    q = [QubitPlaceholder() for _ in range(6)]
    pq = Program(Declare("ro", "BIT"), X(q[0]), CNOT(q[0], q[4]), MEASURE(q[5], MemoryReference("ro", 0)))
    qq = QubitPlaceholder()
    pq.inst(Y(q[2]), X(qq))
    addressed_pq = address_qubits(pq)
    assert addressed_pq.get_qubits() == {0, 1, 2, 3, 4}

    qubit_index = 1
    p = Program(("H", qubit_index))
    assert p.get_qubits() == {qubit_index}
    q1 = QubitPlaceholder()
    q2 = QubitPlaceholder()
    p.inst(("CNOT", q1, q2))


def test_get_qubit_placeholders():
    qs = QubitPlaceholder.register(8)
    pq = Program(Declare("ro", "BIT"), X(qs[0]), CNOT(qs[0], qs[4]), MEASURE(qs[5], MemoryReference("ro", 0)))
    assert set(pq.get_qubits(indices=False)) == {qs[i] for i in [0, 4, 5]}


def test_get_qubits_not_as_indices():
    pq = Program(Declare("ro", "BIT"), X(0), CNOT(0, 4), MEASURE(5, MemoryReference("ro", 0)))
    assert set(pq.get_qubits(indices=False)) == set(Qubit(i) for i in [0, 4, 5])


def test_eq():
    p1 = Program()
    q1 = QubitPlaceholder()
    q2 = QubitPlaceholder()
    p1.inst([H(q1), CNOT(q1, q2)])
    p1 = address_qubits(p1)

    p2 = Program()
    p2.inst([H(0), CNOT(0, 1)])

    assert p1 == p2
    assert not p1 != p2


def test_kraus(snapshot):
    pq = Program(X(0))
    pq.define_noisy_gate("X", (0,), [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    pq.inst(X(1))
    pq.define_noisy_gate("X", (1,), [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    ret = pq.out()
    assert (
        ret
        == """X 0
PRAGMA ADD-KRAUS X 0 "(0.0 1.0 1.0 0.0)"
PRAGMA ADD-KRAUS X 0 "(0.0 0.0 0.0 0.0)"
X 1
PRAGMA ADD-KRAUS X 1 "(0.0 1.0 1.0 0.0)"
PRAGMA ADD-KRAUS X 1 "(0.0 0.0 0.0 0.0)"
"""
    )
    # test error due to bad normalization
    with pytest.raises(ValueError):
        pq.define_noisy_gate("X", (0,), [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
    # test error due to bad shape of kraus op
    with pytest.raises(ValueError):
        pq.define_noisy_gate("X", (0,), [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])

    pq1 = Program(X(0))
    pq1.define_noisy_gate("X", (0,), [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    pq2 = Program(X(1))
    pq2.define_noisy_gate("X", (1,), [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    assert pq1 + pq2 == pq

    pq_nn = Program(X(0))
    pq_nn.no_noise()
    pq_nn.inst(X(1))

    assert pq_nn.out() == snapshot


def test_define_noisy_readout():
    pq = Program(X(0))
    pq.define_noisy_readout(0, 0.8, 0.9)

    pq.inst(X(1))
    pq.define_noisy_readout(1, 0.9, 0.8)

    ret = pq.out()
    assert (
        ret
        == """X 0
PRAGMA READOUT-POVM 0 "(0.8 0.09999999999999998 0.19999999999999996 0.9)"
X 1
PRAGMA READOUT-POVM 1 "(0.9 0.19999999999999996 0.09999999999999998 0.8)"
"""
    )
    # test error due to bad normalization
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, 1.1, 0.5)
    # test error due to bad normalization
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, 0.5, 1.5)
    # test error due to negative probability
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, -0.1, 0.5)
    # test error due to negative probability
    with pytest.raises(ValueError):
        pq.define_noisy_readout(0, 0.5, -1.0)
    # test error due to bad qubit_index value
    with pytest.raises(ValueError):
        pq.define_noisy_readout(-1, 0.5, 0.5)
    # test error due to bad qubit_index type
    with pytest.raises(TypeError):
        pq.define_noisy_readout(1.0, 0.5, 0.5)


# https://github.com/rigetti/pyquil/issues/72
def test_if_then_inherits_defined_gates():
    p1 = Program()
    p1.inst(H(0))
    p1.measure(0, MemoryReference("ro", 0))

    p2 = Program()
    p2.defgate("A", np.array([[1.0, 0.0], [0.0, 1.0]]))
    p2.inst(("A", 0))

    p3 = Program()
    p3.defgate("B", np.array([[0.0, 1.0], [1.0, 0.0]]))
    p3.inst(("B", 0))

    p1.if_then(MemoryReference("ro", 0), p2, p3)
    assert p2.defined_gates[0] in p1.defined_gates
    assert p3.defined_gates[0] in p1.defined_gates


# https://github.com/rigetti/pyquil/issues/124
def test_allocating_qubits_on_multiple_programs():
    p = Program()
    qubit0 = QubitPlaceholder()
    p.inst(X(qubit0))

    q = Program()
    qubit1 = QubitPlaceholder()
    q.inst(X(qubit1))

    assert address_qubits(p + q).out() == "X 0\nX 1\n"


# https://github.com/rigetti/pyquil/issues/163
def test_installing_programs_inside_other_programs():
    p = Program()
    q = Program()
    p.inst(q)
    assert len(p) == 0


# https://github.com/rigetti/pyquil/issues/170
def test_inline_placeholder():
    p = Program()
    p += H(QubitPlaceholder())
    assert address_qubits(p).out() == "H 0\n"


# https://github.com/rigetti/pyquil/issues/138
def test_defgate_integer_input():
    dg = DefGate("TEST", np.array([[1, 0], [0, 1]]))
    assert dg.out() == "DEFGATE TEST AS MATRIX:\n\t1, 0\n\t0, 1\n"


def test_out_vs_str():
    qs = QubitPlaceholder.register(6)
    pq = Program(
        Declare("ro", "BIT", 6),
        X(qs[0]),
        CNOT(qs[0], qs[4]),
        MEASURE(qs[5], MemoryReference("ro", 5)),
    )

    with pytest.raises(ValueError) as e:
        pq.out()
    assert e.match(r"Qubit has not yet been resolved")

    string_version = str(pq)
    should_be_re = r"DECLARE ro BIT\[6\]\nX Placeholder\(QubitPlaceholder\(0x[0-9,A-Z]+\)\)\nCNOT Placeholder\(QubitPlaceholder\(0x[0-9,A-Z]+\)\) Placeholder\(QubitPlaceholder\(0x[0-9,A-Z]+\)\)\nMEASURE Placeholder\(QubitPlaceholder\(0x[0-9,A-Z]+\)\) ro\[5\]\n"
    assert re.fullmatch(should_be_re, string_version, flags=re.MULTILINE)


def test_get_classical_addresses_from_program():
    p = Program(Declare("ro", "BIT", 4), [H(i) for i in range(4)])
    assert get_classical_addresses_from_program(p) == {}

    p += [MEASURE(i, MemoryReference("ro", i)) for i in [0, 3, 1]]
    assert get_classical_addresses_from_program(p) == {"ro": [0, 1, 3]}


def test_get_classical_addresses_from_quil_program():
    """
    Tests that can get_classical_addresses_from_program can handle both MEASURE
    quil instructions with and without explicit classical registers.
    """

    p = Program("\n".join(["X 0", "MEASURE 0"]))
    assert get_classical_addresses_from_program(p) == {}

    p = Program("\n".join(["DECLARE ro BIT[2]", "X 0", "MEASURE 0 ro[1]"]))
    assert get_classical_addresses_from_program(p) == {"ro": [1]}


def test_declare():
    program = Program(Declare("read_out", "BIT", 5), MEASURE(0, MemoryReference("read_out", 4)))
    assert program.out() == ("DECLARE read_out BIT[5]\nMEASURE 0 read_out[4]\n")


def test_reset():
    p = Program()
    p.reset(0)
    p.reset()
    assert p.out() == "RESET 0\nRESET\n"

    program = Program()
    qubit = QubitPlaceholder()
    # address_qubits() won't work unless there's a gate besides
    # RESET on a QubitPlaceholder, this is just here to make
    # addressing work
    program += X(qubit)

    program += RESET(qubit)
    program = address_qubits(program)
    assert program.out() == "X 0\nRESET 0\n"


def test_copy(snapshot):
    prog1 = Program(H(0), CNOT(0, 1))
    prog2 = prog1.copy().measure_all()
    assert prog1.out() == snapshot
    assert prog2.out() == snapshot


def test_measure_all_noncontig(snapshot):
    prog = Program(H(0), H(10))
    prog.measure_all()
    assert prog.out() == snapshot


# As of pyQuil v4, this function is a no-op
def test_validate_supported_quil():
    prog = Program(H(0), RESET())
    validate_supported_quil(prog)
    assert prog.is_supported_on_qpu()


def test_subtracting_memory_regions():
    # https://github.com/rigetti/pyquil/issues/709
    p = Program()
    alpha = p.declare("alpha", "REAL")
    beta = p.declare("beta", "REAL")
    p += RZ(alpha - beta, 0)
    p2 = Program(p.out())
    parsed_rz = p2[-1]
    parsed_param = parsed_rz.params[0]  # type: Gate
    assert isinstance(parsed_param, Sub)
    assert parsed_param.op1 == alpha
    assert parsed_param.op2 == beta


def test_out_of_bounds_memory():
    r = Program().declare("ro", "BIT", 1)
    with pytest.raises(IndexError):
        r[1]


@pytest.mark.timeout(5)
def test_memory_reference_iteration():
    r = Program().declare("ro", "BIT", 10)
    assert len([i for i in r]) == 10


def test_placeholders_preserves_modifiers():
    cs = QubitPlaceholder.register(3)
    ts = QubitPlaceholder.register(1)

    g = X(ts[0])
    for c in cs:
        g = g.controlled(c).dagger()

    p = Program(g)
    a = address_qubits(p)

    assert a[0].modifiers == g.modifiers


def _eval_as_np_pi(exp):
    return eval(exp.replace("pi", repr(np.pi)).replace("theta[0]", "1"))


def test_params_pi_and_precedence():
    trivial_pi = "3 * theta[0] / (2 * pi)"
    prog = Program(f"RX({trivial_pi}) 0")
    exp = str(prog[0].params[0])
    assert _eval_as_np_pi(trivial_pi) == _eval_as_np_pi(exp)

    less_trivial_pi = "3 * theta[0] * 2 / (pi)"
    prog = Program(f"RX({less_trivial_pi}) 0")
    exp = str(prog[0].params[0])
    assert _eval_as_np_pi(less_trivial_pi) == _eval_as_np_pi(exp)

    more_less_trivial_pi = "3 / (theta[0] / (pi + 1)) / pi"
    prog = Program(f"RX({more_less_trivial_pi}) 0")
    exp = str(prog[0].params[0])
    assert _eval_as_np_pi(more_less_trivial_pi) == _eval_as_np_pi(exp)


class TestProgram:
    def test_calibrations(self, snapshot: SnapshotAssertion):
        program = Program(
            "DEFCAL Calibrate 0:\n\tX 0",
            DefCalibration("Reticulating-Splines", [Parameter("Spline")], [Qubit(1)], [Y(1)]),
            DefMeasureCalibration(Qubit(2), MemoryReference("theta"), [Z(2)]),
        )

        calibrations = program.calibrations
        measure_calibrations = program.measure_calibrations
        assert all((isinstance(cal, DefCalibration) for cal in program.calibrations))
        assert all((isinstance(cal, DefMeasureCalibration) for cal in program.measure_calibrations))
        assert calibrations == snapshot
        assert measure_calibrations == snapshot

    def test_frames(self, snapshot: SnapshotAssertion):
        program = Program(
            'DEFFRAME 1 "frame":\n\tCENTER-FREQUENCY: 440',
            DefFrame(Frame([Qubit(1)], "other_frame"), center_frequency=432.0),
        )
        frames = program.frames
        assert all(
            (isinstance(frame, Frame) and isinstance(def_frame, DefFrame) for frame, def_frame in frames.items())
        )
        assert frames == snapshot


def test_copy_everything_except_instructions():
    """Test for https://github.com/rigetti/pyquil/issues/1613"""
    program = Program(
        """
DECLARE beta REAL[1]
RZ(0.5) 0
CPHASE(pi) 0 1
DECLARE ro BIT[2]
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""
    )
    program = program.copy_everything_except_instructions()
    assert len(program.instructions) == 0  # the purpose of copy_everything_except_instructions()
    assert len(program.declarations) == 0  # this is a view on the instructions member; must be consistent

def test_cached_frames():
    frames = [
        DefFrame(Frame([Qubit(0)], "frame0"), center_frequency=432.0),
        DefFrame(Frame([Qubit(1)], "frame1"), sample_rate=44100.0),
    ]

    p = Program(frames[0])
    program_frames = p.frames
    assert program_frames == {frames[0].frame: frames[0]}

    p.inst(frames[1])
    program_frames = p.frames
    assert program_frames == {frames[0].frame: frames[0], frames[1].frame: frames[1]}
