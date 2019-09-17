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
import numpy as np
import pytest

from pyquil.gates import *
from pyquil.parser import parse
from pyquil.quilatom import MemoryReference, Parameter, quil_cos, quil_sin
from pyquil.quilbase import Declare, Reset, ResetQubit
from pyquil.quilbase import Label, JumpTarget, Jump, JumpWhen, JumpUnless, DefGate, DefPermutationGate, Qubit, Pragma, \
    RawInstr
from pyquil.tests.utils import parse_equals


def test_simple_gate():
    parse_equals("A 0", Gate("A", [], [Qubit(0)]))
    parse_equals("A 1 10 100", Gate("A", [], [Qubit(1), Qubit(10), Qubit(100)]))


def test_standard_gates():
    parse_equals("H 0", H(0))
    parse_equals("CNOT 0 1", CNOT(0, 1))
    parse_equals("SWAP 0 1", SWAP(0, 1))


def test_def_gate():
    sqrt_x = DefGate("SQRT-X", np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                                         [0.5 - 0.5j, 0.5 + 0.5j]]))
    hadamard = DefGate("HADAMARD", np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                                             [1 / np.sqrt(2), -1 / np.sqrt(2)]]))
    defgates = """
DEFGATE SQRT-X:
    0.5+0.5i, 0.5-0.5i
    0.5-0.5i, 0.5+0.5i

DEFGATE HADAMARD:
    1/SQRT(2), 1/SQRT(2)
    1/SQRT(2), -1/SQRT(2)
    """.strip()

    parse_equals(defgates, sqrt_x, hadamard)


def test_def_gate_with_variables():
    # Note that technically the RX gate includes -i instead of just i but this messes a bit with the test since
    # it's not smart enough to figure out that -1*i == -i
    theta = Parameter('theta')
    rx = np.array([[quil_cos(theta / 2), 1j * quil_sin(theta / 2)],
                   [1j * quil_sin(theta / 2), quil_cos(theta / 2)]])

    defgate = 'DEFGATE RX(%theta):\n' \
              '    COS(%theta/2), i*SIN(%theta/2)\n' \
              '    i*SIN(%theta/2), COS(%theta/2)\n\n'

    parse_equals(defgate, DefGate('RX', rx, [theta]))


def test_def_gate_as():
    perm_gate_str = 'DEFGATE CCNOT AS PERMUTATION:\n    0, 1, 2, 3, 4, 5, 7, 6'.strip()
    matrix_gate_str = 'DEFGATE CNOT AS MATRIX:\n    1.0, 0.0, 0.0, 0.0\n    0.0, 1.0, 0.0, 0.0\n    0.0, 0.0, 0.0, 1.0\n    0.0, 0.0, 1.0, 0.0'.strip()
    unknown_gate_str = 'DEFGATE CCNOT AS UNKNOWNTYPE:\n    0, 1, 2, 3, 4, 5, 7, 6'.strip()

    parse(perm_gate_str)
    parse(matrix_gate_str)
    with pytest.raises(RuntimeError):
        parse(unknown_gate_str)


def test_def_gate_as_matrix():
    matrix_gate_str = 'DEFGATE CNOT AS MATRIX:\n    1.0, 0.0, 0.0, 0.0\n    0.0, 1.0, 0.0, 0.0\n    0.0, 0.0, 0.0, 1.0\n    0.0, 0.0, 1.0, 0.0'.strip()
    parsed = parse(matrix_gate_str)

    assert len(parsed) == 1
    assert isinstance(parsed[0], DefGate)
    assert not isinstance(parsed[0], DefPermutationGate)


def test_def_permutation_gate():
    perm_gate = DefPermutationGate("CCNOT", [0, 1, 2, 3, 4, 5, 7, 6])

    perm_gate_str = 'DEFGATE CCNOT AS PERMUTATION:\n    0, 1, 2, 3, 4, 5, 7, 6'.strip()

    parse_equals(perm_gate_str, perm_gate)


def test_def_gate_as_permutation():
    perm_gate_str = 'DEFGATE CCNOT AS PERMUTATION:\n    0, 1, 2, 3, 4, 5, 7, 6'.strip()
    parsed = parse(perm_gate_str)

    assert len(parsed) == 1
    assert isinstance(parsed[0], DefGate)
    assert isinstance(parsed[0], DefPermutationGate)

    # perm gates are defined by a single row of entries, unlike general defgates
    bad_perm_gate_str = 'DEFGATE CCNOT AS PERMUTATION:\n    0, 1, 2, 3, 4, 5, 7, 6\n    0, 1, 2, 3, 4, 5, 7, 6'.strip()
    with pytest.raises(RuntimeError):
        parse(bad_perm_gate_str)


def test_parameters():
    parse_equals("RX(123) 0", RX(123, 0))
    parse_equals("CPHASE00(0) 0 1", CPHASE00(0, 0, 1))
    parse_equals("A(8,9) 0", Gate("A", [8, 9], [Qubit(0)]))
    parse_equals("A(8, 9) 0", Gate("A", [8, 9], [Qubit(0)]))


def test_expressions():
    # Test expressions by wrapping them in an RX gate for convenience
    def _expr(expression, expected):
        parse_equals("RX(" + expression + ") 0", RX(expected, 0))

    # Decimals
    _expr("+123", 123)
    _expr("-123", -123)
    _expr("123.456", 123.456)
    _expr("+123.456", 123.456)
    _expr("-123.456", -123.456)

    # Exponential
    _expr("1e3", 1000.0)
    _expr("1.5e2", 150.0)
    _expr("3.5919865395417361e-05", 3.5919865395417361e-05)
    _expr("3.5919865395417361E-05", 3.5919865395417361e-05)

    # Complex
    _expr("123.456i", complex(0, 123.456))
    _expr("+123.456i", complex(0, 123.456))
    # Edge case: making the whole complex number negative makes the real part -0.0
    _expr("-123.456i", complex(-0., -123.456))
    _expr("777+123.456i", complex(777, 123.456))
    _expr("777-123.456i", complex(777, -123.456))
    _expr("+777-123.456i", complex(777, -123.456))

    # Imaginary
    _expr("i * 2", complex(0, 2))
    _expr("2i", complex(0, 2))
    _expr("1 ^ 2", 1)

    # Pi
    _expr("pi", np.pi)
    _expr("pi / 2", np.pi / 2)
    _expr("-pi / 2", np.pi / -2)

    # Expressions
    _expr("1+2", 3)
    _expr("1-2", -1)
    _expr("3*4", 12)
    _expr("6/2", 3.0)
    _expr("2^3", 8)

    # Order of operations
    _expr("3 + 6 * (5 + 4) / 3 - 7", 14.0)
    _expr("3 ^ 2 + 5", 14)

    # Functions
    _expr("SIN(0)", 0.0)
    _expr("COS(0)", 1.0)
    _expr("SQRT(4)", 2.0)
    _expr("EXP(0)", 1.0)
    _expr("CIS(0)", complex(1, 0))

    # Unary precedence
    # https://github.com/rigetti/pyquil/issues/246
    _expr("-3+4", 1)
    _expr("-(3+4)", -7)
    _expr("-(3-4)", 1)
    _expr("-0.1423778799706841+0.5434363975682295i", complex(-0.1423778799706841, 0.5434363975682295))


def test_measure():
    parse_equals("MEASURE 0", MEASURE(0, None))
    parse_equals("MEASURE 0 ro[1]", MEASURE(0, MemoryReference("ro", 1)))


def test_jumps():
    parse_equals("LABEL @test_1", JumpTarget(Label("test_1")))
    parse_equals("JUMP @test_1", Jump(Label("test_1")))
    parse_equals("JUMP-WHEN @test_1 ro[0]", JumpWhen(Label("test_1"), MemoryReference("ro", 0)))
    parse_equals("JUMP-UNLESS @test_1 ro[1]", JumpUnless(Label("test_1"), MemoryReference("ro", 1)))


def test_others():
    parse_equals("RESET", RESET())
    parse_equals("WAIT", WAIT)
    parse_equals("NOP", NOP)


def test_memory_commands():
    parse_equals("DECLARE mem OCTET[32] SHARING mem2 OFFSET 16 REAL OFFSET 32 REAL",
                 Declare("mem", "OCTET", 32, shared_region="mem2", offsets=[(16, "REAL"), (32, "REAL")]))
    parse_equals("STORE mem ro[2] ro[0]", STORE("mem", MemoryReference("ro", 2), MemoryReference("ro", 0)))
    parse_equals("STORE mem ro[2] 7", STORE("mem", MemoryReference("ro", 2), 7))
    parse_equals("LOAD ro[8] mem mem[4]", LOAD(MemoryReference("ro", 8), "mem", MemoryReference("mem", 4)))
    parse_equals("CONVERT ro[1] ro[2]", CONVERT(MemoryReference("ro", 1), MemoryReference("ro", 2)))
    parse_equals("EXCHANGE ro[0] ro[1]", EXCHANGE(MemoryReference("ro", 0), MemoryReference("ro", 1)))
    parse_equals("MOVE mem[2] 4", MOVE(MemoryReference("mem", 2), 4))
    parse_equals("MOVE mem[2] -4", MOVE(MemoryReference("mem", 2), -4))
    parse_equals("MOVE mem[2] -4.1", MOVE(MemoryReference("mem", 2), -4.1))


def test_classical():
    parse_equals("MOVE ro[0] 1", MOVE(MemoryReference("ro", 0), 1))
    parse_equals("MOVE ro[0] 0", MOVE(MemoryReference("ro", 0), 0))
    parse_equals("NOT ro[0]", NOT(MemoryReference("ro", 0)))
    parse_equals("AND ro[0] 1", AND(MemoryReference("ro", 0), 1))
    parse_equals("IOR ro[0] 1", IOR(MemoryReference("ro", 0), 1))
    parse_equals("MOVE ro[0] 1", MOVE(MemoryReference("ro", 0), 1))
    parse_equals("XOR ro[0] 1", XOR(MemoryReference("ro", 0), 1))
    parse_equals("ADD mem[0] 1.2", ADD(MemoryReference("mem", 0), 1.2))
    parse_equals("SUB mem[0] 1.2", SUB(MemoryReference("mem", 0), 1.2))
    parse_equals("MUL mem[0] 1.2", MUL(MemoryReference("mem", 0), 1.2))
    parse_equals("DIV mem[0] 1.2", DIV(MemoryReference("mem", 0), 1.2))
    parse_equals("ADD mem[0] -1.2", ADD(MemoryReference("mem", 0), -1.2))
    parse_equals("SUB mem[0] -1.2", SUB(MemoryReference("mem", 0), -1.2))
    parse_equals("MUL mem[0] -1.2", MUL(MemoryReference("mem", 0), -1.2))
    parse_equals("DIV mem[0] -1.2", DIV(MemoryReference("mem", 0), -1.2))
    parse_equals("EQ comp[1] ro[3] ro[2]",
                 EQ(MemoryReference("comp", 1), MemoryReference("ro", 3), MemoryReference("ro", 2)))
    parse_equals("LT comp[1] ro[3] ro[2]",
                 LT(MemoryReference("comp", 1), MemoryReference("ro", 3), MemoryReference("ro", 2)))
    parse_equals("LE comp[1] ro[3] ro[2]",
                 LE(MemoryReference("comp", 1), MemoryReference("ro", 3), MemoryReference("ro", 2)))
    parse_equals("GT comp[1] ro[3] ro[2]",
                 GT(MemoryReference("comp", 1), MemoryReference("ro", 3), MemoryReference("ro", 2)))
    parse_equals("GE comp[1] ro[3] ro[2]",
                 GE(MemoryReference("comp", 1), MemoryReference("ro", 3), MemoryReference("ro", 2)))
    parse_equals("EQ comp[1] ro[3] 0",
                 EQ(MemoryReference("comp", 1), MemoryReference("ro", 3), 0))
    parse_equals("LT comp[1] ro[3] 1",
                 LT(MemoryReference("comp", 1), MemoryReference("ro", 3), 1))
    parse_equals("LE comp[1] ro[3] 2",
                 LE(MemoryReference("comp", 1), MemoryReference("ro", 3), 2))
    parse_equals("GT comp[1] ro[3] 3",
                 GT(MemoryReference("comp", 1), MemoryReference("ro", 3), 3))
    parse_equals("GE comp[1] ro[3] 4",
                 GE(MemoryReference("comp", 1), MemoryReference("ro", 3), 4))
    parse_equals("EQ comp[1] ro[3] 0.0",
                 EQ(MemoryReference("comp", 1), MemoryReference("ro", 3), 0.0))
    parse_equals("LT comp[1] ro[3] 1.1",
                 LT(MemoryReference("comp", 1), MemoryReference("ro", 3), 1.1))
    parse_equals("LE comp[1] ro[3] 2.2",
                 LE(MemoryReference("comp", 1), MemoryReference("ro", 3), 2.2))
    parse_equals("GT comp[1] ro[3] 3.3",
                 GT(MemoryReference("comp", 1), MemoryReference("ro", 3), 3.3))
    parse_equals("GE comp[1] ro[3] 4.4",
                 GE(MemoryReference("comp", 1), MemoryReference("ro", 3), 4.4))


def test_pragma():
    parse_equals('PRAGMA gate_time H "10 ns"', Pragma('gate_time', ['H'], '10 ns'))
    parse_equals('PRAGMA qubit 0', Pragma('qubit', [0]))
    parse_equals('PRAGMA NO-NOISE', Pragma('NO-NOISE'))


def test_invalid():
    with pytest.raises(RuntimeError):
        parse("H X")


def test_empty_program():
    parse_equals("")


def test_def_circuit():
    defcircuit = """
DEFCIRCUIT bell a b:
    H a
    CNOT a b
""".strip()
    defcircuit_no_qubits = """
DEFCIRCUIT bell:
    H 0
    CNOT 0 1
""".strip()
    parse_equals(defcircuit, RawInstr(defcircuit))
    parse_equals(defcircuit_no_qubits, RawInstr(defcircuit_no_qubits))


def test_parse_reset_qubit():
    reset = """
RESET
    """.strip()
    parse_equals(reset, Reset())

    reset_qubit = """
RESET 5
    """.strip()
    parse_equals(reset_qubit, ResetQubit(Qubit(5)))


def test_defcircuit_measure_qubit():
    defcircuit_measure_named_qubits = """
DEFCIRCUIT test_defcirc_measure_named a b:
    MEASURE a b
""".strip()
    defcircuit_measure_qubits = """
DEFCIRCUIT test_defcirc_measure_qubits:
    MEASURE 0 ro
""".strip()
    defcircuit_measure_qubits_mixed = """
DEFCIRCUIT test_defcirc_measure_mixed q:
    MEASURE q ro
""".strip()
    parse_equals(defcircuit_measure_named_qubits, RawInstr(defcircuit_measure_named_qubits))
    parse_equals(defcircuit_measure_qubits, RawInstr(defcircuit_measure_qubits))
    parse_equals(defcircuit_measure_qubits_mixed, RawInstr(defcircuit_measure_qubits_mixed))


def test_defcircuit_reset_named_qubit():
    defcircuit_reset_named_qubit = """
DEFCIRCUIT test_defcirc_reset_named_qubit a:
    RESET a
""".strip()
    defcircuit_reset_qubit = """
DEFCIRCUIT test_defcirc_reset_qubit:
    RESET 1
""".strip()
    parse_equals(defcircuit_reset_named_qubit, RawInstr(defcircuit_reset_named_qubit))
    parse_equals(defcircuit_reset_qubit, RawInstr(defcircuit_reset_qubit))


def test_parse_dagger():
    s = "DAGGER X 0"
    parse_equals(s, X(0).dagger())
    s = "DAGGER DAGGER X 0"
    parse_equals(s, X(0).dagger().dagger())


def test_parse_controlled():
    s = "CONTROLLED X 0 1"
    parse_equals(s, X(1).controlled(0))


def test_parse_forked():
    s = "FORKED RX(0, pi/2) 0 1"
    parse_equals(s, RX(0, 1).forked(0, [np.pi / 2]))


def test_messy_modifiers():
    s = "FORKED DAGGER CONTROLLED FORKED RX(0.1,0.2,0.3,0.4) 0 1 2 3"
    parse_equals(s,
                 RX(0.1, 3)
                 .forked(2, [0.2])
                 .controlled(1)
                 .dagger()
                 .forked(0, [0.3, 0.4]))
