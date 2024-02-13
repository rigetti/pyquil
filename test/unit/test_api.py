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
from math import pi

import numpy as np

from pyquil.external.rpcq import  CompilerISA, Qubit, Edge
from pyquil.gates import CNOT, H, MEASURE, PHASE, RZ, RX, CZ
from pyquil.paulis import PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Halt, Declare
from pyquil.simulation.tools import program_unitary

EMPTY_PROGRAM = Program()
BELL_STATE = Program(H(0), CNOT(0, 1))
BELL_STATE_MEASURE = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, MemoryReference("ro", 0)),
    MEASURE(1, MemoryReference("ro", 1)),
)
COMPILED_BELL_STATE = Program(
    [
        RZ(pi / 2, 0),
        RX(pi / 2, 0),
        RZ(-pi / 2, 1),
        RX(pi / 2, 1),
        CZ(1, 0),
        RZ(-pi / 2, 0),
        RX(-pi / 2, 1),
        RZ(pi / 2, 1),
        Halt(),
    ]
)
DUMMY_ISA = CompilerISA(
    qubits={
        "0": Qubit(id=0),
        "1": Qubit(id=1),
    },
    edges={
        "0-1": Edge(ids=[0, 1]),
    },
)

COMPILED_BYTES_ARRAY = b"SUPER SECRET PACKAGE"
RB_ENCODED_REPLY = [[0, 0], [1, 1]]
RB_REPLY = [Program("H 0\nH 0\n"), Program("PHASE(pi/2) 0\nPHASE(pi/2) 0\n")]


def test_quil_to_native_quil(compiler):
    response = compiler.quil_to_native_quil(BELL_STATE)
    p_unitary = program_unitary(response, n_qubits=2)
    compiled_p_unitary = program_unitary(COMPILED_BELL_STATE, n_qubits=2)
    from pyquil.simulation.tools import scale_out_phase

    assert np.allclose(p_unitary, scale_out_phase(compiled_p_unitary, p_unitary))


def test_local_rb_sequence(benchmarker, snapshot):
    response = benchmarker.generate_rb_sequence(2, [PHASE(np.pi / 2, 0), H(0)], seed=52)
    assert [prog.out() for prog in response] == snapshot


def test_local_conjugate_request(benchmarker):
    response = benchmarker.apply_clifford_to_pauli(Program("H 0"), PauliTerm("X", 0, 1.0))
    assert isinstance(response, PauliTerm)
    assert str(response) == "(1+0j)*Z0"


def test_apply_clifford_to_pauli(benchmarker):
    response = benchmarker.apply_clifford_to_pauli(Program("H 0"), PauliTerm("I", 0, 0.34))
    assert response == PauliTerm("I", 0, 0.34)
