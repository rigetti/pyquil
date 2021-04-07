##############################################################################
# Copyright 2016-2021 Rigetti Computing
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
import pytest
import os

from pyquil import get_qc, Program
from pyquil.api import QuantumComputer, QPU
from pyquil.gates import H, CNOT, MEASURE
from pyquil.quilbase import Declare


@pytest.fixture()
def qc() -> QuantumComputer:
    quantum_processor_id = os.environ.get("TEST_QUANTUM_PROCESSOR")
    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    computer = get_qc(quantum_processor_id)
    if not isinstance(computer.qam, QPU):
        raise ValueError("'TARGET_QUANTUM_PROCESSOR' should name a QPU. Ensure it is not a QVM.")
    return computer


def test_basic_program(qc: QuantumComputer):
    program = Program(
        Declare("ro", "BIT", 2),
        H(0),
        CNOT(0, 1),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    ).wrap_in_numshots_loop(1000)
    compiled = qc.compile(program)
    results = qc.run(compiled)
    assert results.shape == (1000, 2)
