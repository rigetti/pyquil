import math
from typing import Optional

import pytest
from qcs_sdk.qpu.translation import TranslationBackend
from syrupy.assertion import SnapshotAssertion

from pyquil import Program
from pyquil.api._compiler import (
    IncompatibleBackendForQuantumProcessorIDWarning,
    QPUCompiler,
    select_backend_for_quantum_processor_id,
)
from pyquil.gates import MEASURE, RX, RZ
from pyquil.quilatom import FormalArgument
from pyquil.quilbase import DefCalibration


def simple_program():
    program = Program()
    readout = program.declare("ro", "BIT", 3)
    program += MEASURE(0, readout[0])
    return program


def test_compile_with_quilt_calibrations(compiler: QPUCompiler):
    program = simple_program()
    q = FormalArgument("q")
    defn = DefCalibration("H", [], [q], [RZ(math.pi / 2, q), RX(math.pi / 2, q), RZ(math.pi / 2, q)])
    cals = [defn]
    program.inst(cals)
    # this should more or less pass through
    compilation_result = compiler.quil_to_native_quil(program, protoquil=True)
    assert compilation_result.calibrations == cals
    assert program.calibrations == cals
    assert compilation_result == program


def test_transpile_qasm_2(compiler: QPUCompiler, snapshot: SnapshotAssertion):
    qasm = 'OPENQASM 2.0;\nqreg q[3];\ncreg ro[2];\nmeasure q[0] -> ro[0];\nmeasure q[1] -> ro[1];'
    program = compiler.transpile_qasm_2(qasm)
    assert program.out() == snapshot


@pytest.mark.parametrize(
    "quantum_processor_id,backend,expected,warns",
    [
        ("Aspen-M-3", None, TranslationBackend.V1, False),
        ("Aspen-M-3", TranslationBackend.V1, TranslationBackend.V1, False),
        ("Aspen-M-3", TranslationBackend.V2, TranslationBackend.V1, True),
        ("Not-Aspen", None, TranslationBackend.V2, False),
        ("Not-Aspen", TranslationBackend.V1, TranslationBackend.V2, True),
        ("Not-Aspen", TranslationBackend.V2, TranslationBackend.V2, False),
    ]
)
def test_translation_backend_validation(quantum_processor_id: str, backend: Optional[TranslationBackend], expected: TranslationBackend, warns: bool):
    if warns:
        with pytest.warns(IncompatibleBackendForQuantumProcessorIDWarning):
            actual = select_backend_for_quantum_processor_id(quantum_processor_id, backend)
    else:
        actual = select_backend_for_quantum_processor_id(quantum_processor_id, backend)
    assert actual == expected

