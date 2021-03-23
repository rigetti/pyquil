import math

import pytest
from _pytest.monkeypatch import MonkeyPatch

from pyquil import Program
from pyquil.api import Client
from pyquil.api._compiler import QPUCompiler
from pyquil.quantum_processor import QCSQuantumProcessor
from pyquil.gates import RX, MEASURE, RZ
from pyquil.quilatom import FormalArgument
from pyquil.quilbase import DefCalibration


# TODO(andrew): tests
# QPUCompiler
# QVMCompiler


def simple_program():
    program = Program()
    readout = program.declare("ro", "BIT", 3)
    program += RX(math.pi / 2, 0)
    program += MEASURE(0, readout[0])
    return program


def test_invalid_protocol(qcs_aspen8_quantum_processor: QCSQuantumProcessor, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", "not-http-or-tcp://example.com")
    client = Client()

    with pytest.raises(
        ValueError,
        match="Expected compiler URL 'not-http-or-tcp://example.com' to start with 'tcp://'",
    ):
        QPUCompiler(
            quantum_processor_id=qcs_aspen8_quantum_processor.quantum_processor_id,
            quantum_processor=qcs_aspen8_quantum_processor,
            client=client,
        )


def test_compile_with_quilt_calibrations(compiler: QPUCompiler):
    program = simple_program()
    q = FormalArgument("q")
    defn = DefCalibration("H", [], [q], [RZ(math.pi / 2, q), RX(math.pi / 2, q), RZ(math.pi / 2, q)])
    cals = [defn]
    program._calibrations = cals
    # this should more or less pass through
    compilation_result = compiler.quil_to_native_quil(program, protoquil=True)
    assert compilation_result.calibrations == cals
    assert program.calibrations == cals
    assert compilation_result == program
