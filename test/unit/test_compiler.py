import math

import pytest

from pyquil import Program
from pyquil.api._compiler import QPUCompiler
from pyquil.gates import RX, MEASURE, RZ
from pyquil.quilatom import FormalArgument
from pyquil.quilbase import DefCalibration


def simple_program():
    program = Program()
    readout = program.declare("ro", "BIT", 3)
    program += RX(math.pi / 2, 0)
    program += MEASURE(0, readout[0])
    return program


# TODO: The changes to v4 are causing an extra HALT instruction to be appended to the program
@pytest.mark.skip
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
