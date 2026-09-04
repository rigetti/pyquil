import inspect
import pytest

import numpy as np
from syrupy.assertion import SnapshotAssertion

from pyquil.quil import Program
from pyquil._qpu import add_leading_delay_to_first_pulse_block


_LEADING_DELAY_TEST_CASES = [
    # Single Quil block with gates
    f"""
    DECLARE phase REAL[1]
    ADD phase {np.pi/2}
    RZ(phase) 0
    RX(pi/2) 0
    """,
    # Single Quil block with pulses
    f"""
    DECLARE phase REAL[1]
    ADD phase {np.pi/2}
    SET-PHASE 0 "rf" phase
    NONBLOCKING PULSE 0 "rf" drag_gaussian(alpha: -0.1, anh: -100000000, detuning: 0, duration: 1e-8, fwhm: 1e-8, phase: 0, scale: 0.1, t0: 1e-8)
    """,
    # Multiple blocks with gates
    f"""
    DECLARE ro BIT[1]
    DECLARE phase REAL[1]
    DECLARE index INTEGER[1]
    DECLARE condition BIT[1]
    ADD phase {np.pi/4}
    LABEL @loop
    ADD phase {np.pi/8}
    RZ(phase) 0
    RX(pi/2) 0
    ADD index 1
    LT condition index 5
    JUMP-WHEN @loop condition
    MEASURE 0 ro[0]
    """,
    # Multiple blocks with pulses
    f"""
    DECLARE ro BIT[1]
    DECLARE phase REAL[1]
    DECLARE index INTEGER[1]
    DECLARE condition BIT[1]
    ADD phase {np.pi/4}
    LABEL @loop
    ADD phase {np.pi/8}
    SET-PHASE 0 "rf" phase
    NONBLOCKING PULSE 0 "rf" drag_gaussian(alpha: -0.1, anh: -100000000, detuning: 0, duration: 1e-8, fwhm: 1e-8, phase: 0, scale: 0.1, t0: 1e-8)
    ADD index 1
    LT condition index 5
    JUMP-WHEN @loop condition
    NONBLOCKING CAPTURE 0 "ro_rx" boxcar_kernel(detuning: 0, duration: 1.6e-6, phase: 0, scale: 1) q10_unclassified[0]    
    """,
]



@pytest.mark.parametrize("quil", _LEADING_DELAY_TEST_CASES)
def test_add_leading_delay_to_first_pulse_block(quil: str, snapshot: SnapshotAssertion) -> None:
    """Test that adding a leading delay to a program conforms to the description in the function docstring."""
    program = Program(inspect.cleandoc(quil))
    add_leading_delay_to_first_pulse_block(program, leading_delay_seconds=2e-4)
    assert program.out() == snapshot(name="quil")
    
