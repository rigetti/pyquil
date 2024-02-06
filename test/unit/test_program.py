import numpy as np
from syrupy.assertion import SnapshotAssertion

from pyquil import Program
from pyquil.quil import AbstractInstruction, Declare, Measurement
from pyquil.quilbase import Reset, ResetQubit, Delay, DelayFrames, DelayQubits, Frame, \
    DefMeasureCalibration
from pyquil.quilatom import Label, MemoryReference, Qubit
from pyquil.experiment._program import (
    measure_qubits,
    parameterized_single_qubit_measurement_basis,
    parameterized_single_qubit_state_preparation,
    parameterized_readout_symmetrization,
)
from pyquil.gates import MEASURE, RX, RZ, H, DELAY


def test_measure_qubits():
    p = Program()
    ro = p.declare("ro", "BIT", 2)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    assert measure_qubits([0, 1]).out() == p.out()


def test_parameterized_single_qubit_measurement_basis():
    p = Program()
    alpha = p.declare("measurement_alpha", "REAL", 2)
    beta = p.declare("measurement_beta", "REAL", 2)
    gamma = p.declare("measurement_gamma", "REAL", 2)
    for idx, q in enumerate(range(2)):
        p += RZ(alpha[idx], q)
        p += RX(np.pi / 2, q)
        p += RZ(beta[idx], q)
        p += RX(-np.pi / 2, q)
        p += RZ(gamma[idx], q)
    assert parameterized_single_qubit_measurement_basis([0, 1]).out() == p.out()


def test_parameterized_single_qubit_state_preparation():
    p = Program()
    alpha = p.declare("preparation_alpha", "REAL", 2)
    beta = p.declare("preparation_beta", "REAL", 2)
    gamma = p.declare("preparation_gamma", "REAL", 2)
    p += RZ(alpha[0], 0)
    p += RX(np.pi / 2, 0)
    p += RZ(beta[0], 0)
    p += RX(-np.pi / 2, 0)
    p += RZ(gamma[0], 0)
    p += RZ(alpha[1], 1)
    p += RX(np.pi / 2, 1)
    p += RZ(beta[1], 1)
    p += RX(-np.pi / 2, 1)
    p += RZ(gamma[1], 1)
    assert parameterized_single_qubit_state_preparation([0, 1]).out() == p.out()


def test_parameterized_readout_symmetrization():
    p = Program()
    symmetrization = p.declare("symmetrization", "REAL", 2)
    p += RX(symmetrization[0], 0)
    p += RX(symmetrization[1], 1)
    assert parameterized_readout_symmetrization([0, 1]).out() == p.out()


def test_adding_does_not_mutate():
    # https://github.com/rigetti/pyquil/issues/1476
    p1 = Program(
        """
DEFCAL RX(pi/2) 32:
    FENCE 32
    NONBLOCKING PULSE 32 "rf" drag_gaussian(duration: 3.2e-08, fwhm: 8e-09, t0: 1.6e-08, anh: -190000000.0, alpha: -1.8848698349348032, scale: 0.30631340170943533, phase: 0.0, detuning: 1622438.2425563578)
    FENCE 32

RX(pi/2) 32
"""
    )
    original_p1 = p1.copy()
    p2 = Program(
        """
DEFCAL RX(pi/2) 33:
    FENCE 33
    NONBLOCKING PULSE 33 "rf" drag_gaussian(duration: 2e-08, fwhm: 5e-09, t0: 1e-08, anh: -190000000.0, alpha: -0.9473497322033984, scale: 0.25680107985232403, phase: 0.0, detuning: 1322130.5458282642)
    FENCE 33

RX(pi/2) 33
"""
    )
    p_all = p1 + p2
    assert str(p1) == str(original_p1)
    assert p1.calibrations != p_all.calibrations


def test_with_loop(snapshot: SnapshotAssertion):
    p = Program(
        """DECLARE ro BIT
DECLARE shot_count INTEGER
MEASURE q ro
JUMP-UNLESS @end-reset ro
X q
LABEL @end-reset

DEFCAL I 0:
    DELAY 0 1.0
DEFFRAME 0 \"rx\":
    HARDWARE-OBJECT: \"hardware\"
DEFWAVEFORM custom:
    1,2
I 0
"""
    )
    p_copy = p.copy()

    looped = p.with_loop(100, MemoryReference("shot_count"), Label("start-loop"), Label("end-loop"))

    assert p_copy == p
    assert looped.out() == snapshot


def test_filter_program():
    program = Program(Declare("ro", "BIT", 1), MEASURE(0, MemoryReference("ro", 1)), H(0))

    def predicate(instruction: AbstractInstruction) -> bool:
        if isinstance(instruction, Declare):
            return instruction.name != "ro"
        elif isinstance(instruction, Measurement):
            return instruction.classical_reg.name != "ro"
        else:
            return True

    filtered_program = program.filter_instructions(predicate)
    assert filtered_program == Program(H(0))


def test_filter_quil_t():
    non_quil_t_program = Program(
        """DECLARE ro BIT[1]
H 0
CNOT 0 1
MEASURE 0 ro[0]
MEASURE 1 ro[0]
WAIT
"""
    )

    quil_t_program = Program(
        """
DEFCAL I q:
    DELAY q 4e-08
DEFFRAME 0 "rf":
    DIRECTION: "tx"
DEFCAL MEASURE 0 addr:
    FENCE 0
DEFWAVEFORM q44_q45_cphase/sqrtCPHASE:
    0.0, 0.0, 0.00027685415721916584
CAPTURE 10 "ro_rx" boxcar_kernel(duration: 1.6e-06, scale: 1.0, phase: 0.0, detuning: 0.0) q10_unclassified[0]
NONBLOCKING CAPTURE 10 "ro_rx" boxcar_kernel(duration: 1.6e-06, scale: 1.0, phase: 0.0, detuning: 0.0) q10_unclassified[0]
FENCE 0
PULSE 0 "rf_f12" gaussian(duration: 6.000000000000001e-08, fwhm: 1.5000000000000002e-08, t0: 3.0000000000000004e-08, scale: 0.16297407445283926, phase: 0.0, detuning: 0)
RAW-CAPTURE 0 "out" 200000000 iqs[0]
SET-FREQUENCY 0 "xy" 5400000000
SET-PHASE 0 "xy" pi
SET-SCALE 0 "xy" pi
SHIFT-FREQUENCY 0 "ro" 6100000000
SHIFT-PHASE 0 "xy" (-pi)
SHIFT-PHASE 0 "xy" (%theta*(2/pi))
SWAP-PHASES 2 3 "xy" 3 4 "xy";
"""
    )
    full_program = non_quil_t_program + quil_t_program
    assert full_program.remove_quil_t_instructions() == non_quil_t_program


def test_compatibility_layer():
    """
    Test that the compatibility layer that transforms pyQuil instructions to quil instructions works as intended.
    """
    # Note: `quil` re-orders some instructions in a program (e.g. by shuffling DECLAREs to the top). This isn't a
    # problem for the current set of instructions we're testing, but it's something to keep in mind if we add more.
    instructions = [
        ResetQubit(0),
        Reset(),
        Delay([], [], 0.01),
        DelayFrames([Frame([0], "frame")], 0.01),
        DelayQubits([0, 1], 0.01),
    ]
    program = Program(instructions)
    for (original, transformed) in zip(instructions, program):
        assert isinstance(transformed, AbstractInstruction)
        assert isinstance(transformed, type(original))
        assert transformed == original

    defmeascal = DefMeasureCalibration(
        Qubit(0),
        MemoryReference("foo"),
        instructions
    )
    for original, transformed in zip(instructions, defmeascal.instrs):
        assert isinstance(transformed, AbstractInstruction)
        assert isinstance(transformed, type(original))
        assert transformed == original


def test_delay_conversions():
    f = Frame([Qubit(0)], "hi")

    i = DELAY(f, 7)
    assert isinstance(i, DelayFrames)
    p = Program()
    p += i
    p += DelayFrames([f], 8)
    for x in p:
        assert isinstance(x, DelayFrames), type(x)
