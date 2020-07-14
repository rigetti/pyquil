import numpy as np

from pyquil.quil import Program
from pyquil.quilatom import (
    MemoryReference,
    Parameter,
    FormalArgument,
    Frame,
    WaveformReference,
)
from pyquil.quiltwaveforms import (
    FlatWaveform,
    GaussianWaveform,
    DragGaussianWaveform,
    ErfSquareWaveform,
    HrmGaussianWaveform,
    BoxcarAveragerKernel,
)
from pyquil.quiltcalibrations import (
    CalibrationMatch,
    fill_placeholders,
    match_calibration,
    expand_calibration,
)
from pyquil.quilbase import (
    Gate,
    DefCalibration,
    Capture,
    ShiftPhase,
    Pulse,
    DelayQubits,
    Qubit,
)


def test_waveform_samples():
    # this is a very naive check: can we sample from the built-in template
    # waveforms?
    duration = 1e-6
    waveforms = [
        FlatWaveform(duration=duration, iq=1.0),
        FlatWaveform(duration=duration, iq=1.0 + 2.0j),
        GaussianWaveform(duration=duration, fwhm=2.0, t0=1.0),
        DragGaussianWaveform(
            duration=duration, fwhm=duration / 4, t0=duration / 2, anh=5.0, alpha=3.0
        ),
        HrmGaussianWaveform(
            duration=duration,
            fwhm=duration / 4,
            t0=duration / 2,
            anh=5.0,
            alpha=3.0,
            second_order_hrm_coeff=0.5,
        ),
        ErfSquareWaveform(duration=duration, risetime=duration / 8, pad_left=0.0, pad_right=0.0),
        BoxcarAveragerKernel(duration=duration),
    ]

    rates = [int(1e9), 1e9, 1e9 + 0.5]

    for rate in rates:
        for wf in waveforms:
            assert wf.samples(rate) is not None


def test_waveform_samples_optional_args():
    def flat(**kwargs):
        return FlatWaveform(duration=1e-8, iq=1.0, **kwargs).samples(1e9)

    assert np.array_equal(2.0 * flat(), flat(scale=2.0))
    assert np.array_equal(np.exp(1j) * flat(), flat(phase=1.0))


def _match(cal, instr):
    # get the first line, remove trailing colon if present
    cal_header = cal.splitlines()[0].strip().replace(":", "")
    # convert to quil ast
    full_calibration = cal_header + f":\n    NOP\n\n"
    cal = Program(full_calibration).calibrations[0]

    # we pull the last instr (omitting implicit DECLARE)
    instr = Program(instr)[-1]
    return match_calibration(instr, cal)


def test_simple_gate_calibration_match():
    matches = [
        ("DEFCAL X 0", "X 0"),
        ("DEFCAL X q", "X 0"),
        ("DEFCAL FOO q 0", "FOO 1 0"),
        ("DEFCAL FOO q 0", "FOO 0 0"),
    ]

    for cal, instr in matches:
        assert _match(cal, instr) is not None

    assert _match(cal, instr).settings[FormalArgument("q")] == Qubit(0)

    mismatches = [
        ("DEFCAL X 0", "X 1"),
        ("DEFCAL Y 0", "X 0"),
        # we are case sensitive
        ("DEFCAL foo 0", "FOO 0"),
    ]

    for cal, instr in mismatches:
        assert _match(cal, instr) is None


def test_parametric_calibration_match():
    matches = [
        ("DEFCAL RX(0.0) 0", "RX(0.0) 0"),
        ("DEFCAL RX(0.0) 0", "RX(0*pi) 0"),
        ("DEFCAL RX(pi/2) 0", "RX(pi/2) 0"),
        ("DEFCAL RX(pi/2) q", "RX(pi/2) 0"),
        ("DEFCAL RX(pi/2) q", "RX(pi/2) 1"),
        ("DEFCAL RZ(pi/2) 0", "RZ(pi/2) 0"),
        ("DEFCAL RX(%theta) 0", "RX(pi) 0"),
    ]

    for cal, instr in matches:
        assert _match(cal, instr) is not None

    assert np.isclose(_match(cal, instr).settings[Parameter("theta")], np.pi)

    mismatches = [
        ("DEFCAL RX(pi) q", "RX(0) 0"),
        ("DEFCAL RX(pi) 0", "RX(0) 0"),
        ("DEFCAL RX(pi) 0", "RY(pi) 0"),
    ]

    for cal, instr in mismatches:
        assert not _match(cal, instr)


def test_memory_reference_parameter():
    assert _match("DEFCAL RX(%theta) q", "DECLARE theta REAL\nRZ(theta) 0") is None


def test_measure_calibration_match():
    matches = [
        ("DEFCAL MEASURE 0", "MEASURE 0"),
        ("DEFCAL MEASURE q", "MEASURE 0"),
        ("DEFCAL MEASURE 0 b", "DECLARE ro BIT\nMEASURE 0 ro[0]"),
        ("DEFCAL MEASURE q b", "DECLARE ro BIT\nMEASURE 1 ro[0]"),
    ]

    for cal, instr in matches:
        assert _match(cal, instr) is not None

    assert _match(cal, instr).settings[FormalArgument("q")] == Qubit(1)
    assert _match(cal, instr).settings[FormalArgument("b")] == MemoryReference("ro")

    mismatches = [
        ("DEFCAL MEASURE 1", "MEASURE 0"),
        ("DEFCAL MEASURE 0 b", "MEASURE 0"),
        ("DEFCAL MEASURE q", "DECLARE ro BIT\nMEASURE 0 ro[0]"),
    ]

    for cal, instr in mismatches:
        assert _match(cal, instr) is None


def test_apply_match_shift_phase():
    settings = {FormalArgument("q"): Qubit(0), Parameter("theta"): np.pi}

    instr = ShiftPhase(Frame([FormalArgument("q")], "ff"), Parameter("theta") / (2.0 * np.pi))

    actual = fill_placeholders(instr, settings)

    expected = ShiftPhase(Frame([Qubit(0)], "ff"), 0.5)

    assert actual == expected


def test_apply_match_delay_qubits():
    settings = {FormalArgument("q"): Qubit(0), Parameter("foo"): 1.0}

    instr = DelayQubits([Qubit(1), FormalArgument("q")], duration=Parameter("foo"))

    actual = fill_placeholders(instr, settings)

    expected = DelayQubits([Qubit(1), Qubit(0)], 1.0)

    assert actual == expected


def test_program_match_last():
    first = DefCalibration("X", [], [Qubit(0)], ["foo"])
    second = DefCalibration("X", [], [Qubit(0)], ["bar"])
    prog = Program(first, second)
    match = prog.match_calibrations(Gate("X", [], [Qubit(0)]))
    assert match == CalibrationMatch(cal=second, settings={})


def test_program_calibrate():
    prog = Program('DEFCAL RZ(%theta) q:\n    SHIFT-PHASE q "rf" -%theta')
    calibrated = prog.calibrate(Gate("RZ", [np.pi], [Qubit(0)]))
    assert calibrated == Program('SHIFT-PHASE 0 "rf" -pi').instructions
