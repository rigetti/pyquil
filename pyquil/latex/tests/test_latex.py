import pytest

from pyquil.quil import Program, Pragma
from pyquil.quilbase import Declare, Measurement, JumpTarget, Jump
from pyquil.quilatom import MemoryReference, Label
from pyquil.gates import H, X, Y, RX, CZ, SWAP, MEASURE, CNOT, RESET, WAIT, MOVE
from pyquil.latex import to_latex, DiagramSettings
from pyquil.latex._diagram import split_on_terminal_measures


def test_to_latex():
    """A test to give full coverage of latex_generation."""
    p = Program()
    p.inst(X(0), RX(1.0, 5), Y(0), CZ(0, 2), SWAP(0, 1), MEASURE(0, None),
           CNOT(2, 0), X(0).controlled(1), Y(0).dagger())
    _ = to_latex(p)

    # Modify settings to access non-standard control paths.
    settings = DiagramSettings(impute_missing_qubits=True)
    _ = to_latex(p, settings)

    settings = DiagramSettings(abbreviate_controlled_rotations=True)
    _ = to_latex(p, settings)

    settings = DiagramSettings(label_qubit_lines=False)
    _ = to_latex(p, settings)


def test_fail_on_forked():
    """Check that to_latex raises an exception on FORKED gates."""
    p = Program()
    p.inst(RX(1.0, 0).forked(1, [2.0]))
    with pytest.raises(ValueError):
        _ = to_latex(p)


def test_gate_group_pragma():
    "Check that to_latex does not fail on LATEX_GATE_GROUP pragma."
    p = Program()
    p.inst(Pragma("LATEX_GATE_GROUP", [], 'foo'),
           X(0),
           X(0),
           Pragma("END_LATEX_GATE_GROUP"),
           X(1))
    _ = to_latex(p)


def test_fail_on_bad_pragmas():
    "Check that to_latex raises an error when pragmas are imbalanced."
    # missing END_LATEX_GATE_GROUP
    with pytest.raises(ValueError):
        _ = to_latex(Program(Pragma("LATEX_GATE_GROUP", [], 'foo'), X(0)))

    # missing LATEX_GATE_GROUP
    with pytest.raises(ValueError):
        _ = to_latex(Program(X(0), Pragma("END_LATEX_GATE_GROUP")))

    # nested groups are not currently supported
    with pytest.raises(ValueError):
        _ = to_latex(Program(Pragma("LATEX_GATE_GROUP"),
                             X(0),
                             Pragma("LATEX_GATE_GROUP"),
                             X(1),
                             Pragma("END_LATEX_GATE_GROUP"),
                             Pragma("END_LATEX_GATE_GROUP")))


def test_warn_on_pragma_with_trailing_measures():
    "Check that to_latex warns when measurement alignment conflicts with gate group pragma."
    with pytest.warns(UserWarning):
        _ = to_latex(Program(Declare('ro', 'BIT'),
                             Pragma("LATEX_GATE_GROUP"),
                             MEASURE(0, MemoryReference('ro')),
                             Pragma("END_LATEX_GATE_GROUP"),
                             MEASURE(1, MemoryReference('ro'))))


def test_split_measures():
    """Check that we can split terminal measurements."""

    prog = Program(Declare('ro', 'BIT'),
                   X(0),
                   MEASURE(0, MemoryReference('ro')),
                   X(1),
                   MEASURE(1, MemoryReference('ro')))
    meas, instr = split_on_terminal_measures(prog)
    assert len(meas) == 2
    assert len(instr) == 3
    assert all(isinstance(instr, Measurement) for instr in meas)


def test_unsupported_ops():
    target = Label("target")
    base_prog = Program(
        Declare('reg1', 'BIT'),
        Declare('reg2', 'BIT'),
        H(0),
        JumpTarget(target),
        CNOT(0, 1))

    bad_ops = [RESET(0),
               WAIT,
               Jump(target),
               MOVE(MemoryReference('reg1'), MemoryReference('reg2'))]

    assert to_latex(base_prog)

    for op in bad_ops:
        prog = base_prog + op
        with pytest.raises(ValueError):
            _ = to_latex(prog)
