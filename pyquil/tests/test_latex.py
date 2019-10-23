from pyquil.quil import Program, Pragma
from pyquil.gates import X, Y, RX, CZ, SWAP, MEASURE, CNOT
from pyquil.latex import to_latex, DiagramSettings

import pytest


def test_to_latex():
    """A test to give full coverage of latex_generation."""
    p = Program()
    p.inst(X(0), RX(1.0, 5), Y(0), CZ(0,2), SWAP(0,1), MEASURE(0, None),
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
    p.inst(Pragma("LATEX_GATE_GROUP", ['foo']),
           X(0),
           X(0),
           Pragma("END_LATEX_GATE_GROUP"),
           X(1))
    _ = to_latex(p)


def test_fail_on_bad_pragmas():
    "Check that to_latex raises an error when pragmas are imbalanced."

    # missing END_LATEX_GATE_GROUP
    with pytest.raises(ValueError):
        _ = to_latex(Program(Pragma("LATEX_GATE_GROUP", ['foo']), X(0)))

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
