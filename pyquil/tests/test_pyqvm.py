import pytest

from pyquil import Program
from pyquil.gates import H, X, CNOT, Z, SWAP
from pyquil.pyqvm import PyQVM


@pytest.mark.parametrize(
    "first_gate,second_gate,expected",
    [
        (X(0), H(1), "X 0\nH 1\n"),
        (X(0), H(1).controlled(0), "X 0\nCONTROLLED H 0 1\n"),
        (H(0), X(1), "H 0\nX 1\n"),
        (H(0), X(1).controlled(0), "H 0\nCONTROLLED X 0 1\n"),
        (H(0), Z(1), "H 0\nZ 1\n"),
        (H(0), Z(1).controlled(0), "H 0\nCONTROLLED Z 0 1\n"),
        (X(0), X(1), "X 0\nX 1\n"),
        (X(0), X(1).controlled(0), "X 0\nCONTROLLED X 0 1\n"),
        (H(0), SWAP(0, 1), "H 0\nSWAP 0 1\n"),
        (X(0), CNOT(0, 1), "X 0\nCNOT 0 1\n"),
    ],
)
def test_pyqvm_controlled_gates(first_gate, second_gate, expected):
    """ Unit-test based on the bug report in
        https://github.com/rigetti/pyquil/issues/1259
    """
    p = Program(first_gate, second_gate)
    qvm = PyQVM(n_qubits=2)
    result = qvm.execute(p)
    assert result.program.out() == expected
