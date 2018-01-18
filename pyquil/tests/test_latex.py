from pyquil.quil import Program
from pyquil.gates import X, Y, CZ, SWAP, MEASURE, CNOT
from pyquil.latex import to_latex
from pyquil.latex.latex_config import get_default_settings


def test_to_latex():
    """A test to give full coverage of latex_generation and latex_config."""
    qubits = range(3)
    p = Program()
    p.inst(X(qubits[0]), Y(qubits[0]), CZ(qubits[0], qubits[2]), SWAP(qubits[0], qubits[1]), MEASURE(qubits[0]),
           CNOT(qubits[2], qubits[0]))
    _ = to_latex(p)

    # Modify settings to access non-standard control paths.
    settings = get_default_settings()
    settings['gates']['AllocateQubitGate']['draw_id'] = True
    settings['gate_shadow'] = None
    _ = to_latex(p, settings)

    settings['control']['shadow'] = True
    _ = to_latex(p, settings)
