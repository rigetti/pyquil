r"""
Standard gate set, as detailed in Quil whitepaper (arXiV:1608:03355v2)

Currently includes:
    I - identity :math:`\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`

    X - Pauli-X :math:`\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`

    Y - Pauli-Y :math:`\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}`

    Z - Pauli-Z :math:`\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`

    H - Hadamard
    :math:`\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}`

    S - PHASE(pi/2)
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}`

    T - PHASE(pi/4)
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}`

    PHASE(:math:`\phi`) - PHASE
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \phi} \end{pmatrix}`

    RX(:math:`\phi`) - RX
    :math:`\begin{pmatrix} \cos(\phi / 2) & -i \sin(\phi/2) \\
                           -i \sin(\phi/2) & \cos(\phi/2) \end{pmatrix}`

    RY(:math:`\phi`) - RY
    :math:`\begin{pmatrix} \cos(\phi / 2) & -\sin(\phi / 2) \\
                           \sin(\phi/2) & \cos(\phi/2) \end{pmatrix}`

    RZ(:math:`\phi`) - RZ
    :math:`\begin{pmatrix} \cos(\phi/2) - i \sin(\phi/2) & 0 \\
                           0 & \cos(\phi/2) + i \sin(\phi/2) \end{pmatrix}`

    CZ - controlled-Z
    :math:`P_0 \otimes I + P_1 \otimes Z = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}`

    CNOT - controlled-X / controlled-NOT
    :math:`P_0 \otimes I + P_1 \otimes X = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&0&1 \\ 0&0&1&0 \end{pmatrix}`

    CCNOT - double-controlled-X
    :math:`P_0 \otimes P_0 \otimes I + P_0 \otimes P_1 \otimes I + P_1 \otimes P_0 \otimes I
                                     + P_1 \otimes P_1 \otimes X`

    CPHASE00(:math:`\phi`) - controlled-phase-on-|00>
    :math:`\text{diag}(e^{i \phi}, 1, 1, 1,)`

    CPHASE01(:math:`\phi`) - controlled-phase-on-|01>
    :math:`\text{diag}(1, e^{i \phi}, 1, 1,)`

    CPHASE10(:math:`\phi`) - controlled-phase-on-|10>
    :math:`\text{diag}(1, 1, e^{i \phi}, 1)`

    CPHASE(:math:`\phi`) - controlled-phase-on-|11>
    :math:`\text{diag}(1, 1, 1, e^{i \phi})`

    SWAP - swap
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1 \end{pmatrix}`

    CSWAP - controlled-swap
    :math:`P_0 \otimes I_2 + P_1 \otimes \text{SWAP}`

    ISWAP - i-phase-swap
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&i&0 \\ 0&i&0&0 \\ 0&0&0&1 \end{pmatrix}`

    PSWAP(:math:`\phi`) - phi-phase-swap
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&e^{i\phi}&0 \\ 0&e^{i\phi}&0&0 \\ 0&0&0&1 \end{pmatrix}`

Specialized gates / internal utility gates:
    BARENCO(:math:`\alpha, \phi, \theta`) - Barenco gate
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&e^{i\phi} \cos\theta & -i e^{i(\alpha-\phi)}
     \sin\theta \\ 0&0&-i e^{i(\alpha+\phi)} \sin\theta & e^{i\alpha} \cos\theta \end{pmatrix}`

    P0 - project-onto-zero
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`

    P1 - project-onto-one
    :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
"""
import numpy as np
import cmath

I = np.array([[1.0, 0.0], [0.0, 1.0]])

X = np.array([[0.0, 1.0], [1.0, 0.0]])

Y = np.array([[0.0, 0.0 - 1.0j], [0.0 + 1.0j, 0.0]])

Z = np.array([[1.0, 0.0], [0.0, -1.0]])

H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]])

S = np.array([[1.0, 0.0], [0.0, 1.0j]])

T = np.array([[1.0, 0.0], [0.0, cmath.exp(1.0j * np.pi / 4.0)]])


def PHASE(phi):
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])


def RX(phi):
    return np.array([[np.cos(phi / 2.0), -1j * np.sin(phi / 2.0)],
                     [-1j * np.sin(phi / 2.0), np.cos(phi / 2.0)]])


def RY(phi):
    return np.array([[np.cos(phi / 2.0), -np.sin(phi / 2.0)],
                     [np.sin(phi / 2.0), np.cos(phi / 2.0)]])


def RZ(phi):
    return np.array([[np.cos(phi / 2.0) - 1j * np.sin(phi / 2.0), 0],
                     [0, np.cos(phi / 2.0) + 1j * np.sin(phi / 2.0)]])


CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

CCNOT = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0]])


def CPHASE00(phi):
    return np.diag([np.exp(1j * phi), 1.0, 1.0, 1.0])


def CPHASE01(phi):
    return np.diag([1.0, np.exp(1j * phi), 1.0, 1.0])


def CPHASE10(phi):
    return np.diag([1.0, 1.0, np.exp(1j * phi), 1.0])


def CPHASE(phi):
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * phi)])


SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

CSWAP = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]])

ISWAP = np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]])


def PSWAP(phi):
    return np.array([[1, 0, 0, 0],
                     [0, 0, np.exp(1j * phi), 0],
                     [0, np.exp(1j * phi), 0, 0],
                     [0, 0, 0, 1]])


# Utility gates for internal QVM use
P0 = np.array([[1, 0], [0, 0]])

P1 = np.array([[0, 0], [0, 1]])


# Specialized useful gates; not officially in standard gate set
def BARENCO(alpha, phi, theta):
    lower_unitary = np.array([
        [np.exp(1j * phi) * np.cos(theta), -1j * np.exp(1j * (alpha - phi)) * np.sin(theta)],
        [-1j * np.exp(1j * (alpha + phi)) * np.sin(theta), np.exp(1j * alpha) * np.cos(theta)]])
    return np.kron(P0, np.eye(2)) + np.kron(P1, lower_unitary)


QUANTUM_GATES = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z,
    'H': H,
    'S': S,
    'T': T,
    'PHASE': PHASE,
    'RX': RX,
    'RY': RY,
    'RZ': RZ,
    'CNOT': CNOT,
    'CCNOT': CCNOT,
    'CPHASE00': CPHASE00,
    'CPHASE01': CPHASE01,
    'CPHASE10': CPHASE10,
    'CPHASE': CPHASE,
    'SWAP': SWAP,
    'CSWAP': CSWAP,
    'ISWAP': ISWAP,
    'PSWAP': PSWAP,
    'BARENCO': BARENCO,
    'CZ': CZ
}


def relaxation_operators(p):
    """
    Return the amplitude damping Kraus operators
    """
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - p)]])
    k1 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]])
    return k0, k1


def dephasing_operators(p):
    """
    Return the phase damping Kraus operators
    """
    k0 = np.eye(2) * np.sqrt(1 - p / 2)
    k1 = np.sqrt(p / 2) * Z
    return k0, k1


def depolarizing_operators(p):
    """
    Return the phase damping Kraus operators
    """
    k0 = np.sqrt(1.0 - p) * I
    k1 = np.sqrt(p / 3.0) * X
    k2 = np.sqrt(p / 3.0) * Y
    k3 = np.sqrt(p / 3.0) * Z
    return k0, k1, k2, k3


def phase_flip_operators(p):
    """
    Return the phase flip kraus operators
    """
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p) * Z
    return k0, k1


def bit_flip_operators(p):
    """
    Return the phase flip kraus operators
    """
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p) * X
    return k0, k1


def bitphase_flip_operators(p):
    """
    Return the bitphase flip kraus operators
    """
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p) * Y
    return k0, k1


KRAUS_OPS = {
    'relaxation': relaxation_operators,
    'dephasing': dephasing_operators,
    'depolarizing': depolarizing_operators,
    'phase_flip': phase_flip_operators,
    'bit_flip': bit_flip_operators,
    'bitphase_flip': bitphase_flip_operators,
}

__all__ = list(QUANTUM_GATES.keys()) + ['relaxation_operators', 'dephasing_operators',
                                        'depolarizing_operators', 'phase_flip_operators',
                                        'bit_flip_operators', 'bitphase_flip_operators']
