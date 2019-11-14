import functools
import inspect
import random
from math import pi
from operator import mul

import numpy as np
import pytest

from pyquil.api import WavefunctionSimulator
from pyquil.gates import (CNOT, CPHASE, H, HALT, I, MEASURE, MOVE, PHASE, RESET, RX, RY, RZ, SWAP,
                          X, QUANTUM_GATES)
from pyquil.paulis import PauliTerm, exponentiate, sZ, sX, sI, sY
from pyquil.pyqvm import PyQVM
from pyquil.quil import Program
from pyquil.reference_simulator import ReferenceWavefunctionSimulator
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare

QFT_8_INSTRUCTIONS = [
    H(7),
    CPHASE(1.5707963267948966, 6, 7),
    H(6),
    CPHASE(0.7853981633974483, 5, 7),
    CPHASE(1.5707963267948966, 5, 6),
    H(5),
    CPHASE(0.39269908169872414, 4, 7),
    CPHASE(0.7853981633974483, 4, 6),
    CPHASE(1.5707963267948966, 4, 5),
    H(4),
    CPHASE(0.19634954084936207, 3, 7),
    CPHASE(0.39269908169872414, 3, 6),
    CPHASE(0.7853981633974483, 3, 5),
    CPHASE(1.5707963267948966, 3, 4),
    H(3),
    CPHASE(0.09817477042468103, 2, 7),
    CPHASE(0.19634954084936207, 2, 6),
    CPHASE(0.39269908169872414, 2, 5),
    CPHASE(0.7853981633974483, 2, 4),
    CPHASE(1.5707963267948966, 2, 3),
    H(2),
    CPHASE(0.04908738521234052, 1, 7),
    CPHASE(0.09817477042468103, 1, 6),
    CPHASE(0.19634954084936207, 1, 5),
    CPHASE(0.39269908169872414, 1, 4),
    CPHASE(0.7853981633974483, 1, 3),
    CPHASE(1.5707963267948966, 1, 2),
    H(1),
    CPHASE(0.02454369260617026, 0, 7),
    CPHASE(0.04908738521234052, 0, 6),
    CPHASE(0.09817477042468103, 0, 5),
    CPHASE(0.19634954084936207, 0, 4),
    CPHASE(0.39269908169872414, 0, 3),
    CPHASE(0.7853981633974483, 0, 2),
    CPHASE(1.5707963267948966, 0, 1),
    H(0),
    SWAP(0, 7),
    SWAP(1, 6),
    SWAP(2, 5),
    SWAP(3, 4)
]

QFT_8_WF_PROBS = [
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j, 0.0625 + 0.j,
    0.0625 + 0.j]

HADAMARD_WF = [0.70710678 + 0.j, 0.70710678 + 0.j]

ARBITRARY_STATE_GEN_INSTRUCTIONS = {
    1: [
        RZ(-1.3778211380875056, 0),
        PHASE(1.3778211380875056, 0),
        H(0),
        RY(-1.5707963267948963, 0),
        RZ(-1.3778211380875056, 0)
    ],

    2: [
        RZ(3.9156492624160952, 0),
        PHASE(-3.9156492624160952, 0),
        H(0),
        RY(-1.334414217642601, 0),
        RZ(-0.51915346851116273, 0),
    ],

    3: [
        RZ(1.1065191340906928, 0),
        PHASE(-1.1065191340906928, 0),
        H(0),
        H(1),
        RY(-0.65638478018276281, 0),
        RZ(2.1932738454924952, 0),
        SWAP(0, 1),
        CNOT(1, 0),
        RY(0.84911580533704267, 0),
        CNOT(1, 0),
        RY(-0.72168052145785411, 0),
        CNOT(1, 0),
        RZ(0.4281506526956218, 0),
        CNOT(1, 0),
        RZ(-0.65860405870618055, 0),
    ],

    4: [
        RZ(-2.9462817452102357, 0),
        PHASE(2.9462817452102357, 0),
        H(0),
        H(1),
        RY(0.31100531156953365, 0),
        RZ(2.0797152830290928, 0),
        SWAP(0, 1),
        CNOT(1, 0),
        RY(-0.36621931037037625, 0),
        CNOT(1, 0),
        RY(-0.021200280370654495, 0),
        CNOT(1, 0),
        RZ(-2.3516908466738768, 0),
        CNOT(1, 0),
        RZ(-2.9102620903531329, 0),
    ],

    5: [
        RZ(0.61206055431120343, 0),
        PHASE(-0.61206055431120343, 0),
        H(0),
        H(1),
        H(2),
        RY(-0.54253286978941762, 0),
        RZ(0.97624549655328541, 0),
        SWAP(0, 2),
        CNOT(2, 0),
        RY(0.84410519104922777, 0),
        CNOT(2, 0),
        RY(-0.72669113574566857, 0),
        CNOT(2, 0),
        RZ(-0.64542735535109075, 0),
        CNOT(2, 0),
        RZ(-1.0096122975931727, 0),
        SWAP(0, 1),
        CNOT(2, 0),
        RY(0.31406020694647047, 0),
        CNOT(1, 0),
        RY(0.39187013939133503, 0),
        CNOT(2, 0),
        RY(-0.39352802400611336, 0),
        CNOT(1, 0),
        RY(-0.47133795645097792, 0),
        CNOT(2, 0),
        RZ(-0.8454575464450087, 0),
        CNOT(1, 0),
        RZ(-0.64208935519877453, 0),
        CNOT(2, 0),
        RZ(-1.0062742974408565, 0),
        CNOT(1, 0),
        RZ(-1.2096424886870907, 0),
    ],

    6: [
        RZ(-0.31579129382350124, 0),
        PHASE(0.31579129382350124, 0),
        H(0),
        H(1),
        H(2),
        RY(-0.47578768685496686, 0),
        RZ(-1.3499849361552017, 0),
        SWAP(0, 2),
        CNOT(2, 0),
        RY(0.84249940120594313, 0),
        CNOT(2, 0),
        RY(-0.7282969255889532, 0),
        CNOT(2, 0),
        RZ(-0.90308949304029862, 0),
        CNOT(2, 0),
        RZ(0.13110414929140185, 0),
        SWAP(0, 1),
        CNOT(2, 0),
        RY(-0.14193631437317411, 0),
        CNOT(1, 0),
        RY(0.26731261870634027, 0),
        CNOT(2, 0),
        RY(0.44477400168309028, 0),
        CNOT(1, 0),
        RY(0.035525068603575904, 0),
        CNOT(2, 0),
        RZ(-0.93921566797148937, 0),
        CNOT(1, 0),
        RZ(-0.78258978415601299, 0),
        CNOT(2, 0),
        RZ(-0.17248221114867163, 0),
        CNOT(1, 0),
        RZ(-0.32910809496414811, 0),
    ],

    7: [
        RZ(-1.0234799333421403, 0),
        PHASE(1.0234799333421403, 0),
        H(0),
        H(1),
        H(2),
        RY(-0.3726424473579481, 0),
        RZ(-0.39145353578307929, 0),
        SWAP(0, 2),
        CNOT(2, 0),
        RY(0.34975748639135762, 0),
        CNOT(2, 0),
        RY(-0.13535730412329564, 0),
        CNOT(2, 0),
        RZ(-1.1443562533132183, 0),
        CNOT(2, 0),
        RZ(-0.32944991498726472, 0),
        SWAP(0, 1),
        CNOT(2, 0),
        RY(0.18825066267368085, 0),
        CNOT(1, 0),
        RY(-0.53150374263189404, 0),
        CNOT(2, 0),
        RY(0.33276302936982205, 0),
        CNOT(1, 0),
        RY(-0.51827889211949985, 0),
        CNOT(2, 0),
        RZ(0.29731251042706419, 0),
        CNOT(1, 0),
        RZ(-1.9357444168660682, 0),
        CNOT(2, 0),
        RZ(0.28776058156757334, 0),
        CNOT(1, 0),
        RZ(1.0738847729756915, 0),
    ],

    8: [
        RZ(-0.67505860032458065, 0),
        PHASE(0.67505860032458065, 0),
        H(0),
        H(1),
        H(2),
        RY(0.36905092281568769, 0),
        RZ(2.5582831108133366, 0),
        SWAP(0, 2),
        CNOT(2, 0),
        RY(-0.18585076163528808, 0),
        CNOT(2, 0),
        RY(0.0091171201901780879, 0),
        CNOT(2, 0),
        RZ(0.95713225607488983, 0),
        CNOT(2, 0),
        RZ(0.95753616404943331, 0),
        SWAP(0, 1),
        CNOT(2, 0),
        RY(-0.3965948753889344, 0),
        CNOT(1, 0),
        RY(0.044383340367875335, 0),
        CNOT(2, 0),
        RY(0.16593954296873747, 0),
        CNOT(1, 0),
        RY(-0.51082744171655792, 0),
        CNOT(2, 0),
        RZ(-0.88208839473968337, 0),
        CNOT(1, 0),
        RZ(0.72471364367814428, 0),
        CNOT(2, 0),
        RZ(0.5527632326174664, 0),
        CNOT(1, 0),
        RZ(-1.0810642125567731, 0),
    ]
}

ARBITRARY_STATE_GEN_WF = {
    1: [(0.19177970271211014 + 0.98143799887086813j),
        (7.8139472995122022e-17 + 7.562974420408338e-18j)],
    2: [(-0.12622004606236192 - 0.98496918685231072j),
        (-0.07104210675811623 - 0.094112804606394221j)],
    3: [(0.021410857148335127 - 0.59240496610553273j),
        (-0.13007475927743239 - 0.66091369808157352j),
        (0.20542985640185063 + 0.39073027284233752j),
        (1.1878933570221759e-17 - 7.1752304264728742e-18j)],
    4: [(-0.48745892028933913 + 0.03777196526465338j),
        (-0.19274238624471451 - 0.26608178286580453j),
        (-0.43676745598407479 + 0.15909039387027599j),
        (-0.40679679400276686 + 0.52100742931604693j)],
    5: [(-0.13610412961282184 + 0.41868163552681109j),
        (-0.091966738923278141 - 0.36385909080141082j),
        (0.075119803462548956 - 0.48879148186700738j),
        (-0.10638894071606081 - 0.40939557019047168j),
        (0.36699524549016321 + 0.32735468152948316j),
        (-4.1374363116509636e-17 + 8.3454943089909223e-18j),
        (5.6478205215176971e-17 + 1.8386425349669565e-19j),
        (5.6478205215176983e-17 + 1.8386425349669256e-19j)],
    6: [(-0.18164207306303881 + 0.19119468334521722j),
        (0.50052245670564666 + 0.053853837869599269j),
        (0.48796848381917918 + 0.3363658367396114j),
        (0.22428463744197349 + 0.066995250749173424j),
        (-0.021835516292086861 - 0.29652689793741388j),
        (0.38943738949226864 - 0.17582415643338745j),
        (8.0898705807672802e-17 - 1.887116732004393e-18j),
        (8.0898705807672789e-17 - 1.887116732004413e-18j)],
    7: [(-0.0053775709932576132 + 0.44935394046056309j),
        (0.067503262501559816 + 0.24891586222843232j),
        (0.01536075663774486 - 0.48334497376240482j),
        (0.038373343760319562 + 0.42218998952569198j),
        (-0.0064823038319961612 - 0.31337847560751508j),
        (0.059304958058202878 + 0.36242184150511036j),
        (0.035987291621249168 + 0.2890523483506911j),
        (-1.837382303384102e-17 - 2.0803235442497578e-17j)],
    8: [(0.0057688815507203617 - 0.39274059969851433j),
        (-0.11403364736131791 - 0.14364388388296806j),
        (-0.023056499875803743 + 0.35290693194064132j),
        (-0.0027542891415852436 - 0.081465348632389745j),
        (-0.084523401805212561 + 0.35915193082406421j),
        (0.051787727468772124 + 0.36805031655706943j),
        (-0.029805359564490708 + 0.49958851276613636j),
        (-0.012843340078957877 + 0.39426627712663159j)]
}


@pytest.fixture(params=[1, 2, 3, 4, 5, 6, 7, 8])
def arbitrary_state(request):
    return ARBITRARY_STATE_GEN_INSTRUCTIONS[request.param], ARBITRARY_STATE_GEN_WF[request.param]


def test_generate_arbitrary_states(arbitrary_state):
    prog, v = arbitrary_state
    prog = Program(prog)

    qam = PyQVM(n_qubits=8, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf

    # check actual part of wavefunction
    np.testing.assert_allclose(v, wf[:len(v)], atol=1e-10)

    # check remaining zeros part of wavefunction
    np.testing.assert_allclose(np.zeros(wf.shape[0] - len(v)), wf[len(v):])


def test_if_then():
    # if TRUE creg, then measure 0 should give 0
    prog = Program()
    creg = prog.declare('creg', 'BIT')
    prog.inst(
        MOVE(creg, 1),
        X(0)
    )
    branch_a = Program(X(0))
    branch_b = Program()
    prog.if_then(creg, branch_a, branch_b)
    prog += MEASURE(0, creg)
    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    assert qam.ram['creg'][0] == 0


def test_if_then_2():
    # if FALSE creg, then measure 0 should give 1
    prog = Program()
    creg = prog.declare('creg', 'BIT')
    prog.inst(
        MOVE(creg, 0),
        X(0)
    )
    branch_a = Program(X(0))
    branch_b = Program()
    prog.if_then(creg, branch_a, branch_b)
    prog += MEASURE(0, creg)
    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    assert qam.ram['creg'][0] == 1


def test_while():
    init_register = Program()
    classical_flag_register = init_register.declare("classical_flag_register", 'BIT')
    init_register += MOVE(classical_flag_register, True)

    loop_body = Program(X(0), H(0)).measure(0, classical_flag_register)

    # Put it all together in a loop program:
    loop_prog = init_register.while_do(classical_flag_register, loop_body)

    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(loop_prog)
    assert qam.ram[classical_flag_register.name][0] == 0


def test_halt():
    prog = Program(Declare('ro', 'BIT'), X(0), MEASURE(0, MemoryReference("ro", 0)))
    prog.inst(HALT)
    prog.inst(X(0), MEASURE(0, MemoryReference("ro", 0)))
    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    # HALT should stop execution; measure should give 1
    assert qam.ram['ro'][0] == 1

    prog = Program(Declare('ro', 'BIT'), X(0)).inst(X(0)).inst(MEASURE(0, MemoryReference("ro", 0)))
    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    assert qam.ram['ro'][0] == 0


def test_biased_coin():
    # sample from a 75% heads and 25% tails coin
    prog = Program().inst(Declare("ro", "BIT"), RX(np.pi / 3, 0)).measure(0, MemoryReference("ro", 0))

    results = []
    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    for _ in range(1000):
        qam.execute(prog)
        results += [qam.ram['ro'][0]]
        qam.execute(Program(RESET()))

    coin_bias = sum(results) / 1000
    assert np.isclose(coin_bias, 0.25, atol=0.05, rtol=0.05)


def test_against_ref_hadamard():
    p = Program(H(0))
    qam = PyQVM(n_qubits=1, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(p)
    np.testing.assert_allclose(HADAMARD_WF, qam.wf_simulator.wf)


def test_against_ref_qft_8():
    p = Program(QFT_8_INSTRUCTIONS)
    qam = PyQVM(n_qubits=8, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(p)
    wf = qam.wf_simulator.wf
    np.testing.assert_allclose(QFT_8_WF_PROBS, wf)


def test_bell_state():
    prog = Program().inst([H(0), CNOT(0, 1)])
    qam = PyQVM(n_qubits=2, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    ref_bell = np.zeros(4)
    ref_bell[0] = ref_bell[-1] = 1.0 / np.sqrt(2)
    np.testing.assert_allclose(ref_bell, wf)


def test_occupation_basis():
    prog = Program().inst([X(0), X(1), I(2), I(3)])
    state = np.zeros(2 ** 4)
    state[3] = 1.0

    qam = PyQVM(n_qubits=4, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    np.testing.assert_allclose(state, qam.wf_simulator.wf)


def test_exp_circuit():
    true_wf = np.array([0.54030231 - 0.84147098j,
                        0.00000000 + 0.j,
                        0.00000000 + 0.j,
                        0.00000000 + 0.j,
                        0.00000000 + 0.j,
                        0.00000000 + 0.j,
                        0.00000000 + 0.j,
                        0.00000000 + 0.j])

    create2kill1 = PauliTerm("X", 1, -0.25) * PauliTerm("Y", 2)
    create2kill1 += PauliTerm("Y", 1, 0.25) * PauliTerm("Y", 2)
    create2kill1 += PauliTerm("Y", 1, 0.25) * PauliTerm("X", 2)
    create2kill1 += PauliTerm("X", 1, 0.25) * PauliTerm("X", 2)
    create2kill1 += PauliTerm("I", 0, 1.0)
    prog = Program()
    for term in create2kill1.terms:
        single_exp_prog = exponentiate(term)
        prog += single_exp_prog

    qam = PyQVM(n_qubits=3, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    np.testing.assert_allclose(wf.dot(np.conj(wf).T), true_wf.dot(np.conj(true_wf).T))


def test_qaoa_circuit():
    wf_true = [0.00167784 + 1.00210180e-05 * 1j, 0.50000000 - 4.99997185e-01 * 1j,
               0.50000000 - 4.99997185e-01 * 1j, 0.00167784 + 1.00210180e-05 * 1j]
    prog = Program()
    prog.inst([RY(np.pi / 2, 0), RX(np.pi, 0),
               RY(np.pi / 2, 1), RX(np.pi, 1),
               CNOT(0, 1), RX(-np.pi / 2, 1), RY(4.71572463191, 1),
               RX(np.pi / 2, 1), CNOT(0, 1),
               RX(-2 * 2.74973750579, 0), RX(-2 * 2.74973750579, 1)])

    qam = PyQVM(n_qubits=2, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf
    np.testing.assert_allclose(wf_true, wf, atol=1e-8)


def test_larger_qaoa_circuit():
    square_qaoa_circuit = [H(0), H(1), H(2), H(3),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           CNOT(0, 1),
                           RZ(0.78564882604980579, 1),
                           CNOT(0, 1),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           CNOT(0, 3),
                           RZ(0.78564882604980579, 3),
                           CNOT(0, 3),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           CNOT(1, 2),
                           RZ(0.78564882604980579, 2),
                           CNOT(1, 2),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           X(0),
                           PHASE(0.3928244130249029, 0),
                           CNOT(2, 3),
                           RZ(0.78564882604980579, 3),
                           CNOT(2, 3),
                           H(0),
                           RZ(-0.77868204192240842, 0),
                           H(0),
                           H(1),
                           RZ(-0.77868204192240842, 1),
                           H(1),
                           H(2),
                           RZ(-0.77868204192240842, 2),
                           H(2),
                           H(3),
                           RZ(-0.77868204192240842, 3),
                           H(3)]

    prog = Program(square_qaoa_circuit)
    qam = PyQVM(n_qubits=4, quantum_simulator_type=ReferenceWavefunctionSimulator)
    qam.execute(prog)
    wf = qam.wf_simulator.wf

    wf_true = np.array([8.43771693e-05 - 0.1233845 * 1j, -1.24927731e-01 + 0.00329533 * 1j,
                        -1.24927731e-01 + 0.00329533 * 1j,
                        -2.50040954e-01 + 0.12661547 * 1j,
                        -1.24927731e-01 + 0.00329533 * 1j, -4.99915497e-01 - 0.12363516 * 1j,
                        -2.50040954e-01 + 0.12661547 * 1j, -1.24927731e-01 + 0.00329533 * 1j,
                        -1.24927731e-01 + 0.00329533 * 1j, -2.50040954e-01 + 0.12661547 * 1j,
                        -4.99915497e-01 - 0.12363516 * 1j, -1.24927731e-01 + 0.00329533 * 1j,
                        -2.50040954e-01 + 0.12661547 * 1j, -1.24927731e-01 + 0.00329533 * 1j,
                        -1.24927731e-01 + 0.00329533 * 1j,
                        8.43771693e-05 - 0.1233845 * 1j])

    np.testing.assert_allclose(wf_true, wf)


def _generate_random_program(n_qubits, length, include_measures=False):
    """Randomly sample gates and arguments (qubits, angles)"""
    if n_qubits < 3:
        raise ValueError("Please request n_qubits >= 3 so we can use 3-qubit gates.")

    gates = list(QUANTUM_GATES.values())

    prog = Program()
    if include_measures:
        gates.append(MEASURE)
        # one bit of classical memory per qubit
        prog.declare('ro', 'BIT', n_qubits)
    for _ in range(length):
        gate = random.choice(gates)
        possible_qubits = set(range(n_qubits))
        sig = inspect.signature(gate)

        param_vals = []
        for param in sig.parameters:
            if param in ['qubit', 'q1', 'q2', 'control',
                         'control1', 'control2', 'target', 'target_1', 'target_2']:
                param_val = random.choice(list(possible_qubits))
                possible_qubits.remove(param_val)
            elif param == 'classical_reg':
                qubit = random.choice(list(possible_qubits))
                param_val = MemoryReference("ro", qubit)
                possible_qubits.remove(qubit)
            elif param == 'angle':
                param_val = random.uniform(-2 * pi, 2 * pi)
            else:
                raise ValueError("Unknown gate parameter {}".format(param))

            param_vals.append(param_val)

        prog += gate(*param_vals)

    return prog


@pytest.fixture(params=list(range(3, 5)))
def n_qubits(request):
    return request.param


@pytest.fixture(params=[2, 50, 100])
def prog_length(request):
    return request.param


@pytest.fixture(params=[True, False])
def include_measures(request):
    return request.param


def test_vs_lisp_qvm(qvm, n_qubits, prog_length):
    for repeat_i in range(10):
        prog = _generate_random_program(n_qubits=n_qubits, length=prog_length)
        lisp_wf = WavefunctionSimulator()
        # force lisp wfs to allocate all qubits
        lisp_wf = lisp_wf.wavefunction(Program(I(q) for q in range(n_qubits)) + prog)
        lisp_wf = lisp_wf.amplitudes

        ref_qam = PyQVM(n_qubits=n_qubits, quantum_simulator_type=ReferenceWavefunctionSimulator)
        ref_qam.execute(prog)
        ref_wf = ref_qam.wf_simulator.wf

        np.testing.assert_allclose(lisp_wf, ref_wf, atol=1e-15)


def test_default_wf_simulator():
    qam = PyQVM(n_qubits=2)
    qam.execute(Program(H(0), H(1)))
    assert qam.wf_simulator.wf.reshape(-1).shape == (4,)


def test_expectation():
    wfn = ReferenceWavefunctionSimulator(n_qubits=3)
    val = wfn.expectation(0.4 * sZ(0) + sX(2))
    assert val == 0.4


def _generate_random_pauli(n_qubits, n_terms):
    paulis = [sI, sX, sY, sZ]
    all_op_inds = np.random.randint(len(paulis), size=(n_terms, n_qubits))
    operator = sI(0)
    for op_inds in all_op_inds:
        op = functools.reduce(mul, (paulis[pi](i) for i, pi in enumerate(op_inds)), sI(0))
        op *= np.random.uniform(-1, 1)
        operator += op
    return operator


def test_expectation_vs_lisp_qvm(qvm, n_qubits):
    for repeat_i in range(20):
        prog = _generate_random_program(n_qubits=n_qubits, length=10)
        operator = _generate_random_pauli(n_qubits=n_qubits, n_terms=5)
        lisp_wf = WavefunctionSimulator()
        lisp_exp = lisp_wf.expectation(prep_prog=prog, pauli_terms=operator)

        ref_wf = ReferenceWavefunctionSimulator(n_qubits=n_qubits).do_program(prog)
        ref_exp = ref_wf.expectation(operator=operator)
        np.testing.assert_allclose(lisp_exp, ref_exp, atol=1e-15)
