import pytest
from pyquil.gates import (CNOT, CPHASE, CPHASE00, CPHASE01, CPHASE10, CZ, H, I, ISWAP, PHASE, PSWAP,
                          RX, RY, RZ, S, SWAP, T, X, Y, Z)
from pyquil.quilbase import _strip_modifiers


@pytest.fixture(params=[I, X, Y, Z, H, S, T, ])
def oneq_gate(request):
    return request.param


@pytest.fixture(params=[PHASE, RX, RY, RZ])
def param_oneq_gate(request):
    return request.param


@pytest.fixture(params=[CZ, CNOT, SWAP, ISWAP])
def twoq_gate(request):
    return request.param


@pytest.fixture(params=[CPHASE, CPHASE00, CPHASE01, CPHASE10, PSWAP, ])
def param_twoq_gate(request):
    return request.param


def test_oneq_gate(oneq_gate):
    g = oneq_gate(234)
    assert g.out() == "{} 234".format(g.name)

    func_name = oneq_gate.__name__
    assert g.name == func_name


def test_oneq_gate_kwarg(oneq_gate):
    g = oneq_gate(qubit=234)
    assert g.out() == "{} 234".format(g.name)


def test_param_oneq_gate(param_oneq_gate):
    g = param_oneq_gate(0.2, 234)
    assert g.out() == "{}(0.2) 234".format(g.name)

    func_name = param_oneq_gate.__name__
    assert g.name == func_name


def test_param_oneq_gate_kwarg(param_oneq_gate):
    g = param_oneq_gate(angle=0.2, qubit=234)
    assert g.out() == "{}(0.2) 234".format(g.name)


def test_twoq_gate(twoq_gate):
    g = twoq_gate(234, 567)
    assert g.out() == "{} 234 567".format(g.name)

    func_name = twoq_gate.__name__
    assert g.name == func_name


def test_twoq_gate_kwarg(twoq_gate):
    func_name = twoq_gate.__name__
    if func_name.startswith('C'):
        qubits = {'control': 234, 'target': 567}
    elif 'SWAP' in func_name:
        qubits = {'q1': 234, 'q2': 567}
    else:
        raise ValueError()

    g = twoq_gate(**qubits)
    assert g.out() == "{} 234 567".format(g.name)


def test_param_twoq_gate(param_twoq_gate):
    g = param_twoq_gate(0.2, 234, 567)
    assert g.out() == "{}(0.2) 234 567".format(g.name)

    func_name = param_twoq_gate.__name__
    assert g.name == func_name


def test_param_twoq_gate_kwarg(param_twoq_gate):
    func_name = param_twoq_gate.__name__
    if func_name.startswith('C'):
        qubits = {'control': 234, 'target': 567}
    elif 'SWAP' in func_name:
        qubits = {'q1': 234, 'q2': 567}
    else:
        raise ValueError()
    g = param_twoq_gate(angle=0.2, **qubits)
    assert g.out() == "{}(0.2) 234 567".format(g.name)


def test_controlled_gate():
    g = X(0).controlled(1)
    assert g.out() == "CONTROLLED X 1 0"
    g = X(0).controlled(1).controlled(2)
    assert g.out() == "CONTROLLED CONTROLLED X 2 1 0"


def test_dagger_gate():
    g = X(0).dagger()
    assert g.out() == "DAGGER X 0"
    # This will be compiled away by quilc
    g = X(0).dagger().dagger() == "DAGGER DAGGER X 0"


def test_forked_gate():
    g = RX(0.0, 0).forked(1, [1.0])
    assert g.out() == "FORKED RX(0,1.0) 1 0"
    g = RX(0.0, 0).forked(1, [1.0]).forked(2, [2.0, 3.0])
    assert g.out() == "FORKED FORKED RX(0,1.0,2.0,3.0) 2 1 0"


def test_dagger_controlled_gate():
    g = X(0).dagger().controlled(1)
    assert g.out() == "CONTROLLED DAGGER X 1 0"
    g = X(0).controlled(1).dagger()
    assert g.out() == "DAGGER CONTROLLED X 1 0"
    g = X(0).controlled(1).dagger().controlled(2)
    assert g.out() == "CONTROLLED DAGGER CONTROLLED X 2 1 0"


def test_mixed_gate_modifiers():
    g = RX(0.1, 3) \
        .forked(2, [0.2]) \
        .controlled(1) \
        .dagger() \
        .forked(0, [0.3, 0.4])
    assert g.out() == "FORKED DAGGER CONTROLLED FORKED RX(0.1,0.2,0.3,0.4) 0 1 2 3"


def test_strip_gate_modifiers():
    g0 = RX(0.1, 3)
    g1 = RX(0.1, 3).forked(2, [0.2]).controlled(1)
    g2 = RX(0.1, 3).forked(2, [0.2]).controlled(1).dagger()

    assert _strip_modifiers(g1) == g0
    assert _strip_modifiers(g2) == g0
    assert _strip_modifiers(g2, 3) == g0
    assert _strip_modifiers(g2, 1) == g1
