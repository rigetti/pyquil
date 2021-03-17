from pyquil.magic import (
    CNOT,
    H,
    I,
    MEASURE,
    X,
    Program,
    magicquil,
)


@magicquil
def bell_state(q1, q2):
    H(q1)
    CNOT(q1, q2)


def test_bell_state():
    assert bell_state(0, 1) == Program("H 0\nCNOT 0 1")


@magicquil
def fast_reset(q1):
    reg1 = MEASURE(q1)
    if reg1:
        X(q1)
    else:
        I(q1)


def test_fast_reset():
    assert fast_reset(0) == Program("DECLARE ro BIT\nMEASURE 0 ro[0]").if_then(
        ("ro", 0), Program("X 0"), Program("I 0")
    )


@magicquil
def no_else(q1):
    reg1 = MEASURE(q1)
    if reg1:
        X(q1)


def test_no_else():
    assert no_else(0) == Program("DECLARE ro BIT\nMEASURE 0 ro[0]").if_then(("ro", 0), Program("X 0"))


@magicquil
def with_elif(q1, q2):
    reg1 = MEASURE(q1)
    reg2 = MEASURE(q2)
    if reg1:
        X(q1)
    elif reg2:
        X(q2)


def test_with_elif():
    assert with_elif(0, 1) == Program("DECLARE ro BIT[2]\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]").if_then(
        ("ro", 0), Program("X 0"), Program().if_then(("ro", 1), Program("X 1"))
    )


@magicquil
def calls_another(q1, q2, q3):
    bell_state(q1, q2)
    CNOT(q2, q3)


def test_calls_another():
    assert calls_another(0, 1, 2) == Program("H 0\nCNOT 0 1\nCNOT 1 2")


@magicquil
def still_works_with_bools():
    if 1 + 1 == 2:
        H(0)
    else:
        H(1)


def test_stills_works_with_bools():
    assert still_works_with_bools() == Program("H 0")
