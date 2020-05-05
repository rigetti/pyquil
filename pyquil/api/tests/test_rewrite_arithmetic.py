from pyquil.quil import Program
from pyquil.api._rewrite_arithmetic import rewrite_arithmetic
from rpcq.messages import (
    ParameterAref,
    ParameterSpec,
    RewriteArithmeticResponse,
)


def test_rewrite_arithmetic_no_params():
    prog = Program("X 0")
    response = rewrite_arithmetic(prog)
    assert response == RewriteArithmeticResponse(quil=Program("X 0").out())


def test_rewrite_arithmetic_simple_mref():
    prog = Program("DECLARE theta REAL", "RZ(theta) 0")
    response = rewrite_arithmetic(prog)
    assert response == RewriteArithmeticResponse(
        original_memory_descriptors={"theta": ParameterSpec(length=1, type="REAL")},
        quil=Program("DECLARE theta REAL[1]", "RZ(theta[0]) 0").out(),
        recalculation_table={},
    )


def test_rewrite_arithmetic_duplicate_exprs():
    prog = Program(
        "DECLARE theta REAL",
        "RZ(theta*1.5) 0",
        "RX(theta*1.5) 0",  # this is not a native gate, but it is a protoquil program
    )

    response = rewrite_arithmetic(prog)

    assert response == RewriteArithmeticResponse(
        original_memory_descriptors={"theta": ParameterSpec(length=1, type="REAL")},
        recalculation_table={ParameterAref(index=0, name="__P1"): "theta[0]*1.5"},
        quil=Program(
            "DECLARE __P1 REAL[1]", "DECLARE theta REAL[1]", "RZ(__P1[0]) 0", "RX(__P1[0]) 0"
        ).out(),
    )


def test_rewrite_arithmetic_mixed():
    prog = Program(
        "DECLARE theta REAL", "DECLARE beta REAL", "RZ(3 * theta) 0", "RZ(beta+theta) 0",
    )
    response = rewrite_arithmetic(prog)
    assert response.original_memory_descriptors == {
        "theta": ParameterSpec(length=1, type="REAL"),
        "beta": ParameterSpec(length=1, type="REAL"),
    }
    assert response.recalculation_table == {
        ParameterAref(index=0, name="__P2"): "3*theta[0]",
        ParameterAref(index=1, name="__P2"): "beta[0] + theta[0]",
    }
    assert (
        response.quil
        == Program(
            "DECLARE __P2 REAL[2]",
            "DECLARE theta REAL[1]",
            "DECLARE beta REAL[1]",
            "RZ(__P2[0]) 0",
            "RZ(__P2[1]) 0",
        ).out()
    )
