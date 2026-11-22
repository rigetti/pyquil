import warnings

from pyquil.external.rpcq import (
    CompilerISA,
    Edge,
    GateInfo,
    Qubit,
    Supported1QGate,
    Supported2QGate,
    compiler_isa_to_target_quantum_processor,
    make_edge_id,
)


def test_qubit_dead_serialization():
    """Test that the qubit is marked dead if it has no gates"""
    gate_info = GateInfo(
        operator=Supported1QGate.RX,
        parameters=[0.0],
        arguments=[0],
        fidelity=1e0,
        duration=50,
    )
    qubit = Qubit(
        id=0,
        gates=[gate_info],
    )
    assert "dead" not in qubit.dict().keys()

    qubit = Qubit(
        id=0,
        gates=[],
        dead=True,
    )
    assert qubit.dict()["dead"] is True


def test_edge_dead_serialization():
    """Test that the edge is marked dead if it has no gates"""
    gate_info = GateInfo(
        operator=Supported2QGate.CZ,
        parameters=[],
        arguments=["_", "_"],
        fidelity=0.89,
        duration=200,
    )
    edge = Edge(
        ids=[0, 1],
        gates=[gate_info],
    )
    assert "dead" not in edge.dict().keys()

    edge = Edge(
        ids=[0, 1],
        gates=[],
        dead=True,
    )
    assert edge.dict()["dead"] is True


def test_compiler_isa_to_target_quantum_processor_does_not_warn():
    """Test that converting a CompilerISA does not warn about a pyQuil deprecation."""
    compiler_isa = CompilerISA(
        qubits={
            "0": Qubit(
                id=0,
                gates=[
                    GateInfo(
                        operator=Supported1QGate.RX,
                        parameters=[0.0],
                        arguments=[0],
                        fidelity=1e0,
                        duration=50,
                    )
                ],
            )
        },
        edges={
            make_edge_id(0, 1): Edge(
                ids=[0, 1],
                gates=[
                    GateInfo(
                        operator=Supported2QGate.CZ,
                        parameters=[],
                        arguments=["_", "_"],
                        fidelity=0.89,
                        duration=200,
                    )
                ],
            )
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        target_quantum_processor = compiler_isa_to_target_quantum_processor(compiler_isa)

    assert target_quantum_processor.isa["1Q"]["0"]["id"] == 0
    assert target_quantum_processor.isa["2Q"][make_edge_id(0, 1)]["ids"] == [0, 1]
    assert target_quantum_processor.specs == {}
