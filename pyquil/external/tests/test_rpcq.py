from pyquil.external.rpcq import Qubit, Edge, GateInfo, Supported1QGate, Supported2QGate


def test_qubit_dead_serialization():
    "test that the qubit is marked dead if it has no gates"
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
    assert qubit.dict()['dead'] is True


def test_edge_dead_serialization():
    "test that the edge is marked dead if it has no gates"
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
    assert edge.dict()['dead'] is True
