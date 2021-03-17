from pyquil.quilbase import Gate
from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.external.rpcq import CompilerISA, Supported1QGate, Supported2QGate, GateInfo, Edge
from typing import List, Optional
import logging

_log = logging.getLogger(__name__)
THETA = Parameter("theta")


def _get_qvm_noise_supported_gates(isa: CompilerISA) -> List[Gate]:
    """
    Generate the gate set associated with an ISA for which QVM noise is supported.

    :param isa: The instruction set architecture for a QPU.
    :return: A list of Gate objects encapsulating all gates compatible with the ISA.
    """
    gates = []
    for _qubit_id, q in isa.qubits.items():
        if q.dead:
            continue
        for gate in q.gates:
            if gate.operator == Supported1QGate.MEASURE:
                continue

            assert isinstance(gate, GateInfo)
            qvm_noise_supported_gate = _transform_rpcq_qubit_gate_info_to_qvm_noise_supported_gate(
                qubit_id=q.id,
                gate=gate,
            )
            if qvm_noise_supported_gate is not None:
                gates.append(qvm_noise_supported_gate)

    for _edge_id, edge in isa.edges.items():
        if edge.dead:
            continue

        qvm_noise_supported_gates = _transform_rpcq_edge_gate_info_to_qvm_noise_supported_gates(edge)
        gates.extend(qvm_noise_supported_gates)

    return gates


def _transform_rpcq_qubit_gate_info_to_qvm_noise_supported_gate(qubit_id: int, gate: GateInfo) -> Optional[Gate]:
    if gate.operator == Supported1QGate.RX:
        if len(gate.parameters) == 1 and gate.parameters[0] == 0.0:
            return None

        parameters = [Parameter(param) if isinstance(param, str) else param for param in gate.parameters]
        return Gate(gate.operator, parameters, [unpack_qubit(qubit_id)])

    if gate.operator == Supported1QGate.RZ:
        return Gate(Supported1QGate.RZ, [Parameter("theta")], [unpack_qubit(qubit_id)])

    if gate.operator == Supported1QGate.I:
        return Gate(Supported1QGate.I, [], [unpack_qubit(qubit_id)])

    _log.warning("Unknown qubit gate operator: {}".format(gate.operator))
    return None


def _transform_rpcq_edge_gate_info_to_qvm_noise_supported_gates(edge: Edge) -> List[Gate]:
    operators = [gate.operator for gate in edge.gates]
    targets = [unpack_qubit(t) for t in edge.ids]

    gates = []
    if Supported2QGate.CZ in operators:
        gates.append(Gate("CZ", [], targets))
        gates.append(Gate("CZ", [], targets[::-1]))
        return gates

    if Supported2QGate.ISWAP in operators:
        gates.append(Gate("ISWAP", [], targets))
        gates.append(Gate("ISWAP", [], targets[::-1]))
        return gates

    if Supported2QGate.CPHASE in operators:
        gates.append(Gate("CPHASE", [THETA], targets))
        gates.append(Gate("CPHASE", [THETA], targets[::-1]))
        return gates

    if Supported2QGate.XY in operators:
        gates.append(Gate("XY", [THETA], targets))
        gates.append(Gate("XY", [THETA], targets[::-1]))
        return gates

    if Supported2QGate.WILDCARD in operators:
        gates.append(Gate("_", "_", targets))
        gates.append(Gate("_", "_", targets[::-1]))
        return gates

    _log.warning(f"no gate for edge {edge.ids}")
    return gates
