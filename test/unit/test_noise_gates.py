import numpy as np
from pyquil.gates import RZ, RX, I, CZ, ISWAP, CPHASE
from pyquil.noise_gates import _get_qvm_noise_supported_gates, THETA


def test_get_qvm_noise_supported_gates_from_compiler_isa(compiler_isa):
    gates = _get_qvm_noise_supported_gates(compiler_isa)
    for q in [0, 1, 2]:
        for g in [
            I(q),
            RX(np.pi / 2, q),
            RX(-np.pi / 2, q),
            RX(np.pi, q),
            RX(-np.pi, q),
            RZ(THETA, q),
        ]:
            assert g in gates

    assert CZ(0, 1) in gates
    assert CZ(1, 0) in gates
    assert ISWAP(1, 2) in gates
    assert ISWAP(2, 1) in gates
    assert CPHASE(THETA, 2, 0) in gates
    assert CPHASE(THETA, 0, 2) in gates


ASPEN_8_QUBITS_NO_RX = {8, 9, 10, 18, 19, 28, 29, 31}
ASPEN_8_QUBITS_NO_RZ = {8, 9, 10, 18, 19, 28, 29, 31}
ASPEN_8_EDGES_NO_CZ = {(0, 1), (10, 11), (1, 2), (21, 22), (17, 10), (12, 25)}


def test_get_qvm_noise_supported_gates_from_aspen8_isa(qcs_aspen8_quantum_processor, noise_model_dict):
    gates = _get_qvm_noise_supported_gates(qcs_aspen8_quantum_processor.to_compiler_isa())

    for q in range(len(qcs_aspen8_quantum_processor._isa.architecture.nodes)):
        if q not in ASPEN_8_QUBITS_NO_RX:
            for g in [
                RX(np.pi / 2, q),
                RX(-np.pi / 2, q),
                RX(np.pi, q),
                RX(-np.pi, q),
            ]:
                assert g in gates
        if q not in ASPEN_8_QUBITS_NO_RZ:
            assert RZ(THETA, q) in gates

    for edge in qcs_aspen8_quantum_processor._isa.architecture.edges:
        if (
            edge.node_ids[0],
            edge.node_ids[1],
        ) in ASPEN_8_EDGES_NO_CZ:
            continue
        assert CZ(edge.node_ids[0], edge.node_ids[1]) in gates
        assert CZ(edge.node_ids[1], edge.node_ids[0]) in gates
