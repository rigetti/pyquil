from typing import Any, Dict

from qcs_sdk.qpu.isa import InstructionSetArchitecture

from pyquil.external.rpcq import CompilerISA, make_edge_id
from pyquil.quantum_processor import QCSQuantumProcessor
from pyquil.quantum_processor.transformers import qcs_isa_to_compiler_isa


def test_qcs_isa_to_compiler_isa(qcs_aspen8_isa: InstructionSetArchitecture, aspen8_compiler_isa: CompilerISA):
    """Test that ``qcs_isa_to_compiler_isa`` accurately transforms an ``InstructionSetArchitecture``
    to a ``CompilerISA``. The qubits and edges in the former should all be present in the latter.

    Note, this is a data driven test using fixtures defined in conftest.py. This
    comprehensively checks that ``qcs_isa_to_compiler_isa`` transforms all operators and fidelities
    accurately.
    """
    compiler_isa = qcs_isa_to_compiler_isa(qcs_aspen8_isa)

    for node in qcs_aspen8_isa.architecture.nodes:
        assert str(node.node_id) in compiler_isa.qubits

    for edge in qcs_aspen8_isa.architecture.edges:
        assert make_edge_id(edge.node_ids[0], edge.node_ids[1]) in compiler_isa.edges

    assert compiler_isa == aspen8_compiler_isa
