from typing import Dict, Any
from pyquil.external.rpcq import make_edge_id
from pyquil.quantum_processor import QCSQuantumProcessor
from pyquil.quantum_processor.transformers import qcs_isa_to_compiler_isa
from pyquil.noise import NoiseModel
from pyquil.external.rpcq import CompilerISA
from qcs_api_client.models import InstructionSetArchitecture


def test_qcs_isa_to_compiler_isa(qcs_aspen8_isa: InstructionSetArchitecture, aspen8_compiler_isa: CompilerISA):
    """
    Test that ``qcs_isa_to_compiler_isa`` accurately transforms an ``InstructionSetArchitecture``
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


def test_qcs_noise_model(qcs_aspen8_isa: InstructionSetArchitecture, noise_model_dict: Dict[str, Any]):
    """
    Test that ``NoiseModel.from_dict`` initializes a ``NoiseModel``, which users may, in turn,
    pass to ``QCSQuantumProcessor`` for later initializing a noisy QVM.
    """

    noise_model = NoiseModel.from_dict(noise_model_dict)
    device = QCSQuantumProcessor("Aspen-8", qcs_aspen8_isa, noise_model=noise_model)
    assert device.quantum_processor_id == "Aspen-8"

    assert isinstance(device.noise_model, NoiseModel)
    assert device.noise_model == noise_model
