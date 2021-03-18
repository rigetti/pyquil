from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from pyquil.external.rpcq import CompilerISA
from pyquil.device.transformers import qcs_isa_to_compiler_isa, qcs_isa_to_graph
from pyquil.device import AbstractDevice
from pyquil.noise import NoiseModel
from pyquil.api import Client
import networkx as nx
from typing import List, Optional


class QCSDevice(AbstractDevice):
    """
    An AbstractDevice initialized with an ``InstructionSetArchitecture`` returned
    from the QCS API. Notably, this class is able to serialize a ``CompilerISA`` based
    on the architecture instructions.
    """

    quantum_processor_id: str
    _isa: InstructionSetArchitecture
    noise_model: Optional[NoiseModel]

    def __init__(
        self,
        quantum_processor_id: str,
        isa: InstructionSetArchitecture,
        noise_model: Optional[NoiseModel] = None,
    ):
        """
        Initialize a new QCSDevice.

        :param quantum_processor_id: The id of the quantum processor.
        :param isa: The QCS API ``InstructionSetArchitecture``.
        :param noise_model: An optional ``NoiseModel`` for configuring a noisy device
        on the ``QVM``.
        """

        self.quantum_processor_id = quantum_processor_id
        self._isa = isa
        self.noise_model = noise_model

    def qubits(self) -> List[int]:
        return sorted(node.node_id for node in self._isa.architecture.nodes)

    def qubit_topology(self) -> nx.Graph:
        return qcs_isa_to_graph(self._isa)

    def to_compiler_isa(self) -> CompilerISA:
        return qcs_isa_to_compiler_isa(self._isa)

    def __str__(self) -> str:
        return "<QCSDevice {}>".format(self.quantum_processor_id)

    def __repr__(self) -> str:
        return str(self)


def get_qcs_device(client: Client, quantum_processor_id: str) -> QCSDevice:
    isa = client.qcs_request(
        get_instruction_set_architecture, quantum_processor_id=quantum_processor_id
    )

    return QCSDevice(quantum_processor_id=quantum_processor_id, isa=isa)
