"""An implementation of AbstractQuantumProcessor based on an InstructionSetArchitecture returned from the QCS API."""

from typing import Optional

import networkx as nx
from qcs_sdk import QCSClient
from qcs_sdk.qpu.isa import InstructionSetArchitecture, get_instruction_set_architecture

from pyquil.external.rpcq import CompilerISA
from pyquil.noise import NoiseModel
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quantum_processor.transformers import qcs_isa_to_compiler_isa, qcs_isa_to_graph


class QCSQuantumProcessor(AbstractQuantumProcessor):
    """An AbstractQuantumProcessor initialized with an ``InstructionSetArchitecture`` returned from the QCS API.

    Notably, this class is able to serialize a ``CompilerISA`` based on the architecture instructions.
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
        """Initialize a new QCSQuantumProcessor.

        :param quantum_processor_id: The id of the quantum processor.
        :param isa: The QCS API ``InstructionSetArchitecture``.
        :param noise_model: An optional ``NoiseModel`` for configuring a noisy quantum_processor on the ``QVM``.
        """
        self.quantum_processor_id = quantum_processor_id
        self._isa = isa
        self.noise_model = noise_model

    def qubits(self) -> list[int]:
        """Return the qubits in the quantum_processor topology."""
        return sorted(node.node_id for node in self._isa.architecture.nodes)

    def qubit_topology(self) -> nx.Graph:
        """Return the qubit topology as a NetworkX graph."""
        return qcs_isa_to_graph(self._isa)

    def to_compiler_isa(self) -> CompilerISA:
        """Return a CompilerISA representation of the quantum_processor's InstructionSetArchitecture."""
        return qcs_isa_to_compiler_isa(self._isa)

    def __str__(self) -> str:
        """Return a string representation of the quantum_processor."""
        return f"<QCSQuantumProcessor {self.quantum_processor_id}>"

    def __repr__(self) -> str:
        """Return a string representation of the quantum_processor."""
        return str(self)


def get_qcs_quantum_processor(
    quantum_processor_id: str,
    client_configuration: Optional[QCSClient] = None,
    timeout: float = 10.0,
) -> QCSQuantumProcessor:
    """Retrieve an instruction set architecture for the specified QPU ID and initialize a ``QCSQuantumProcessor`` with it.

    :param quantum_processor_id: QCS ID for the quantum processor.
    :param timeout: Time limit for request, in seconds.
    :param client_configuration: Optional client configuration. If none is provided, a default one will
           be loaded.

    :return: A ``QCSQuantumProcessor`` with the requested ISA.
    """
    isa = get_instruction_set_architecture(client=client_configuration, quantum_processor_id=quantum_processor_id)

    return QCSQuantumProcessor(quantum_processor_id=quantum_processor_id, isa=isa)
