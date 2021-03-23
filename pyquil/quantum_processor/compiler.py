from pyquil.quantum_processor._base import AbstractQuantumProcessor
from typing import List
from pyquil.quantum_processor.transformers import compiler_isa_to_graph
from pyquil.external.rpcq import CompilerISA

import networkx as nx


class CompilerQuantumProcessor(AbstractQuantumProcessor):
    """
    An AbstractQuantumProcessor initialized with a user constructed ``CompilerISA``.
    """

    _isa: CompilerISA

    def __init__(self, isa: CompilerISA) -> None:
        self._isa = isa

    def qubit_topology(self) -> nx.Graph:
        return compiler_isa_to_graph(self._isa)

    def to_compiler_isa(self) -> CompilerISA:
        return self._isa

    def qubits(self) -> List[int]:
        return sorted([int(node_id) for node_id, node in self._isa.qubits.items()])
