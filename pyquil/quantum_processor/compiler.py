"""An implementation of an AbstractQuantumProcessor initialized with a user constructed ``CompilerISA``."""

import networkx as nx

from pyquil.external.rpcq import CompilerISA
from pyquil.quantum_processor._base import AbstractQuantumProcessor
from pyquil.quantum_processor.transformers import compiler_isa_to_graph


class CompilerQuantumProcessor(AbstractQuantumProcessor):
    """An AbstractQuantumProcessor initialized with a user constructed ``CompilerISA``."""

    _isa: CompilerISA

    def __init__(self, isa: CompilerISA) -> None:
        """Initialize a CompilerQuantumProcessor with a given ``CompilerISA``."""
        self._isa = isa

    def qubit_topology(self) -> nx.Graph:
        """Return a NetworkX graph that represents the connectivity of qubits in this quantum_processor."""
        return compiler_isa_to_graph(self._isa)

    def to_compiler_isa(self) -> CompilerISA:
        """Return the CompilerISA that this quantum_processor is initialized with."""
        return self._isa

    def qubits(self) -> list[int]:
        """Return the qubits in the quantum_processor topology as a sorted list."""
        return sorted([int(node_id) for node_id, node in self._isa.qubits.items()])
