from typing import Any, Tuple

from pyquil.quantum_processor._base import AbstractQuantumProcessor
from pyquil.quantum_processor.transformers import graph_to_compiler_isa
from typing import List, Optional
from pyquil.external.rpcq import CompilerISA

import networkx as nx


class NxQuantumProcessor(AbstractQuantumProcessor):
    """
    An AbstractQuantumProcessor initialized with a user constructed NetworkX graph topology.
    Notably, this class is able to serialize a ``CompilerISA`` based on the
    graph topology and the configured 1Q and 2Q gates.
    """

    def __init__(
        self,
        topology: nx.Graph,
        gates_1q: Optional[List[str]] = None,
        gates_2q: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a new NxQuantumProcessor.

        :param topology: The graph topology of the quantum_processor.
        :param gates_1q: A list of 1Q gate names supported by all qubits in the quantum_processor.
        :param gates_2q: A list of 2Q gate names supported all edges in the quantum_processor.
        """
        self.topology = topology
        self.gates_1q = gates_1q
        self.gates_2q = gates_2q

    def qubit_topology(self) -> nx.Graph:
        return self.topology

    def to_compiler_isa(self) -> CompilerISA:
        """
        Generate a ``CompilerISA`` object based on a NetworkX graph and the
        ``gates_1q`` and ``gates_2q`` with which the quantum_processor was initialized.

        May raise ``GraphGateError`` if the specified gates are not supported.
        """
        return graph_to_compiler_isa(self.topology, gates_1q=self.gates_1q, gates_2q=self.gates_2q)

    def qubits(self) -> List[int]:
        return sorted(self.topology.nodes)

    def edges(self) -> List[Tuple[Any, ...]]:
        return sorted(tuple(sorted(pair)) for pair in self.topology.edges)
