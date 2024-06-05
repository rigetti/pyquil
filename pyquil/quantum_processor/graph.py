"""An implementation of an AbstractQuantumProcessor based on a NetworkX graph topology."""

from typing import Any, Optional

import networkx as nx

from pyquil.external.rpcq import CompilerISA
from pyquil.quantum_processor._base import AbstractQuantumProcessor
from pyquil.quantum_processor.transformers import graph_to_compiler_isa


class NxQuantumProcessor(AbstractQuantumProcessor):
    """An AbstractQuantumProcessor initialized with a user constructed NetworkX graph topology.

    Notably, this class is able to serialize a ``CompilerISA`` based on the
    graph topology and the configured 1Q and 2Q gates.
    """

    def __init__(
        self,
        topology: nx.Graph,
        gates_1q: Optional[list[str]] = None,
        gates_2q: Optional[list[str]] = None,
    ) -> None:
        """Initialize a new NxQuantumProcessor.

        :param topology: The graph topology of the quantum_processor.
        :param gates_1q: A list of 1Q gate names supported by all qubits in the quantum_processor.
        :param gates_2q: A list of 2Q gate names supported all edges in the quantum_processor.
        """
        self.topology = topology
        self.gates_1q = gates_1q
        self.gates_2q = gates_2q

    def qubit_topology(self) -> nx.Graph:
        """Return the NetworkX graph that represents the connectivity of qubits in this quantum_processor."""
        return self.topology

    def to_compiler_isa(self) -> CompilerISA:
        """Generate a ``CompilerISA`` object based on the NetworkX graph, ``gates_1q``, and ``gates_2q``.

        May raise ``GraphGateError`` if the specified gates are not supported.
        """
        return graph_to_compiler_isa(self.topology, gates_1q=self.gates_1q, gates_2q=self.gates_2q)

    def qubits(self) -> list[int]:
        """Return a sorted list of qubits in the quantum_processor."""
        return sorted(self.topology.nodes)

    def edges(self) -> list[tuple[Any, ...]]:
        """Return a sorted list of edges in the quantum_processor."""
        return sorted(tuple(sorted(pair)) for pair in self.topology.edges)
