##############################################################################
# Copyright 2016-2019 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
from abc import ABC, abstractmethod
import networkx as nx
from typing import List
from pyquil.external.rpcq import CompilerISA


class AbstractQuantumProcessor(ABC):
    """
    A generic interface describing the qubits, topology, and compiler representation
    of any arbitrary quantum_processor class.
    """

    @abstractmethod
    def qubits(self) -> List[int]:
        """
        A sorted list of qubits in the quantum_processor topology.
        """

    @abstractmethod
    def qubit_topology(self) -> nx.Graph:
        """
        The connectivity of qubits in this quantum_processor given as a NetworkX graph.
        """

    @abstractmethod
    def to_compiler_isa(self) -> CompilerISA:
        """
        Construct an ISA suitable for targeting by compilation.
        This will raise an exception if the requested ISA is not supported by the quantum_processor.
        """
