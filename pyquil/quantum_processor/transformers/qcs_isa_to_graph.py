"""Convert a QCS Instruction Set Architecture to a NetworkX graph."""

import networkx as nx
from qcs_sdk.qpu.isa import InstructionSetArchitecture


def qcs_isa_to_graph(isa: InstructionSetArchitecture) -> nx.Graph:
    """Convert a QCS Instruction Set Architecture to a NetworkX graph."""
    return nx.from_edgelist(edge.node_ids for edge in isa.architecture.edges)
