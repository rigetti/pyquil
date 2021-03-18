from pyquil.external.rpcq import CompilerISA

import networkx as nx


def compiler_isa_to_graph(compiler_isa: CompilerISA) -> nx.Graph:
    """
    Generate an ``nx.Graph`` based on the qubits and edges of any ``CompilerISA``.
    """
    return nx.from_edgelist([int(i) for i in edge.ids] for edge in compiler_isa.edges.values())
