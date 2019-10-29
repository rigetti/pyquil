import networkx as nx
from . import Device
from ._specs import Specs, specs_from_graph
from ._isa import ISA, isa_from_graph, isa_from_digraph


def ibmq_ourense(oneq_type='Xhalves', twoq_type='CNOT'):
    graph = nx.from_edgelist([
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (1, 3), (3, 1),
        (3, 4), (4, 3)
    ])

    return Device(name="ibmq_ourense",
                  raw={"isa": isa_from_digraph(graph, oneq_type=oneq_type, twoq_type=twoq_type).to_dict(),
                       "specs": specs_from_graph(graph).to_dict()})


def ibmq_yorktown(oneq_type='Xhalves', twoq_type='CNOT'):
    graph = nx.from_edgelist([
        (0, 1), (0, 2),
        (1, 2), (3, 2),
        (3, 4), (4, 2)
    ])

    return Device(name="ibmq_yorktown",
                  raw={"isa": isa_from_digraph(graph, oneq_type=oneq_type, twoq_type=twoq_type).to_dict(),
                       "specs": specs_from_graph(graph).to_dict()})

def google_bristlecone(oneq_type='Xhalves', twoq_type='CZ'):
    height = 12
    width = 6

    edges = []
    for i in range(height - 1):
        for j in range(width):
            qi = j + width * i
            bi = qi + width
            edges.append((qi, bi))
            if i % 2 == 0 and (qi % width) > 0:
                edges.append((qi, bi - 1))
            elif i % 2 != 0 and ((qi + 1) % width) > 0:
                edges.append((qi, bi + 1))

    graph = nx.from_edgelist(edges)

    return Device(name="google_bristlecone",
                  raw={"isa": isa_from_graph(graph).to_dict(),
                       "specs": specs_from_graph(graph).to_dict()})
