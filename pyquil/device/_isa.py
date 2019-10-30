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
from collections import namedtuple
from typing import Union

import networkx as nx
import numpy as np

from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.quilbase import Gate

THETA = Parameter("theta")
"Used as the symbolic parameter in RZ, CPHASE gates."

DEFAULT_QUBIT_TYPE = "Xhalves"
DEFAULT_EDGE_TYPE = "CZ"

Qubit = namedtuple("Qubit", ["id", "type", "dead", "gates"])
Edge = namedtuple("Edge", ["targets", "type", "dead", "gates"])
_ISA = namedtuple("_ISA", ["qubits", "edges"])

MeasureInfo = namedtuple("MeasureInfo", ["operator", "qubit", "target", "duration", "fidelity"])
GateInfo = namedtuple("GateInfo", ["operator", "parameters", "arguments", "duration", "fidelity"])

# make Qubit and Edge arguments optional
Qubit.__new__.__defaults__ = (None,) * len(Qubit._fields)
Edge.__new__.__defaults__ = (None,) * len(Edge._fields)
MeasureInfo.__new__.__defaults__ = (None,) * len(MeasureInfo._fields)
GateInfo.__new__.__defaults__ = (None,) * len(GateInfo._fields)


class ISA(_ISA):
    """
    Basic Instruction Set Architecture specification.

    :ivar Sequence[Qubit] qubits: The qubits associated with the ISA.
    :ivar Sequence[Edge] edges: The multi-qubit gates.
    """

    def to_dict(self):
        """
        Create a JSON-serializable representation of the ISA.

        The dictionary representation is of the form::

            {
                "1Q": {
                    "0": {
                        "type": "Xhalves"
                    },
                    "1": {
                        "type": "Xhalves",
                        "dead": True
                    },
                    ...
                },
                "2Q": {
                    "1-4": {
                        "type": "CZ"
                    },
                    "1-5": {
                        "type": "CZ"
                    },
                    ...
                },
                ...
            }

        :return: A dictionary representation of self.
        :rtype: Dict[str, Any]
        """

        def _maybe_configure(o, t):
            # type: (Union[Qubit,Edge], str) -> dict
            """
            Exclude default values from generated dictionary.

            :param Union[Qubit,Edge] o: The object to serialize
            :param str t: The default value for ``o.type``.
            :return: d
            """
            d = {}
            if o.gates is not None:
                d["gates"] = [
                    {"operator": i.operator,
                     "parameters": i.parameters,
                     "arguments": i.arguments,
                     "fidelity": i.fidelity,
                     "duration": i.duration} if isinstance(i, GateInfo) else
                    {"operator": "MEASURE",
                     "qubit": i.qubit,
                     "target": i.target,
                     "duration": i.duration,
                     "fidelity": i.fidelity} for i in o.gates]
            if o.gates is None and o.type != t:
                d["type"] = o.type
            if o.dead:
                d["dead"] = o.dead
            return d

        return {
            "1Q": {"{}".format(q.id): _maybe_configure(q, DEFAULT_QUBIT_TYPE) for q in self.qubits},
            "2Q": {"{}-{}".format(*edge.targets): _maybe_configure(edge, DEFAULT_EDGE_TYPE)
                   for edge in self.edges}
        }

    @staticmethod
    def from_dict(d):
        """
        Re-create the ISA from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored ISA.
        :rtype: ISA
        """
        return ISA(
            qubits=sorted([Qubit(id=int(qid),
                                 type=q.get("type", DEFAULT_QUBIT_TYPE),
                                 dead=q.get("dead", False))
                           for qid, q in d["1Q"].items()],
                          key=lambda qubit: qubit.id),
            edges=sorted([Edge(targets=[int(q) for q in eid.split('-')],
                               type=e.get("type", DEFAULT_EDGE_TYPE),
                               dead=e.get("dead", False))
                          for eid, e in d["2Q"].items()],
                         key=lambda edge: edge.targets),
        )


def gates_in_isa(isa):
    """
    Generate the full gateset associated with an ISA.

    :param ISA isa: The instruction set architecture for a QPU.
    :return: A sequence of Gate objects encapsulating all gates compatible with the ISA.
    :rtype: Sequence[Gate]
    """
    gates = []
    for q in isa.qubits:
        if q.dead:
            # TODO: dead qubits may in the future lead to some implicit re-indexing
            continue
        if q.type in ["Xhalves"]:
            gates.extend([
                Gate("I", [], [unpack_qubit(q.id)]),
                Gate("RX", [np.pi / 2], [unpack_qubit(q.id)]),
                Gate("RX", [-np.pi / 2], [unpack_qubit(q.id)]),
                Gate("RX", [np.pi], [unpack_qubit(q.id)]),
                Gate("RX", [-np.pi], [unpack_qubit(q.id)]),
                Gate("RZ", [THETA], [unpack_qubit(q.id)]),
            ])
        else:  # pragma no coverage
            raise ValueError("Unknown qubit type: {}".format(q.type))

    for e in isa.edges:
        if e.dead:
            continue
        targets = [unpack_qubit(t) for t in e.targets]
        if e.type in ["CZ", "ISWAP"]:
            gates.append(Gate(e.type, [], targets))
            gates.append(Gate(e.type, [], targets[::-1]))
        elif e.type in ["CPHASE"]:
            gates.append(Gate(e.type, [THETA], targets))
            gates.append(Gate(e.type, [THETA], targets[::-1]))
        else:  # pragma no coverage
            raise ValueError("Unknown edge type: {}".format(e.type))
    return gates


def isa_from_graph(graph: nx.Graph, oneq_type='Xhalves', twoq_type='CZ') -> ISA:
    """
    Generate an ISA object from a NetworkX graph.

    :param graph: The graph
    :param oneq_type: The type of 1-qubit gate. Currently 'Xhalves'
    :param twoq_type: The type of 2-qubit gate. One of 'CZ' or 'CPHASE'.
    """
    all_qubits = list(range(max(graph.nodes) + 1))
    qubits = [Qubit(i, type=oneq_type, dead=i not in graph.nodes) for i in all_qubits]
    edges = [Edge(sorted((a, b)), type=twoq_type, dead=False) for a, b in graph.edges]
    return ISA(qubits, edges)


def isa_to_graph(isa: ISA) -> nx.Graph:
    """
    Construct a NetworkX qubit topology from an ISA object.

    This discards information about supported gates.

    :param isa: The ISA.
    """
    return nx.from_edgelist(e.targets for e in isa.edges if not e.dead)
