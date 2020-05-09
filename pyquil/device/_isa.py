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
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import networkx as nx
import numpy as np

from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.quilbase import Gate

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass

DEFAULT_QUBIT_TYPE = "Xhalves"
DEFAULT_EDGE_TYPE = "CZ"
THETA = Parameter("theta")
"Used as the symbolic parameter in RZ, CPHASE gates."


@dataclass
class MeasureInfo:
    operator: Optional[str] = None
    qubit: Optional[Union[int, str]] = None
    target: Optional[Union[int, str]] = None
    duration: Optional[float] = None
    fidelity: Optional[float] = None


@dataclass
class GateInfo:
    operator: Optional[str] = None
    parameters: Optional[Sequence[Union[str, float]]] = None
    arguments: Optional[Sequence[Union[str, float]]] = None
    duration: Optional[float] = None
    fidelity: Optional[float] = None


@dataclass
class Qubit:
    id: int
    type: Optional[str] = None
    dead: Optional[bool] = None
    gates: Optional[Sequence[Union[GateInfo, MeasureInfo]]] = None


@dataclass
class Edge:
    targets: Tuple[int, ...]
    type: Optional[Union[List[str], str]] = None
    dead: Optional[bool] = None
    gates: Optional[Sequence[GateInfo]] = None


@dataclass
class ISA:
    """
    Basic Instruction Set Architecture specification.

    :ivar qubits: The qubits associated with the ISA.
    :ivar edges: The multi-qubit gates.
    """

    qubits: Sequence[Qubit]
    edges: Sequence[Edge]

    def to_dict(self) -> Dict[str, Any]:
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
        """

        def _maybe_configure(o: Union[Qubit, Edge], t: Union[str, List[str]]) -> Dict[str, Any]:
            """
            Exclude default values from generated dictionary.

            :param o: The object to serialize
            :param t: The default value for ``o.type``.
            :return: d
            """
            d: Dict[str, Any] = {}

            if o.gates is None or len(o.gates) == 0:
                inferred_type = o.type if (o.type is not None and o.type != t) else t
                inferred_gates = convert_gate_type_to_gate_information(inferred_type)
            else:
                inferred_gates = cast(List[Union[GateInfo, MeasureInfo]], o.gates)

            d["gates"] = [
                {
                    "operator": i.operator,
                    "parameters": i.parameters,
                    "arguments": i.arguments,
                    "fidelity": i.fidelity,
                    "duration": i.duration,
                }
                if isinstance(i, GateInfo)
                else {
                    "operator": "MEASURE",
                    "qubit": i.qubit,
                    "target": i.target,
                    "duration": i.duration,
                    "fidelity": i.fidelity,
                }
                for i in inferred_gates
            ]
            if o.dead:
                d["dead"] = o.dead
            return d

        return {
            "1Q": {"{}".format(q.id): _maybe_configure(q, DEFAULT_QUBIT_TYPE) for q in self.qubits},
            "2Q": {
                "{}-{}".format(*edge.targets): _maybe_configure(edge, DEFAULT_EDGE_TYPE)
                for edge in self.edges
            },
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ISA":
        """
        Re-create the ISA from a dictionary representation.

        :param d: The dictionary representation.
        :return: The restored ISA.
        """
        return ISA(
            qubits=sorted(
                [
                    Qubit(
                        id=int(qid),
                        type=q.get("type", DEFAULT_QUBIT_TYPE),
                        dead=q.get("dead", False),
                    )
                    for qid, q in d["1Q"].items()
                ],
                key=lambda qubit: qubit.id,
            ),
            edges=sorted(
                [
                    Edge(
                        targets=tuple(int(q) for q in eid.split("-")),
                        type=e.get("type", DEFAULT_EDGE_TYPE),
                        dead=e.get("dead", False),
                    )
                    for eid, e in d["2Q"].items()
                ],
                key=lambda edge: edge.targets,
            ),
        )


def convert_gate_type_to_gate_information(
    gate_type: Union[str, List[str]]
) -> List[Union[GateInfo, MeasureInfo]]:
    if isinstance(gate_type, str):
        gate_type = [gate_type]

    gate_information: List[Union[GateInfo, MeasureInfo]] = []

    for type_keyword in gate_type:
        if type_keyword == "Xhalves":
            gate_information.extend(
                [
                    GateInfo("I", [], ["_"]),
                    GateInfo("RX", [0], ["_"]),
                    GateInfo("RX", [np.pi / 2], ["_"]),
                    GateInfo("RX", [-np.pi / 2], ["_"]),
                    GateInfo("RX", [np.pi], ["_"]),
                    GateInfo("RX", [-np.pi], ["_"]),
                    GateInfo("RZ", ["theta"], ["_"]),
                    MeasureInfo(operator="MEASURE", qubit="_", target="_"),
                    MeasureInfo(operator="MEASURE", qubit="_", target=None),
                ]
            )
        elif type_keyword == "WILDCARD":
            gate_information.extend([GateInfo("_", "_", ["_"]), GateInfo("_", "_", ["_", "_"])])
        elif type_keyword == "CZ":
            gate_information.extend([GateInfo("CZ", [], ["_", "_"])])
        elif type_keyword == "ISWAP":
            gate_information.extend([GateInfo("ISWAP", [], ["_", "_"])])
        elif type_keyword == "CPHASE":
            gate_information.extend([GateInfo("CPHASE", ["theta"], ["_", "_"])])
        elif type_keyword == "XY":
            gate_information.extend([GateInfo("XY", ["theta"], ["_", "_"])])
        else:
            raise ValueError("Unknown edge type: {}".format(type_keyword))

    return gate_information


def gates_in_isa(isa: ISA) -> List[Gate]:
    """
    Generate the full gateset associated with an ISA.

    :param isa: The instruction set architecture for a QPU.
    :return: A sequence of Gate objects encapsulating all gates compatible with the ISA.
    """
    gates = []
    for q in isa.qubits:
        if q.dead:
            # TODO: dead qubits may in the future lead to some implicit re-indexing
            continue
        if q.type == "Xhalves":
            gates.extend(
                [
                    Gate("I", [], [unpack_qubit(q.id)]),
                    Gate("RX", [np.pi / 2], [unpack_qubit(q.id)]),
                    Gate("RX", [-np.pi / 2], [unpack_qubit(q.id)]),
                    Gate("RX", [np.pi], [unpack_qubit(q.id)]),
                    Gate("RX", [-np.pi], [unpack_qubit(q.id)]),
                    Gate("RZ", [THETA], [unpack_qubit(q.id)]),
                ]
            )
        elif q.type == "WILDCARD":
            gates.extend([Gate("_", "_", [unpack_qubit(q.id)])])
        else:  # pragma no coverage
            raise ValueError("Unknown qubit type: {}".format(q.type))

    for e in isa.edges:
        if e.dead:
            continue
        targets = [unpack_qubit(t) for t in e.targets]
        assert e.type is not None
        edge_type = e.type if isinstance(e.type, list) else [e.type]
        if "CZ" in edge_type:
            gates.append(Gate("CZ", [], targets))
            gates.append(Gate("CZ", [], targets[::-1]))
            continue
        if "ISWAP" in edge_type:
            gates.append(Gate("ISWAP", [], targets))
            gates.append(Gate("ISWAP", [], targets[::-1]))
            continue
        if "CPHASE" in edge_type:
            gates.append(Gate("CPHASE", [THETA], targets))
            gates.append(Gate("CPHASE", [THETA], targets[::-1]))
            continue
        if "XY" in edge_type:
            gates.append(Gate("XY", [THETA], targets))
            gates.append(Gate("XY", [THETA], targets[::-1]))
            continue
        assert e.type is not None
        if "WILDCARD" in e.type:
            gates.append(Gate("_", "_", targets))
            gates.append(Gate("_", "_", targets[::-1]))
            continue

        raise ValueError("Unknown edge type: {}".format(e.type))
    return gates


def isa_from_graph(
    graph: nx.Graph, oneq_type: str = "Xhalves", twoq_type: Optional[Union[str, List[str]]] = None
) -> ISA:
    """
    Generate an ISA object from a NetworkX graph.

    :param graph: The graph
    :param oneq_type: The type of 1-qubit gate. Currently 'Xhalves'
    :param twoq_type: The type of 2-qubit gate. One or more of 'CZ', 'CPHASE', 'ISWAP', 'XY'.
                      The default, None, is a synonym for ["CZ", "XY"].
    """
    all_qubits = list(range(max(graph.nodes) + 1))
    qubits = [Qubit(i, type=oneq_type, dead=i not in graph.nodes) for i in all_qubits]
    edges = [
        Edge(
            tuple(sorted((a, b))), type=["CZ", "XY"] if twoq_type is None else twoq_type, dead=False
        )
        for a, b in graph.edges
    ]
    return ISA(qubits, edges)


def isa_to_graph(isa: ISA) -> nx.Graph:
    """
    Construct a NetworkX qubit topology from an ISA object.

    This discards information about supported gates.

    :param isa: The ISA.
    """
    return nx.from_edgelist(e.targets for e in isa.edges if not e.dead)
