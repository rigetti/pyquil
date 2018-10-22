##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, List, Tuple

import networkx as nx
import numpy as np

from pyquil.noise import NoiseModel
from pyquil.parameters import Parameter
from pyquil.quilatom import unpack_qubit
from pyquil.quilbase import Gate

THETA = Parameter("theta")
"Used as the symbolic parameter in RZ, CPHASE gates."

DEFAULT_QUBIT_TYPE = "Xhalves"
DEFAULT_EDGE_TYPE = "CZ"

Qubit = namedtuple("Qubit", ["id", "type", "dead"])
Edge = namedtuple("Edge", ["targets", "type", "dead"])
_ISA = namedtuple("_ISA", ["qubits", "edges"])
QubitSpecs = namedtuple("_QubitSpecs", ["id", "fRO", "f1QRB", "T1", "T2", "fActiveReset"])
EdgeSpecs = namedtuple("_QubitQubitSpecs", ["targets", "fBellState", "fCZ", "fCPHASE"])
_Specs = namedtuple("_Specs", ["qubits_specs", "edges_specs"])


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
            if o.type != t:
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


class Specs(_Specs):
    """
    Basic specifications for the device, such as gate fidelities and coherence times.

    :ivar List[QubitSpecs] qubits_specs: The specs associated with individual qubits.
    :ivar List[EdgesSpecs] edges_specs: The specs associated with edges, or qubit-qubit pairs.
    """

    def f1QRBs(self):
        """
        Get a dictionary of single-qubit randomized benchmarking fidelities (normalized to unity)
        from the specs, keyed by qubit index.

        :return: A dictionary of 1QRBs, normalized to unity.
        :rtype: Dict[int, float]
        """
        return {qs.id: qs.f1QRB for qs in self.qubits_specs}

    def fROs(self):
        """
        Get a dictionary of single-qubit readout fidelities (normalized to unity)
        from the specs, keyed by qubit index.

        :return: A dictionary of RO fidelities, normalized to unity.
        :rtype: Dict[int, float]
        """
        return {qs.id: qs.fRO for qs in self.qubits_specs}

    def fActiveResets(self):
        """
        Get a dictionary of single-qubit active reset fidelities (normalized to unity) from the
        specs, keyed by qubit index.

        :return: A dictionary of reset fidelities, normalized to unity.
        """
        return {qs.id: qs.fActiveReset for qs in self.qubits_specs}

    def T1s(self):
        """
        Get a dictionary of T1s (in seconds) from the specs, keyed by qubit index.

        :return: A dictionary of T1s, in seconds.
        :rtype: Dict[int, float]
        """
        return {qs.id: qs.T1 for qs in self.qubits_specs}

    def T2s(self):
        """
        Get a dictionary of T2s (in seconds) from the specs, keyed by qubit index.

        :return: A dictionary of T2s, in seconds.
        :rtype: Dict[int, float]
        """
        return {qs.id: qs.T2 for qs in self.qubits_specs}

    def fBellStates(self):
        """
        Get a dictionary of two-qubit Bell state fidelities (normalized to unity)
        from the specs, keyed by targets (qubit-qubit pairs).

        :return: A dictionary of Bell state fidelities, normalized to unity.
        :rtype: Dict[tuple(int, int), float]
        """
        return {tuple(es.targets): es.fBellState for es in self.edges_specs}

    def fCZs(self):
        """
        Get a dictionary of CZ fidelities (normalized to unity) from the specs,
        keyed by targets (qubit-qubit pairs).

        :return: A dictionary of CZ fidelities, normalized to unity.
        :rtype: Dict[tuple(int, int), float]
        """
        return {tuple(es.targets): es.fCZ for es in self.edges_specs}

    def fCPHASEs(self):
        """
        Get a dictionary of CPHASE fidelities (normalized to unity) from the specs,
        keyed by targets (qubit-qubit pairs).

        :return: A dictionary of CPHASE fidelities, normalized to unity.
        :rtype: Dict[tuple(int, int), float]
        """
        return {tuple(es.targets): es.fCPHASE for es in self.edges_specs}

    def to_dict(self):
        """
        Create a JSON-serializable representation of the device Specs.

        The dictionary representation is of the form::

            {
                '1Q': {
                    "0": {
                        "f1QRB": 0.99,
                        "T1": 20e-6,
                        ...
                    },
                    "1": {
                        "f1QRB": 0.989,
                        "T1": 19e-6,
                        ...
                    },
                    ...
                },
                '2Q': {
                    "1-4": {
                        "fBellState": 0.93,
                        "fCZ": 0.92,
                        "fCPHASE": 0.91
                    },
                    "1-5": {
                        "fBellState": 0.9,
                        "fCZ": 0.89,
                        "fCPHASE": 0.88
                    },
                    ...
                },
                ...
            }

        :return: A dctionary representation of self.
        :rtype: Dict[str, Any]
        """
        return {
            '1Q': {
                "{}".format(qs.id): {
                    'f1QRB': qs.f1QRB,
                    'fRO': qs.fRO,
                    'T1': qs.T1,
                    'T2': qs.T2,
                    'fActiveReset': qs.fActiveReset
                } for qs in self.qubits_specs
            },
            '2Q': {
                "{}-{}".format(*es.targets): {
                    'fBellState': es.fBellState,
                    'fCZ': es.fCZ,
                    'fCPHASE': es.fCPHASE
                } for es in self.edges_specs
            }
        }

    @staticmethod
    def from_dict(d):
        """
        Re-create the Specs from a dictionary representation.

        :param Dict[str, Any] d: The dictionary representation.
        :return: The restored Specs.
        :rtype: Specs
        """
        return Specs(
            qubits_specs=sorted([QubitSpecs(id=int(q),
                                            fRO=qspecs.get('fRO'),
                                            f1QRB=qspecs.get('f1QRB'),
                                            T1=qspecs.get('T1'),
                                            T2=qspecs.get('T2'),
                                            fActiveReset=qspecs.get('fActiveReset'))
                                 for q, qspecs in d["1Q"].items()],
                                key=lambda qubit_specs: qubit_specs.id),
            edges_specs=sorted([EdgeSpecs(targets=[int(q) for q in e.split('-')],
                                          fBellState=especs.get('fBellState'),
                                          fCZ=especs.get('fCZ'),
                                          fCPHASE=especs.get('fCPHASE'))
                                for e, especs in d["2Q"].items()],
                               key=lambda edge_specs: edge_specs.targets)
        )


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


def specs_from_graph(graph: nx.Graph):
    """
    Generate a Specs object from a NetworkX graph with placeholder values for the actual specs.

    :param graph: The graph
    """
    qspecs = [QubitSpecs(id=q, fRO=0.90, f1QRB=0.99, T1=30e-6, T2=30e-6, fActiveReset=0.99)
              for q in graph.nodes]
    especs = [EdgeSpecs(targets=(q1, q2), fBellState=0.90, fCZ=0.90, fCPHASE=0.80)
              for q1, q2 in graph.edges]
    return Specs(qspecs, especs)


def isa_to_graph(isa: ISA) -> nx.Graph:
    """
    Construct a NetworkX qubit topology from an ISA object.

    This discards information about supported gates.

    :param isa: The ISA.
    """
    return nx.from_edgelist(e.targets for e in isa.edges if not e.dead)


class AbstractDevice(ABC):

    @abstractmethod
    def qubits(self):
        """
        A sorted list of qubits in the device topology.
        """

    @abstractmethod
    def qubit_topology(self) -> nx.Graph:
        """
        The connectivity of qubits in this device given as a NetworkX graph.
        """

    @abstractmethod
    def get_isa(self, oneq_type='Xhalves', twoq_type='CZ') -> ISA:
        """
        Construct an ISA suitable for targeting by compilation.

        This will raise an exception if the requested ISA is not supported by the device.

        :param oneq_type: The family of one-qubit gates to target
        :param twoq_type: The family of two-qubit gates to target
        """

    @abstractmethod
    def get_specs(self) -> Specs:
        """
        Construct a Specs object required by compilation
        """


class Device(AbstractDevice):
    """
    A device (quantum chip) that can accept programs.

    Only devices that are online will actively be
    accepting new programs. In addition to the ``self._raw`` attribute, two other attributes are
    optionally constructed from the entries in ``self._raw`` -- ``isa`` and ``noise_model`` -- which
    should conform to the dictionary format required by the ``.from_dict()`` methods for ``ISA``
    and ``NoiseModel``, respectively.

    :ivar dict _raw: Raw JSON response from the server with additional information about the device.
    :ivar ISA isa: The instruction set architecture (ISA) for the device.
    :ivar NoiseModel noise_model: The noise model for the device.
    """

    def __init__(self, name, raw):
        """
        :param name: name of the device
        :param raw: raw JSON response from the server with additional information about this device.
        """
        self.name = name
        self._raw = raw

        # TODO: Introduce distinction between supported ISAs and target ISA
        self._isa = ISA.from_dict(raw['isa']) if 'isa' in raw and raw['isa'] != {} else None
        self.specs = Specs.from_dict(raw['specs']) if raw.get('specs') else None
        self.noise_model = NoiseModel.from_dict(raw['noise_model']) \
            if raw.get('noise_model') else None

    @property
    def isa(self):
        warnings.warn("Accessing the static ISA is deprecated. Use `get_isa`", DeprecationWarning)
        return self._isa

    def qubits(self):
        return sorted(q.id for q in self._isa.qubits if not q.dead)

    def qubit_topology(self) -> nx.Graph:
        """
        The connectivity of qubits in this device given as a NetworkX graph.
        """
        return isa_to_graph(self._isa)

    def get_isa(self, oneq_type='Xhalves', twoq_type='CZ') -> ISA:
        """
        Construct an ISA suitable for targeting by compilation.

        This will raise an exception if the requested ISA is not supported by the device.

        :param oneq_type: The family of one-qubit gates to target
        :param twoq_type: The family of two-qubit gates to target
        """
        qubits = [Qubit(id=q.id, type=oneq_type, dead=q.dead) for q in self._isa.qubits]
        edges = [Edge(targets=e.targets, type=twoq_type, dead=e.dead) for e in self._isa.edges]
        return ISA(qubits, edges)

    def get_specs(self):
        return self.specs

    def __str__(self):
        return '<Device {}>'.format(self.name)

    def __repr__(self):
        return str(self)


class NxDevice(AbstractDevice):
    """A shim over the AbstractDevice API backed by a NetworkX graph.

    A ``Device`` holds information about the physical device.
    Specifically, you might want to know about connectivity, available gates, performance specs,
    and more. This class implements the AbstractDevice API for devices not available via
    ``get_devices()``. Instead, the user is responsible for constructing a NetworkX
    graph which represents a chip topology.
    """

    def __init__(self, topology: nx.Graph) -> None:
        self.topology = topology

    def qubit_topology(self):
        return self.topology

    def get_isa(self, oneq_type='Xhalves', twoq_type='CZ'):
        return isa_from_graph(self.topology, oneq_type=oneq_type, twoq_type=twoq_type)

    def get_specs(self):
        return specs_from_graph(self.topology)

    def qubits(self) -> List[int]:
        return sorted(self.topology.nodes)

    def edges(self) -> List[Tuple[int, int]]:
        return sorted(tuple(sorted(pair)) for pair in self.topology.edges)  # type: ignore
