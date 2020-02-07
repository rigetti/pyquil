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
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from pyquil.device._isa import Edge, GateInfo, ISA, MeasureInfo, Qubit, isa_from_graph, isa_to_graph
from pyquil.device._specs import Specs, specs_from_graph
from pyquil.noise import NoiseModel

PERFECT_FIDELITY = 1e0
PERFECT_DURATION = 1 / 100
DEFAULT_CZ_DURATION = 200
DEFAULT_CZ_FIDELITY = 0.89
DEFAULT_ISWAP_DURATION = 200
DEFAULT_ISWAP_FIDELITY = 0.90
DEFAULT_CPHASE_DURATION = 200
DEFAULT_CPHASE_FIDELITY = 0.85
DEFAULT_XY_DURATION = 200
DEFAULT_XY_FIDELITY = 0.86
DEFAULT_RX_DURATION = 50
DEFAULT_RX_FIDELITY = 0.95
DEFAULT_MEASURE_FIDELITY = 0.90
DEFAULT_MEASURE_DURATION = 2000


class AbstractDevice(ABC):
    @abstractmethod
    def qubits(self) -> List[int]:
        """
        A sorted list of qubits in the device topology.
        """

    @abstractmethod
    def qubit_topology(self) -> nx.Graph:
        """
        The connectivity of qubits in this device given as a NetworkX graph.
        """

    @abstractmethod
    def get_isa(self, oneq_type: str = "Xhalves", twoq_type: str = "CZ") -> ISA:
        """
        Construct an ISA suitable for targeting by compilation.

        This will raise an exception if the requested ISA is not supported by the device.

        :param oneq_type: The family of one-qubit gates to target
        :param twoq_type: The family of two-qubit gates to target
        """

    @abstractmethod
    def get_specs(self) -> Optional[Specs]:
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

    def __init__(self, name: str, raw: Dict[str, Any]):
        """
        :param name: name of the device
        :param raw: raw JSON response from the server with additional information about this device.
        """
        self.name = name
        self._raw = raw

        # TODO: Introduce distinction between supported ISAs and target ISA
        self._isa = ISA.from_dict(raw["isa"]) if "isa" in raw and raw["isa"] != {} else None
        self.specs = Specs.from_dict(raw["specs"]) if raw.get("specs") else None
        self.noise_model = (
            NoiseModel.from_dict(raw["noise_model"]) if raw.get("noise_model") else None
        )

    @property
    def isa(self) -> Optional[ISA]:
        warnings.warn("Accessing the static ISA is deprecated. Use `get_isa`", DeprecationWarning)
        return self._isa

    def qubits(self) -> List[int]:
        assert self._isa is not None
        return sorted(q.id for q in self._isa.qubits if not q.dead)

    def qubit_topology(self) -> nx.Graph:
        """
        The connectivity of qubits in this device given as a NetworkX graph.
        """
        assert self._isa is not None
        return isa_to_graph(self._isa)

    def get_specs(self) -> Optional[Specs]:
        return self.specs

    def get_isa(self, oneq_type: Optional[str] = None, twoq_type: Optional[str] = None) -> ISA:
        """
        Construct an ISA suitable for targeting by compilation.

        This will raise an exception if the requested ISA is not supported by the device.
        """
        if oneq_type is not None or twoq_type is not None:
            raise ValueError(
                "oneq_type and twoq_type are both fatally deprecated. If you want to "
                "make an ISA with custom gate types, you'll have to do it by hand."
            )

        def safely_get(attr: str, index: Union[int, Tuple[int, ...]], default: Any) -> Any:
            if self.specs is None:
                return default

            getter = getattr(self.specs, attr, None)
            if getter is None:
                return default

            array = getter()
            if (isinstance(index, int) and index < len(array)) or index in array:
                return array[index]
            else:
                return default

        def qubit_type_to_gates(q: Qubit) -> List[Union[GateInfo, MeasureInfo]]:
            gates: List[Union[GateInfo, MeasureInfo]] = [
                MeasureInfo(
                    operator="MEASURE",
                    qubit=q.id,
                    target="_",
                    fidelity=safely_get("fROs", q.id, DEFAULT_MEASURE_FIDELITY),
                    duration=DEFAULT_MEASURE_DURATION,
                ),
                MeasureInfo(
                    operator="MEASURE",
                    qubit=q.id,
                    target=None,
                    fidelity=safely_get("fROs", q.id, DEFAULT_MEASURE_FIDELITY),
                    duration=DEFAULT_MEASURE_DURATION,
                ),
            ]
            if q.type is None or "Xhalves" in q.type:
                gates += [
                    GateInfo(
                        operator="RZ",
                        parameters=["_"],
                        arguments=[q.id],
                        duration=PERFECT_DURATION,
                        fidelity=PERFECT_FIDELITY,
                    ),
                    GateInfo(
                        operator="RX",
                        parameters=[0.0],
                        arguments=[q.id],
                        duration=DEFAULT_RX_DURATION,
                        fidelity=PERFECT_FIDELITY,
                    ),
                ]
                gates += [
                    GateInfo(
                        operator="RX",
                        parameters=[param],
                        arguments=[q.id],
                        duration=DEFAULT_RX_DURATION,
                        fidelity=safely_get("f1QRBs", q.id, DEFAULT_RX_FIDELITY),
                    )
                    for param in [np.pi, -np.pi, np.pi / 2, -np.pi / 2]
                ]
            if q.type is not None and "WILDCARD" in q.type:
                gates += [
                    GateInfo(
                        operator="_",
                        parameters="_",
                        arguments=[q.id],
                        duration=PERFECT_DURATION,
                        fidelity=PERFECT_FIDELITY,
                    )
                ]
            return gates

        def edge_type_to_gates(e: Edge) -> List[GateInfo]:
            gates: List[GateInfo] = []
            if (
                e is None
                or isinstance(e.type, str)
                and "CZ" == e.type
                or isinstance(e.type, list)
                and "CZ" in e.type
            ):
                gates += [
                    GateInfo(
                        operator="CZ",
                        parameters=[],
                        arguments=["_", "_"],
                        duration=DEFAULT_CZ_DURATION,
                        fidelity=safely_get("fCZs", tuple(e.targets), DEFAULT_CZ_FIDELITY),
                    )
                ]
            if (
                e is None
                or isinstance(e.type, str)
                and "ISWAP" == e.type
                or isinstance(e.type, list)
                and "ISWAP" in e.type
            ):
                gates += [
                    GateInfo(
                        operator="ISWAP",
                        parameters=[],
                        arguments=["_", "_"],
                        duration=DEFAULT_ISWAP_DURATION,
                        fidelity=safely_get("fISWAPs", tuple(e.targets), DEFAULT_ISWAP_FIDELITY),
                    )
                ]
            if (
                e is None
                or isinstance(e.type, str)
                and "CPHASE" == e.type
                or isinstance(e.type, list)
                and "CPHASE" in e.type
            ):
                gates += [
                    GateInfo(
                        operator="CPHASE",
                        parameters=["theta"],
                        arguments=["_", "_"],
                        duration=DEFAULT_CPHASE_DURATION,
                        fidelity=safely_get("fCPHASEs", tuple(e.targets), DEFAULT_CPHASE_FIDELITY),
                    )
                ]
            if (
                e is None
                or isinstance(e.type, str)
                and "XY" == e.type
                or isinstance(e.type, list)
                and "XY" in e.type
            ):
                gates += [
                    GateInfo(
                        operator="XY",
                        parameters=["theta"],
                        arguments=["_", "_"],
                        duration=DEFAULT_XY_DURATION,
                        fidelity=safely_get("fXYs", tuple(e.targets), DEFAULT_XY_FIDELITY),
                    )
                ]
            if (
                e is None
                or isinstance(e.type, str)
                and "WILDCARD" == e.type
                or isinstance(e.type, list)
                and "WILDCARD" in e.type
            ):
                gates += [
                    GateInfo(
                        operator="_",
                        parameters="_",
                        arguments=["_", "_"],
                        duration=PERFECT_DURATION,
                        fidelity=PERFECT_FIDELITY,
                    )
                ]
            return gates

        assert self._isa is not None
        qubits = [
            Qubit(id=q.id, type=None, dead=q.dead, gates=qubit_type_to_gates(q))
            for q in self._isa.qubits
        ]
        edges = [
            Edge(targets=e.targets, type=None, dead=e.dead, gates=edge_type_to_gates(e))
            for e in self._isa.edges
        ]
        return ISA(qubits, edges)

    def __str__(self) -> str:
        return "<Device {}>".format(self.name)

    def __repr__(self) -> str:
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

    def qubit_topology(self) -> nx.Graph:
        return self.topology

    def get_isa(
        self, oneq_type: str = "Xhalves", twoq_type: Optional[Union[str, List[str]]] = None
    ) -> ISA:
        return isa_from_graph(self.topology, oneq_type=oneq_type, twoq_type=twoq_type)

    def get_specs(self) -> Specs:
        return specs_from_graph(self.topology)

    def qubits(self) -> List[int]:
        return sorted(self.topology.nodes)

    def edges(self) -> List[Tuple[Any, ...]]:
        return sorted(tuple(sorted(pair)) for pair in self.topology.edges)
