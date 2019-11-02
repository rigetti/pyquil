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
from typing import List, Tuple

import networkx as nx
import numpy as np

from pyquil.device._isa import (Edge, GateInfo, ISA, MeasureInfo, Qubit, isa_from_graph,
                                isa_to_graph)
from pyquil.device._specs import Specs, specs_from_graph
from pyquil.noise import NoiseModel

PERFECT_FIDELITY = 1e0
PERFECT_DURATION = 1 / 100
DEFAULT_CZ_DURATION = 200
DEFAULT_CZ_FIDELITY = 0.89
DEFAULT_RX_DURATION = 50
DEFAULT_RX_FIDELITY = 0.95
DEFAULT_MEASURE_FIDELITY = 0.90
DEFAULT_MEASURE_DURATION = 2000


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

    def get_specs(self):
        return self.specs

    def get_isa(self, oneq_type=None, twoq_type=None) -> ISA:
        """
        Construct an ISA suitable for targeting by compilation.

        This will raise an exception if the requested ISA is not supported by the device.
        """
        if oneq_type is not None or twoq_type is not None:
            raise ValueError("oneq_type and twoq_type are both fatally deprecated. If you want to "
                             "make an ISA with custom gate types, you'll have to do it by hand.")

        qubits = [Qubit(id=q.id, type=None, dead=q.dead, gates=[
            MeasureInfo(operator="MEASURE", qubit=q.id, target="_",
                        fidelity=self.specs.fROs()[q.id] or DEFAULT_MEASURE_FIDELITY,
                        duration=DEFAULT_MEASURE_DURATION),
            MeasureInfo(operator="MEASURE", qubit=q.id, target=None,
                        fidelity=self.specs.fROs()[q.id] or DEFAULT_MEASURE_FIDELITY,
                        duration=DEFAULT_MEASURE_DURATION),
            GateInfo(operator="RZ", parameters=["_"], arguments=[q.id],
                     duration=PERFECT_DURATION, fidelity=PERFECT_FIDELITY),
            GateInfo(operator="RX", parameters=[0.0], arguments=[q.id],
                     duration=DEFAULT_RX_DURATION, fidelity=PERFECT_FIDELITY)] + [
                GateInfo(operator="RX", parameters=[param], arguments=[q.id],
                         duration=DEFAULT_RX_DURATION,
                         fidelity=self.specs.f1QRBs()[q.id] or DEFAULT_RX_FIDELITY)
                for param in [np.pi, -np.pi, np.pi / 2, -np.pi / 2]])
            for q in self._isa.qubits]
        edges = [Edge(targets=e.targets, type=None, dead=e.dead, gates=[
                    GateInfo(operator="CZ", parameters=[], arguments=["_", "_"],
                             duration=DEFAULT_CZ_DURATION,
                             fidelity=self.specs.fCZs()[tuple(e.targets)] or DEFAULT_CZ_FIDELITY)])
                 for e in self._isa.edges]
        return ISA(qubits, edges)

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
