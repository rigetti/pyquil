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
from collections import namedtuple

import numpy as np
from typing import Union, Sequence

Qubit = namedtuple("Qubit", ["id", "type", "dead"])
Edge = namedtuple("Edge", ["targets", "type", "dead"])
_ISA = namedtuple("_ISA", ["name", "version", "timestamp", "qubits", "edges"])


class ISA(_ISA):
    """
    Basic Instruction Set Architecture specification.

    :ivar str name: The QPU ISA name
    :ivar str version: The version of the ISA
    :ivar int|float timestamp: A timestamp of when the ISA was defined.
    :ivar Sequence[Qubit] qubits: The qubits associated with the ISA
    :ivar Sequence[Edge] edges: The multi-qubit gates.
    """

    def to_dict(self):
        """
        Create a JSON serializable representation of the ISA.

        :return: A dictionary representation of self.
        :rtype: Dict[str,Any]
        """

        def _maybe_configure(d, o):
            # type: (dict, Union[Qubit,Edge]) -> dict
            if o.type:
                d["type"] = o.type
            if o.dead:
                d["dead"] = o.dead
            return d

        return {
            "id": {
                "name": self.name,
                "version": self.version,
                "timestamp": self.timestamp,
            },
            "logical-hardware": [
                [_maybe_configure({"qubit-id": q.id}, q) for q in self.qubits],
                [_maybe_configure({"action-qubits": a.targets}, a) for a in self.edges],
            ]
        }

    @classmethod
    def from_dict(cls, d):
        """
        Re-create the ISA from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored ISA.
        :rtype: ISA
        """
        return cls(
            name=d["id"]["name"],
            version=d["id"].get("version", "0.0"),
            timestamp=d["id"].get("timestamp"),
            qubits=[Qubit(id=q["qubit-id"], type=q.get("type"), dead=q.get("dead"))
                    for q in d["logical-hardware"][0]],
            edges=[Edge(targets=e["action-qubits"], type=e.get("type"), dead=e.get("dead"))
                   for e in d["logical-hardware"][1]],
        )


_KrausModel = namedtuple("_KrausModel", ["gate", "params", "targets", "kraus_ops", "fidelity"])


class KrausModel(_KrausModel):

    @staticmethod
    def unpack_kraus_matrix(m):
        """
        Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.

        :param list|np.array m: The representation of a Kraus operator. Either a complex square
        matrix (as numpy array or nested lists) or a pair of real square matrices (as numpy arrays
        or nested lists) representing the element-wise real and imaginary part of m.
        :return: A complex square numpy array representing the Kraus operator.
        :rtype: np.array
        """
        m = np.asarray(m, dtype=complex)
        if m.ndim == 3:
            m = m[0] + 1j * m[1]
        if not m.ndim == 2:  # pragma no coverage
            raise ValueError("Need 2d array.")
        if not m.shape[0] == m.shape[1]:  # pragma no coverage
            raise ValueError("Need square matrix.")
        return m

    def to_dict(self):
        res = self._asdict()
        res['kraus_ops'] = [[k.real.tolist(), k.imag.tolist()] for k in self.kraus_ops]
        return res

    @classmethod
    def from_dict(cls, d):
        kraus_ops = [KrausModel.unpack_kraus_matrix(k) for k in d['kraus_ops']]
        return cls(d['gate'], d['params'], d['targets'], kraus_ops, d['fidelity'])

    def __eq__(self, other):
        return isinstance(other, KrausModel) and self.to_dict() == other.to_dict()


_NoiseModel = namedtuple("_NoiseModel", ["isa_name", "gates", "assignment_probs"])


class NoiseModel(_NoiseModel):
    """
    Encapsulate the QPU noise model containing information about the noisy identity gates,
    RX(pi/2) gates and CZ gates on the defined graph edges.
    The tomographies and assignment probabilities are ordered in the same way as they appear
    in the ISA.

    :ivar str isa_name: The name of the instruction set architecture for the QPU.
    :ivar Sequence[KrausModel] gates: The tomographic estimates of all gates.
    :ivar Dict[int,np.array] assignment_probs: The single qubit readout assignment
    probability matrices keyed by qubit id.
    """

    def to_dict(self):
        """
        Create a JSON serializable representation of the noise model.

        :return: A dictionary representation of self.
        :rtype: Dict[str,Any]
        """
        return {
            "isa_name": self.isa_name,
            "gates": [km.to_dict() for km in self.gates],
            "assignment_probs": {qid: a.tolist() for qid, a in self.assignment_probs.items()},
        }

    @classmethod
    def from_dict(cls, d):
        """
        Re-create the noise model from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored noise model.
        :rtype: NoiseModel
        """
        return cls(
            isa_name=d["isa_name"],
            gates=[KrausModel.from_dict(t) for t in d["gates"]],
            assignment_probs={qid: np.array(a) for qid, a in d["assignment_probs"].items()},
        )

    def gates_by_name(self, name):
        """
        Return all defined noisy gates of a particular gate name.

        :param str name: The gate name
        :return: A list of noise models representing that gate.
        :rtype: Sequence[KrausModel]
        """
        return [g for g in self.gates if g.gate == name]

    def __eq__(self, other):
        return isinstance(other, NoiseModel) and self.to_dict() == other.to_dict()
