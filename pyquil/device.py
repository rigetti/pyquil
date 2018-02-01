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

    :ivar str name: The QPU ISA name.
    :ivar str version: The version of the ISA.
    :ivar Union[int,float] timestamp: A timestamp of when the ISA was defined.
    :ivar Sequence[Qubit] qubits: The qubits associated with the ISA.
    :ivar Sequence[Edge] edges: The multi-qubit gates.
    """

    def to_dict(self):
        """
        Create a JSON serializable representation of the ISA.

        The dictionary representation is of the form::

            {
                "id": {
                    "name": "example_qpu",
                    "version": "0.1",
                    "timestamp": "23423423"
                },
                "logical-hardware": [
                    [
                        {
                            "qubit-id": 0,
                            "type": "Xhalves",
                            "dead": False
                        },
                        {
                            "qubit-id": 1,
                            "type": "Xhalves",
                            "dead": False
                        }
                    ],
                    [
                        {
                            "action-qubits": [0, 1],
                            "type": "CZ",
                            "dead": False
                        }
                    ]
                ]
            }

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

    @staticmethod
    def from_dict(d):
        """
        Re-create the ISA from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored ISA.
        :rtype: ISA
        """
        return ISA(
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
    """
    Encapsulate a single gate's noise model.

    :ivar str gate: The name of the gate.
    :ivar Sequence[float] params: Optional parameters for the gate.
    :ivar Sequence[int] targets: The target qubit ids.
    :ivar Sequence[np.array] kraus_ops: The Kraus operators (must be square complex numpy arrays).
    :ivar float fidelity: The average gate fidelity associated with the Kraus map relative to the
        ideal operation.
    """

    @staticmethod
    def unpack_kraus_matrix(m):
        """
        Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.

        :param Union[list,np.array] m: The representation of a Kraus operator. Either a complex
            square matrix (as numpy array or nested lists) or a pair of real matrices (as numpy
            arrays or nested lists) representing the element-wise real and imaginary part of m.
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
        """
        Create a dictionary representation of a KrausModel.

        For example::

            {
                "gate": "RX",
                "params": np.pi,
                "targets": [0],
                "kraus_ops": [            # In this example single Kraus op = ideal RX(pi) gate
                    [[[0,   0],           # element-wise real part of matrix
                      [0,   0]],
                      [[0, -1],           # element-wise imaginary part of matrix
                      [-1, 0]]]
                ],
                "fidelity": 1.0
            }

        :return: A JSON compatible dictionary representation.
        :rtype: Dict[str,Any]
        """
        res = self._asdict()
        res['kraus_ops'] = [[k.real.tolist(), k.imag.tolist()] for k in self.kraus_ops]
        return res

    @staticmethod
    def from_dict(d):
        """
        Recreate a KrausModel from the dictionary representation.

        :param dict d: The dictionary representing the KrausModel. See `to_dict` for an
            example.
        :return: The deserialized KrausModel.
        :rtype: KrausModel
        """
        kraus_ops = [KrausModel.unpack_kraus_matrix(k) for k in d['kraus_ops']]
        return KrausModel(d['gate'], d['params'], d['targets'], kraus_ops, d['fidelity'])

    def __eq__(self, other):
        return isinstance(other, KrausModel) and self.to_dict() == other.to_dict()

    def __neq__(self, other):
        return not self.__eq__(other)


_NoiseModel = namedtuple("_NoiseModel", ["isa_name", "gates", "assignment_probs"])


class NoiseModel(_NoiseModel):
    """
    Encapsulate the QPU noise model containing information about the noisy gates.

    :ivar str isa_name: The name of the instruction set architecture for the QPU.
    :ivar Sequence[KrausModel] gates: The tomographic estimates of all gates.
    :ivar Dict[int,np.array] assignment_probs: The single qubit readout assignment
        probability matrices keyed by qubit id.
    """

    def to_dict(self):
        """
        Create a JSON serializable representation of the noise model.

        For example::

            {
                "isa_name": "example_qpu",
                "gates": [
                    # list of embedded dictionary representations of KrausModels here [...]
                ]
                "assignment_probs": {
                    "0": [[.8, .1],
                          [.2, .9]],
                    "1": [[.9, .4],
                          [.1, .6]],
                }
            }

        :return: A dictionary representation of self.
        :rtype: Dict[str,Any]
        """
        return {
            "isa_name": self.isa_name,
            "gates": [km.to_dict() for km in self.gates],
            "assignment_probs": {str(qid): a.tolist() for qid, a in self.assignment_probs.items()},
        }

    @staticmethod
    def from_dict(d):
        """
        Re-create the noise model from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored noise model.
        :rtype: NoiseModel
        """
        return NoiseModel(
            isa_name=d["isa_name"],
            gates=[KrausModel.from_dict(t) for t in d["gates"]],
            assignment_probs={int(qid): np.array(a) for qid, a in d["assignment_probs"].items()},
        )

    def gates_by_name(self, name):
        """
        Return all defined noisy gates of a particular gate name.

        :param str name: The gate name.
        :return: A list of noise models representing that gate.
        :rtype: Sequence[KrausModel]
        """
        return [g for g in self.gates if g.gate == name]

    def __eq__(self, other):
        return isinstance(other, NoiseModel) and self.to_dict() == other.to_dict()

    def __neq__(self, other):
        return not self.__eq__(other)


class Device(object):
    """
    A device (quantum chip) that can accept programs. Only devices that are online will actively be
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
        self.isa = ISA.from_dict(raw['isa']) if 'isa' in raw and raw['isa'] != {} else None
        self.noise_model = NoiseModel.from_dict(raw['noise_model']) if 'noise_model' in raw \
            and raw['noise_model'] != {} else None

    def is_online(self):
        """
        Whether or not the device is online and accepting new programs.

        :rtype: bool
        """
        return self._raw['is_online']

    def is_retuning(self):
        """
        Whether or not the device is currently retuning.

        :rtype: bool
        """
        return self._raw['is_retuning']

    @property
    def status(self):
        """Returns a string describing the device's status

            - **online**: The device is online and ready for use
            - **retuning** : The device is not accepting new jobs because it is re-calibrating
            - **offline**: The device is not available for use, potentially because you don't
              have the right permissions.
        """
        if self.is_online():
            return 'online'
        elif self.is_retuning():
            return 'retuning'
        else:
            return 'offline'

    def __str__(self):
        return '<Device {} {}>'.format(self.name, self.status)

    def __repr__(self):
        return str(self)
