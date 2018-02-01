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
from typing import Union
import numpy as np

from pyquil.parameters import Parameter
from pyquil.quilatom import unpack_qubit
from pyquil.quilbase import Gate

THETA = Parameter("theta")
"Used as the symbolic parameter in RZ, CPHASE gates."

DEFAULT_QUBIT_TYPE = "Xhalves"
DEFAULT_EDGE_TYPE = "CZ"

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

        def _maybe_configure(d, o, t):
            """
            Exclude default values from generated dictionary.

            :param dict d: The dictionary (which is also modified in place!).
            :param Union[Qubit,Edge] o: The object to serialize
            :param str t: The default value for ``o.type``.
            :return: d
            """
            # type: (dict, Union[Qubit,Edge], str) -> dict
            if o.type != t:
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
                [_maybe_configure({"qubit-id": q.id}, q, DEFAULT_QUBIT_TYPE) for q in self.qubits],
                [_maybe_configure({"action-qubits": a.targets}, a, DEFAULT_EDGE_TYPE)
                 for a in self.edges],
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
            qubits=[Qubit(id=q["qubit-id"],
                          type=q.get("type", DEFAULT_QUBIT_TYPE),
                          dead=q.get("dead", False))
                    for q in d["logical-hardware"][0]],
            edges=[Edge(targets=e["action-qubits"],
                        type=e.get("type", DEFAULT_EDGE_TYPE),
                        dead=e.get("dead", False))
                   for e in d["logical-hardware"][1]],
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
        # avoid circular imports
        from pyquil.noise import NoiseModel
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
