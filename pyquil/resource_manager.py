#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
"""
Contains objects and functions to manage allocation and de-allocation of quantum and classical
resources in pyQuil.
"""

from itertools import count
from .quil_atom import QuilAtom
from copy import copy
from six import integer_types

class AbstractQubit(QuilAtom):
    """
    Representation of a qubit.
    """

    def index(self):
        raise NotImplementedError()


class DirectQubit(AbstractQubit):
    """
    A reference to a particularly indexed qubit.
    """

    def __init__(self, index):
        if not isinstance(index, integer_types) or index < 0:
            raise TypeError("index should be a non-negative int")
        self._index = index

    def index(self):
        return self._index

    def __repr__(self):
        return "<DirectQubit %d>" % self.index()

    def __str__(self):
        return str(self.index())

    def __eq__(self, other):
        if not isinstance(other, DirectQubit):
            raise TypeError("Can only compare DirectQubit instances with DirectQubit instances")
        return self.index() == other.index()


class Qubit(AbstractQubit):
    """
    A qubit whose index is determined by a ResourceManager.
    """

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.live = True
        self.assignment = None

    def index(self):
        """
        :returns: The index of this qubit.
        :rtype: int
        """
        if not instantiated(self):
            raise RuntimeError("Can't get the index of an unassigned qubit.")
        return self.assignment

    def __repr__(self):
        return "<Qubit %X>" % id(self)

    def __str__(self):
        return str(self.index())


def check_live_qubit(qubit):
    """
    Ensure that an object is a live qubit. Raises errors if not.

    :param qubit: A qubit, either a DirectQubit or Qubit object.
    """
    if isinstance(qubit, DirectQubit):
        pass
    elif isinstance(qubit, Qubit):
        if not qubit.live:
            raise RuntimeError("qubit is not live when it is expected to be")
    else:
        raise TypeError("qubit should be an AbstractQubit instance")


def instantiated(qubit):
    """
    Is the qubit instantiated?

    :param AbstractQubit qubit: A AbstractQubit instance.
    :return: A boolean.
    :rtype: bool
    """
    return qubit.assignment is not None


class ResourceManager(object):
    def __init__(self):
        self.live_qubits = []
        self.dead_qubits = []
        self.in_use = {}

    def reset(self):
        """
        Frees all qubits, resetting this ResourceManager.
        """
        # A copy is necessary here so that the modifications to the list don't affect the traversal.
        for qubit in copy(self.live_qubits):
            self.free_qubit(qubit)
            qubit.resource_manager = None
        self.live_qubits = []
        self.dead_qubits = []
        self.in_use = {}

    def allocate_qubit(self):
        """
        Create a new Qubit object.

        :return: A Qubit instance.
        :rtype: Qubit
        """
        new_q = Qubit(self)
        self.live_qubits.append(new_q)
        return new_q

    def free_qubit(self, q):
        """
        Free the previously allocated qubit.

        :param Qubit q: A live Qubit that was previously allocated.
        """
        if q not in self.live_qubits:
            raise RuntimeError("Qubit {} is not part of the ResourceManager".format(q))
        self.live_qubits.remove(q)
        q.live = False
        self.dead_qubits.append(q)

    def instantiate(self, qubit):
        """
        Assign a number to a qubit.

        :param Qubit qubit: A Qubit object.
        """

        def find_available_qubit(d):
            # Just do a linear search.
            for i in count(start=0, step=1):
                if not d.get(i, False):
                    return i

        if not (isinstance(qubit, Qubit)):
            raise TypeError(qubit, "should be a Qubit")

        qubit.assignment = find_available_qubit(self.in_use)
        self.in_use[qubit.assignment] = qubit

    def uninstantiate_index(self, i):
        """
        Free up an index that was previously instantiated.

        :param int i: The index.
        """
        q = self.in_use.get(i, False)
        if q and isinstance(q, Qubit):
            del self.in_use[i]


def merge_resource_managers(rm1, rm2):
    """
    Merge two resource managers into a new resource manager. All qubits in the old managers will
    point to the new one. The in_use labels of the second resource manager have priority.

    :param ResourceManager rm1: A ResourceManager.
    :param ResourceManager rm2: A ResourceManager.
    :return: A merged ResourceManager.
    :rtype: ResourceManager
    """
    rm = ResourceManager()
    rm.live_qubits = rm1.live_qubits + rm2.live_qubits
    rm.dead_qubits = rm1.dead_qubits + rm2.dead_qubits
    for q in rm.dead_qubits + rm.live_qubits:
        q.resource_manager = rm
    rm.in_use.update(rm1.in_use)
    rm.in_use.update(rm2.in_use)
    return rm
