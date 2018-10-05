##############################################################################
# Copyright 2018 Rigetti Computing
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
from abc import ABC, abstractmethod

from rpcq.core_messages import ParameterAref

from pyquil.api._error_reporting import _record_call


class QAMError(RuntimeError):
    pass


class QAM(ABC):
    """
    The platonic ideal of this class is as a generic interface describing how a classical computer
    interacts with a live quantum computer.  Eventually, it will turn into a thin layer over the
    QPU and QVM's "QPI" interfaces.

    The reality is that neither the QPU nor the QVM currently support a full-on QPI interface,
    and so the undignified job of this class is to collect enough state that it can convincingly
    pretend to be a QPI-compliant quantum computer.
    """
    @_record_call
    def __init__(self):
        self.variables_shim = {}
        self.n_shots = None
        self.n_bits = None
        self.binary = None
        self.bitstrings = None

        self.status = 'connected'

    @_record_call
    def load(self, binary):
        """
        Initialize a QAM into a fresh state.

        :param binary: Load a compiled executable onto the QAM.
        """
        assert self.status in ['connected', 'done']

        self.variables_shim = {}
        self.binary = binary
        self.bitstrings = None
        self.status = 'loaded'
        return self

    @_record_call
    def write_memory(self, *, region_name: str, offset: int = 0, value=None):
        """
        Writes a value into a memory region on the QAM at a specified offset.

        :param region_name: Name of the declared memory region on the QAM.
        :param offset: Integer offset into the memory region to write to.
        :param value: Value to store at the indicated location.
        """
        assert self.status in ['loaded', 'done']

        aref = ParameterAref(name=region_name, index=offset)
        self.variables_shim[aref] = value

        return self

    @abstractmethod
    def run(self):
        """
        Reset the program counter on a QAM and run its loaded Quil program.
        """
        self.status = 'running'

        return self

    @_record_call
    def wait(self):
        """
        Blocks until the QPU enters the halted state.
        """
        assert self.status == 'running'
        self.status = 'done'
        return self

    @_record_call
    def read_from_memory_region(self, *, region_name: str, offsets=None):
        """
        Inspects a memory region named region_name on the QAM and extracts the values stored at
        locations indicated by offsets.

        :param region_name: The string naming the declared memory region.
        :param offsets: Either a list of offset indices or the value True for the entire region.
        :return: A list of values of the appropriate type.
        """
        assert self.status == 'done'
        if region_name != "ro":
            raise QAMError("Currently only allowed to read measurement data from ro.")
        if self.bitstrings is None:
            raise QAMError("Bitstrings have not yet been populated. Something has gone wrong.")

        if isinstance(offsets, list):
            raise ValueError("Reading out a subset of addresses is not currently supported.")
        else:
            return self.bitstrings
