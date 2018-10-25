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
import logging

import warnings
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from rpcq import Client
from rpcq.messages import (BinaryExecutableRequest, BinaryExecutableResponse,
                           NativeQuilRequest, TargetDevice,
                           PyQuilExecutableResponse, ParameterSpec)

from pyquil.api._base_connection import ForestConnection
from pyquil.api._qac import AbstractCompiler
from pyquil.api._error_reporting import _record_call
from pyquil.device import AbstractDevice
from pyquil.parser import parse_program
from pyquil.quil import Program, Measurement, Declare


_log = logging.getLogger(__name__)

PYQUIL_PROGRAM_PROPERTIES = ["native_quil_metadata", "num_shots"]


def _extract_attribute_dictionary_from_program(program: Program) -> Dict[str, Any]:
    """
    Collects the attributes from PYQUIL_PROGRAM_PROPERTIES on the Program object program
    into a dictionary.

    :param program: Program to collect attributes from.
    :return: Dictionary of attributes, keyed on the string attribute name.
    """
    attrs = {}
    for prop in PYQUIL_PROGRAM_PROPERTIES:
        attrs[prop] = getattr(program, prop)
    return attrs


def _extract_program_from_pyquil_executable_response(response: PyQuilExecutableResponse) -> Program:
    """
    Unpacks a rpcq PyQuilExecutableResponse object into a pyQuil Program object.

    :param response: PyQuilExecutableResponse object to be unpacked.
    :return: Resulting pyQuil Program object.
    """
    p = Program(response.program)
    for attr, val in response.attributes.items():
        setattr(p, attr, val)
    return p


def _collect_classical_memory_write_locations(program: Program) -> List[Optional[Tuple[int, int]]]:
    """Collect classical memory locations that are the destination of MEASURE instructions

    These locations are important for munging output buffers returned from the QPU
    server to the shape expected by the user.

    This is secretly stored on BinaryExecutableResponse. We're careful to make sure
    these objects are json serializable.

    :return: list whose value `(q, m)` at index `addr` records that the `m`-th measurement of
        qubit `q` was measured into `ro` address `addr`. A value of `None` means nothing was
        measured into `ro` address `addr`.
    """
    ro_size = None
    for instr in program:
        if isinstance(instr, Declare) and instr.name == "ro":
            if ro_size is not None:
                raise ValueError("I found multiple places where a register named `ro` is declared! "
                                 "Please only declare one register named `ro`.")
            ro_size = instr.memory_size

    measures_by_qubit: Dict[int, int] = Counter()
    ro_sources: Dict[int, Tuple[int, int]] = {}

    for instr in program:
        if isinstance(instr, Measurement):
            q = instr.qubit.index
            if instr.classical_reg:
                offset = instr.classical_reg.offset
                assert instr.classical_reg.name == "ro", instr.classical_reg.name
                if offset in ro_sources:
                    _log.warning(f"Overwriting the measured result in register "
                                 f"{instr.classical_reg} from qubit {ro_sources[offset]} "
                                 f"to qubit {q}")
                # we track how often each qubit is measured (per shot) and into which register it is
                # measured in its n-th measurement.
                ro_sources[offset] = (q, measures_by_qubit[q])
            measures_by_qubit[q] += 1
    if ro_size:
        return [ro_sources.get(i) for i in range(ro_size)]
    elif ro_sources:
        raise ValueError("Found MEASURE instructions, but no 'ro' or 'ro_table' "
                         "region was declared.")
    else:
        return []


def _collect_memory_descriptors(program: Program) -> Dict[str, ParameterSpec]:
    """Collect Declare instructions that are important for building the patch table.

    This is secretly stored on BinaryExecutableResponse. We're careful to make sure
    these objects are json serializable.

    :return: A dictionary of variable names to specs about the declared region.
    """
    return {
        instr.name: ParameterSpec(type=instr.memory_type, length=instr.memory_size)
        for instr in program if isinstance(instr, Declare)
    }


class QPUCompiler(AbstractCompiler):
    @_record_call
    def __init__(self, endpoint: str, device: AbstractDevice) -> None:
        """
        Client to communicate with the Compiler Server.

        :param endpoint: TCP or IPC endpoint of the Compiler Server
        :param device: PyQuil Device object to use as compilation target
        """

        self.client = Client(endpoint)
        self.target_device = TargetDevice(isa=device.get_isa().to_dict(),
                                          specs=device.get_specs().to_dict())

    def get_version_info(self) -> dict:
        return self.client.call('get_version_info')

    @_record_call
    def quil_to_native_quil(self, program: Program) -> Program:
        request = NativeQuilRequest(quil=program.out(), target_device=self.target_device)
        response = self.client.call('quil_to_native_quil', request).asdict()  # type: Dict
        nq_program = parse_program(response['quil'])
        nq_program.native_quil_metadata = response['metadata']
        nq_program.num_shots = program.num_shots
        return nq_program

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> BinaryExecutableResponse:
        if nq_program.native_quil_metadata is None:
            warnings.warn("It looks like you're trying to call `native_quil_to_binary` on a "
                          "Program that hasn't been compiled via `quil_to_native_quil`. This is "
                          "ok if you've hand-compiled your program to our native gateset, "
                          "but be careful!")

        request = BinaryExecutableRequest(quil=nq_program.out(), num_shots=nq_program.num_shots)
        response = self.client.call('native_quil_to_binary', request)
        # hack! we're storing a little extra info in the executable binary that we don't want to
        # expose to anyone outside of our own private lives: not the user, not the Forest server,
        # not anyone.
        response.memory_descriptors = _collect_memory_descriptors(nq_program)
        response.ro_sources = _collect_classical_memory_write_locations(nq_program)
        return response


class QVMCompiler(AbstractCompiler):
    @_record_call
    def __init__(self, endpoint: str, device: AbstractDevice) -> None:
        """
        Client to communicate with the Compiler Server.

        :param endpoint: TCP or IPC endpoint of the Compiler Server
        :param device: PyQuil Device object to use as compilation target
        """
        self.client = Client(endpoint)
        self.target_device = TargetDevice(isa=device.get_isa().to_dict(),
                                          specs=device.get_specs().to_dict())

    def get_version_info(self) -> dict:
        return self.client.call('get_version_info')

    @_record_call
    def quil_to_native_quil(self, program: Program) -> Program:
        request = NativeQuilRequest(quil=program.out(), target_device=self.target_device)
        response = self.client.call('quil_to_native_quil', request).asdict()  # type: Dict
        nq_program = parse_program(response['quil'])
        nq_program.native_quil_metadata = response['metadata']
        nq_program.num_shots = program.num_shots
        return nq_program

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> PyQuilExecutableResponse:
        return PyQuilExecutableResponse(
            program=nq_program.out(),
            attributes=_extract_attribute_dictionary_from_program(nq_program))


class LocalQVMCompiler(AbstractCompiler):
    def __init__(self, endpoint: str, device: AbstractDevice) -> None:
        """
        Client to communicate with a locally executing quilc instance.

        :param endpoint: HTTP endpoint of the quilc instance.
        :param device: PyQuil Device object to use as the compilation target.
        """
        self.endpoint = endpoint
        self.isa = device.get_isa()
        self.specs = device.get_specs()

        self._connection = ForestConnection(sync_endpoint=endpoint)
        self.session = self._connection.session  # backwards compatibility

    def get_version_info(self) -> dict:
        return self._connection._quilc_get_version_info()

    def quil_to_native_quil(self, program: Program) -> Program:
        response = self._connection._quilc_compile(program, self.isa, self.specs)

        compiled_program = Program(response['compiled-quil'])
        compiled_program.native_quil_metadata = response['metadata']
        compiled_program.num_shots = program.num_shots

        return compiled_program

    def native_quil_to_executable(self, nq_program: Program):
        return PyQuilExecutableResponse(
            program=nq_program.out(),
            attributes=_extract_attribute_dictionary_from_program(nq_program))
