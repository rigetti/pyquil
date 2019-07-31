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
                           PyQuilExecutableResponse, ParameterSpec,
                           RewriteArithmeticRequest)

from pyquil import __version__
from pyquil.api._qac import AbstractCompiler
from pyquil.api._error_reporting import _record_call
from pyquil.device import AbstractDevice
from pyquil.parser import parse_program
from pyquil.quil import Program, Measurement, Declare


_log = logging.getLogger(__name__)

PYQUIL_PROGRAM_PROPERTIES = ["native_quil_metadata", "num_shots"]


class QuilcVersionMismatch(Exception):
    pass


class QuilcNotRunning(Exception):
    pass


class QPUCompilerNotRunning(Exception):
    pass


def check_quilc_version(version_dict: Dict[str, str]):
    """
    Verify that there is no mismatch between pyquil and quilc versions.

    :param version_dict: Dictionary containing version information about quilc.
    """
    quilc_version = version_dict['quilc']
    major, minor, patch = map(int, quilc_version.split('.'))
    if major == 1 and minor < 8:
        raise QuilcVersionMismatch('Must use quilc >= 1.8.0 with pyquil >= 2.8.0, but you '
                                   f'have quilc {quilc_version} and pyquil {__version__}')


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
    def __init__(self,
                 quilc_endpoint: str,
                 qpu_compiler_endpoint: Optional[str],
                 device: AbstractDevice,
                 timeout: int = 10,
                 name: Optional[str] = None) -> None:
        """
        Client to communicate with the Compiler Server.

        :param quilc_endpoint: TCP or IPC endpoint of the Quil Compiler (quilc)
        :param qpu_compiler_endpoint: TCP or IPC endpoint of the QPU Compiler
        :param device: PyQuil Device object to use as compilation target
        :param timeout: Number of seconds to wait for a response from the client.
        :param name: Name of the lattice being targeted
        """

        if not quilc_endpoint.startswith('tcp://'):
            raise ValueError(f"PyQuil versions >= 2.4 can only talk to quilc "
                             f"versions >= 1.4 over network RPCQ.  You've supplied the "
                             f"endpoint '{quilc_endpoint}', but this doesn't look like a network "
                             f"ZeroMQ address, which has the form 'tcp://domain:port'. "
                             f"You might try clearing (or correcting) your COMPILER_URL "
                             f"environment variable and removing (or correcting) the "
                             f"compiler_server_address line from your .forest_config file.")

        self.quilc_client = Client(quilc_endpoint, timeout=timeout)
        if qpu_compiler_endpoint is not None:
            self.qpu_compiler_client = Client(qpu_compiler_endpoint, timeout=timeout)
        else:
            self.qpu_compiler_client = None
            warnings.warn("It looks like you are initializing a QPUCompiler object without a "
                          "qpu_compiler_address. If you didn't do this manually, then "
                          "you probably don't have a qpu_compiler_address entry in your "
                          "~/.forest_config file, meaning that you are not engaged to the QPU.")
        self.target_device = TargetDevice(isa=device.get_isa().to_dict(),
                                          specs=device.get_specs().to_dict())
        self.name = name

        try:
            self.connect()
        except QuilcNotRunning as e:
            warnings.warn(f'{e}. Compilation using quilc will not be available.')
        except QPUCompilerNotRunning as e:
            warnings.warn(f'{e}. Compilation using the QPU compiler will not be available.')

    def connect(self):
        self._connect_quilc()
        if self.qpu_compiler_client:
            self._connect_qpu_compiler()

    def _connect_quilc(self):
        try:
            quilc_version_dict = self.quilc_client.call('get_version_info', rpc_timeout=1)
            check_quilc_version(quilc_version_dict)
        except TimeoutError:
            raise QuilcNotRunning(f'No quilc server running at {self.quilc_client.endpoint}')

    def _connect_qpu_compiler(self):
        try:
            self.qpu_compiler_client.call('get_version_info', rpc_timeout=1)
        except TimeoutError:
            raise QPUCompilerNotRunning('No QPU compiler server running at '
                                        f'{self.qpu_compiler_client.endpoint}')

    def get_version_info(self) -> dict:
        quilc_version_info = self.quilc_client.call('get_version_info', rpc_timeout=1)
        if self.qpu_compiler_client:
            qpu_compiler_version_info = self.qpu_compiler_client.call('get_version_info',
                                                                      rpc_timeout=1)
            return {'quilc': quilc_version_info, 'qpu_compiler': qpu_compiler_version_info}
        return {'quilc': quilc_version_info}

    @_record_call
    def quil_to_native_quil(self, program: Program, *, protoquil=None) -> Program:
        self._connect_quilc()
        request = NativeQuilRequest(quil=program.out(), target_device=self.target_device)
        response = self.quilc_client.call('quil_to_native_quil', request, protoquil=protoquil).asdict()  # type: Dict
        nq_program = parse_program(response['quil'])
        nq_program.native_quil_metadata = response['metadata']
        nq_program.num_shots = program.num_shots
        return nq_program

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> Optional[BinaryExecutableResponse]:
        if not self.qpu_compiler_client:
            raise ValueError("It looks like you're trying to compile to an executable, but "
                             "do not have access to the QPU compiler endpoint. Make sure you "
                             "are engaged to the QPU before trying to do this.")

        self._connect_qpu_compiler()

        if nq_program.native_quil_metadata is None:
            warnings.warn("It looks like you're trying to call `native_quil_to_binary` on a "
                          "Program that hasn't been compiled via `quil_to_native_quil`. This is "
                          "ok if you've hand-compiled your program to our native gateset, "
                          "but be careful!")
        if self.name is not None:
            targeted_lattice = self.qpu_compiler_client.call('get_config_info')['lattice_name']
            if targeted_lattice and targeted_lattice != self.name:
                warnings.warn(f'You requested compilation for device {self.name}, '
                              f'but you are engaged on device {targeted_lattice}.')

        arithmetic_request = RewriteArithmeticRequest(quil=nq_program.out())
        arithmetic_response = self.quilc_client.call('rewrite_arithmetic', arithmetic_request)

        request = BinaryExecutableRequest(quil=arithmetic_response.quil,
                                          num_shots=nq_program.num_shots)
        response = self.qpu_compiler_client.call('native_quil_to_binary', request)

        # hack! we're storing a little extra info in the executable binary that we don't want to
        # expose to anyone outside of our own private lives: not the user, not the Forest server,
        # not anyone.
        response.recalculation_table = arithmetic_response.recalculation_table
        response.memory_descriptors = _collect_memory_descriptors(nq_program)
        response.ro_sources = _collect_classical_memory_write_locations(nq_program)
        return response


class QVMCompiler(AbstractCompiler):
    @_record_call
    def __init__(self, endpoint: str, device: AbstractDevice, timeout: float = 10) -> None:
        """
        Client to communicate with the Compiler Server.

        :param endpoint: TCP or IPC endpoint of the Compiler Server
        :param device: PyQuil Device object to use as compilation target
        """

        if not endpoint.startswith('tcp://'):
            raise ValueError(f"PyQuil versions >= 2.4 can only talk to quilc "
                             f"versions >= 1.4 over network RPCQ.  You've supplied the "
                             f"endpoint '{endpoint}', but this doesn't look like a network "
                             f"ZeroMQ address, which has the form 'tcp://domain:port'. "
                             f"You might try clearing (or correcting) your COMPILER_URL "
                             f"environment variable and removing (or correcting) the "
                             f"compiler_server_address line from your .forest_config file.")

        self.client = Client(endpoint, timeout=timeout)
        self.target_device = TargetDevice(isa=device.get_isa().to_dict(),
                                          specs=device.get_specs().to_dict())

        try:
            self.connect()
        except QuilcNotRunning as e:
            warnings.warn(f'{e}. Compilation using quilc will not be available.')

    def connect(self):
        try:
            version_dict = self.get_version_info()
            check_quilc_version(version_dict)
        except TimeoutError:
            raise QuilcNotRunning(f'No quilc server running at {self.client.endpoint}')

    def get_version_info(self) -> dict:
        return self.client.call('get_version_info', rpc_timeout=1)

    @_record_call
    def quil_to_native_quil(self, program: Program, *, protoquil=None) -> Program:
        self.connect()
        request = NativeQuilRequest(quil=program.out(), target_device=self.target_device)
        response = self.client.call('quil_to_native_quil', request, protoquil=protoquil).asdict()  # type: Dict
        nq_program = parse_program(response['quil'])
        nq_program.native_quil_metadata = response['metadata']
        nq_program.num_shots = program.num_shots
        return nq_program

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> PyQuilExecutableResponse:
        return PyQuilExecutableResponse(
            program=nq_program.out(),
            attributes=_extract_attribute_dictionary_from_program(nq_program))
