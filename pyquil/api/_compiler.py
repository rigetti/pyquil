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
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Iterator
import asyncio

import httpx
from pyquil.parser import parse_program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare
import qcs_sdk
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.operations.sync import (
    get_quilt_calibrations,
)
from rpcq.messages import ParameterSpec

from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable, EncryptedProgram
from pyquil.api._qcs_client import qcs_client
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program


class QPUCompilerNotRunning(Exception):
    pass


def parse_mref(val: str) -> MemoryReference:
    """Parse a memory reference from its string representation."""
    val = val.strip()
    try:
        if val[-1] == "]":
            name, offset = val.split("[")
            return MemoryReference(name, int(offset[:-1]))
        else:
            return MemoryReference(val)
    except Exception:
        raise ValueError(f"Unable to parse memory reference {val}.")


def _collect_memory_descriptors(program: Program) -> Dict[str, ParameterSpec]:
    """Collect Declare instructions that are important for building the patch table.

    :return: A dictionary of variable names to specs about the declared region.
    """

    return {
        instr.name: ParameterSpec(type=instr.memory_type, length=instr.memory_size)
        for instr in program
        if isinstance(instr, Declare)
    }


class QPUCompiler(AbstractCompiler):
    """
    Client to communicate with the compiler and translation service.
    """

    def __init__(
        self,
        *,
        quantum_processor_id: str,
        quantum_processor: AbstractQuantumProcessor,
        timeout: float = 10.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """
        Instantiate a new QPU compiler client.

        :param quantum_processor_id: Processor to target.
        :param quantum_processor: Quantum processor to use as compilation target.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """
        super().__init__(
            quantum_processor=quantum_processor,
            timeout=timeout,
            client_configuration=client_configuration,
            event_loop=event_loop,
        )

        self.quantum_processor_id = quantum_processor_id
        self._calibration_program: Optional[Program] = None

    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        """
        Convert a native Quil program into an executable binary which can be executed by a QPU.
        """
        rewrite_response = qcs_sdk.rewrite_arithmetic(nq_program.out())

        # This is a work-around needed because calling `qcs_sdk.translate` happens _before_
        # the event loop is available. Wrapping it in a Python async function ensures that
        # the event loop is available. This is a limitation of pyo3:
        # https://pyo3.rs/v0.17.1/ecosystem/async-await.html#a-note-about-asynciorun
        async def _translate(*args) -> qcs_sdk.TranslationResult:  # type: ignore
            return await qcs_sdk.translate(*args)

        translated_program = self._event_loop.run_until_complete(
            _translate(
                rewrite_response["program"],
                nq_program.num_shots,
                self.quantum_processor_id,
            )
        )

        return EncryptedProgram(
            program=translated_program["program"],
            memory_descriptors=_collect_memory_descriptors(nq_program),
            ro_sources={parse_mref(mref): source for mref, source in translated_program["ro_sources"] or []},
            recalculation_table=rewrite_response["recalculation_table"],
            _memory=nq_program._memory.copy(),
        )

    def _fetch_calibration_program(self) -> Program:
        with self._qcs_client() as qcs_client:  # type: httpx.Client
            response = get_quilt_calibrations(client=qcs_client, quantum_processor_id=self.quantum_processor_id).parsed
        return parse_program(response.quilt)

    def get_calibration_program(self, force_refresh: bool = False) -> Program:
        """
        Get the Quil-T calibration program associated with the underlying QPU.

        This will return a cached copy of the calibration program if present.
        Otherwise (or if forcing a refresh), it will fetch and store the
        calibration program from the QCS API.

        A calibration program contains a number of DEFCAL, DEFWAVEFORM, and
        DEFFRAME instructions. In sum, those instructions describe how a Quil-T
        program should be translated into analog instructions for execution on
        hardware.

        :param force_refresh: Whether or not to fetch a new calibration program before returning.
        :returns: A Program object containing the calibration definitions."""
        if force_refresh or self._calibration_program is None:
            try:
                self._calibration_program = self._fetch_calibration_program()
            except Exception as ex:
                raise RuntimeError("Could not fetch calibrations") from ex

        assert self._calibration_program is not None
        return self._calibration_program

    def reset(self) -> None:
        """
        Reset the state of the QPUCompiler.
        """
        super().reset()
        self._calibration_program = None

    @contextmanager
    def _qcs_client(self) -> Iterator[httpx.Client]:
        with qcs_client(
            client_configuration=self._client_configuration, request_timeout=self._timeout
        ) as client:  # type: httpx.Client
            yield client


class QVMCompiler(AbstractCompiler):
    """
    Client to communicate with the compiler.
    """

    def __init__(
        self,
        *,
        quantum_processor: AbstractQuantumProcessor,
        timeout: float = 10.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """
        Client to communicate with compiler.

        :param quantum_processor: Quantum processor to use as compilation target.
        :param timeout: Number of seconds to wait for a response from the client.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """
        super().__init__(
            quantum_processor=quantum_processor,
            timeout=timeout,
            client_configuration=client_configuration,
            event_loop=event_loop,
        )

    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        return nq_program
