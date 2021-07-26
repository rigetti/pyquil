from contextlib import contextmanager
from dataclasses import dataclass
import dataclasses
from pyquil.quilbase import Gate
from typing import Dict, Iterator, Optional, cast

from pyquil.api._abstract_compiler import AbstractCompiler, EncryptedProgram, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, rewrite_arithmetic, _collect_memory_descriptors
from pyquil.experimental._program import ExperimentalProgram
from pyquil.quilatom import ExpressionDesignator, MemoryReference
from qcs_api_client.grpc.client import Client as GrpcClient
from qcs_api_client.grpc.models.controller import EncryptedControllerJob
from rpcq.messages import NativeQuilMetadata, ParameterAref, ParameterSpec
from pyquil.parser import parse_program, parse


@dataclass
class ExperimentalExecutable:
    job: EncryptedControllerJob

    recalculation_table: Dict[ParameterAref, ExpressionDesignator]
    """A mapping from memory references to the original gate arithmetic."""


class ExperimentalQPUCompiler:
    quantum_processor_id: str
    _timeout: Optional[int] = None

    def __init__(self, *, quantum_processor_id: str, timeout: Optional[int] = None):
        self.quantum_processor_id = quantum_processor_id
        self._timeout = timeout

    async def quil_to_native_quil(self, program: ExperimentalProgram):
        raise NotImplementedError("compilation of quil to native quil is not yet supported for ExperimentalProgram")

    async def native_quil_to_executable(self, native_quil_program: ExperimentalProgram) -> EncryptedControllerJob:
        """
        Compile the provided native quil program to an executable suitable for use on the
        experimental backend.
        """

        # TODO: Expand calibrations within the program, and then remove calibrations and unused frames and waveforms

        arithmetic_response = rewrite_arithmetic(native_quil_program)

        with self._qcs_client() as client:
            job = await client.translate_quil_to_encrypted_controller_job(
                quantum_processor_id=self.quantum_processor_id,
                quil_program=arithmetic_response.quil,
                num_shots=native_quil_program.num_shots,
            )

        return ExperimentalExecutable(
            job=job,
            recalculation_table={
                mref: _to_expression(rule) for mref, rule in arithmetic_response.recalculation_table.items()
            },
        )

    @contextmanager
    def _qcs_client(self) -> Iterator[GrpcClient]:
        client = GrpcClient(url="https://grpc.qcs.rigetti.com")
        try:
            yield client
        finally:
            client.close()


def _to_expression(rule: str) -> ExpressionDesignator:
    # We can only parse complete lines of Quil, so we wrap the arithmetic expression
    # in a valid Quil instruction to parse it.
    # TODO: This hack should be replaced after #687
    return cast(ExpressionDesignator, cast(Gate, parse(f"RZ({rule}) 0")[0]).params[0])
