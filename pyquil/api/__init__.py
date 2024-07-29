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
"""Sub-package for facilitating connections to the QVM / QPU."""

from qcs_sdk import QCSClient, RegisterMatrixConversionError
from qcs_sdk.qpu import RawQPUReadoutData
from qcs_sdk.qpu.api import ConnectionStrategy, ExecutionOptions, ExecutionOptionsBuilder
from qcs_sdk.qvm import RawQVMReadoutData

from pyquil.api._benchmark import BenchmarkConnection
from pyquil.api._compiler import (
    AbstractCompiler,
    EncryptedProgram,
    QPUCompiler,
    QPUCompilerAPIOptions,
    QuantumExecutable,
    QVMCompiler,
)
from pyquil.api._qam import QAM, MemoryMap, QAMExecutionResult
from pyquil.api._qpu import QPU, QPUExecuteResponse
from pyquil.api._quantum_computer import (
    QuantumComputer,
    get_qc,
    list_quantum_computers,
    local_forest_runtime,
)
from pyquil.api._qvm import QVM
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.quantum_processor import QCSQuantumProcessor

__all__ = [
    "QCSClient",
    "RegisterMatrixConversionError",
    "RawQPUReadoutData",
    "RawQVMReadoutData",
    "ConnectionStrategy",
    "ExecutionOptions",
    "BenchmarkConnection",
    "AbstractCompiler",
    "EncryptedProgram",
    "QPUCompiler",
    "QPUCompilerAPIOptions",
    "QuantumExecutable",
    "QVMCompiler",
    "QAM",
    "MemoryMap",
    "QAMExecutionResult",
    "QPU",
    "QPUExecuteResponse",
    "QuantumComputer",
    "get_qc",
    "list_quantum_computers",
    "local_forest_runtime",
    "QVM",
    "WavefunctionSimulator",
    "QCSQuantumProcessor",
    "ExecutionOptionsBuilder",
]
