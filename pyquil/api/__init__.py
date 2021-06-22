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
Sub-package for facilitating connections to the QVM / QPU.
"""

__all__ = [
    "AbstractCompiler",
    "BenchmarkConnection",
    "EncryptedProgram",
    "EngagementManager",
    "get_qc",
    "list_quantum_computers",
    "local_forest_runtime",
    "QAM",
    "QAMExecutionResult",
    "QCSClientConfiguration",
    "QCSQuantumProcessor",
    "QPU",
    "QPUCompiler",
    "QuantumComputer",
    "QuantumExecutable",
    "QVM",
    "QVMCompiler",
    "WavefunctionSimulator",
]

from qcs_api_client.client import QCSClientConfiguration

from pyquil.api._benchmark import BenchmarkConnection
from pyquil.api._compiler import QVMCompiler, QPUCompiler, QuantumExecutable, EncryptedProgram, AbstractCompiler
from pyquil.api._engagement_manager import EngagementManager
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import (
    QuantumComputer,
    list_quantum_computers,
    get_qc,
    local_forest_runtime,
)
from pyquil.api._qvm import QVM
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.quantum_processor import QCSQuantumProcessor
