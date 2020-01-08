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
Module for facilitating connections to the QVM / QPU.
"""
import warnings
from typing import Any

__all__ = [
    "QVMConnection",
    "QVMCompiler",
    "QPUCompiler",
    "Device",
    "ForestConnection",
    "pyquil_protect",
    "WavefunctionSimulator",
    "QuantumComputer",
    "list_quantum_computers",
    "get_qc",
    "local_forest_runtime",
    "local_qvm",
    "QAM",
    "QVM",
    "QPU",
    "QPUConnection",
    "BenchmarkConnection",
    "get_benchmarker",
]

from pyquil.api._base_connection import ForestConnection
from pyquil.api._benchmark import BenchmarkConnection, get_benchmarker
from pyquil.api._compiler import QVMCompiler, QPUCompiler
from pyquil.api._error_reporting import pyquil_protect
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import (
    QuantumComputer,
    list_quantum_computers,
    get_qc,
    local_forest_runtime,
    local_qvm,
)
from pyquil.api._qvm import QVMConnection, QVM
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.device._main import Device


class QPUConnection(QPU):
    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "QPUConnection's semantics have changed for Forest 2. Consider using "
            "pyquil.get_qc('...') instead of creating this object directly. "
            "Please consult the migration guide for full details.",
            DeprecationWarning,
        )
        super(QPU, self).__init__(*args, **kwargs)
