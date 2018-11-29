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

__all__ = ['QVMConnection', 'LocalQVMCompiler', 'QVMCompiler', 'QPUCompiler',
           'Job', 'Device', 'ForestConnection', 'pyquil_protect',
           'WavefunctionSimulator', 'QuantumComputer', 'list_quantum_computers', 'get_qc',
           'QAM', 'QVM', 'QPU', 'QPUConnection',
           'BenchmarkConnection', 'LocalBenchmarkConnection', 'get_benchmarker']

from pyquil.api._base_connection import ForestConnection
from pyquil.api._benchmark import BenchmarkConnection, LocalBenchmarkConnection, get_benchmarker
from pyquil.api._compiler import QVMCompiler, QPUCompiler, LocalQVMCompiler
from pyquil.api._error_reporting import pyquil_protect
from pyquil.api._job import Job
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer, list_quantum_computers, get_qc, local_qvm
from pyquil.api._qvm import QVMConnection, QVM
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.device import Device


class SyncConnection(QVMConnection):
    def __init__(self, *args, **kwargs):
        warnings.warn("SyncConnection has been renamed to QVMConnection and will be removed in the future",
                      stacklevel=2)
        super(SyncConnection, self).__init__(*args, **kwargs)


class JobConnection(object):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("""
JobConnection has been deprecated and will be removed in a future version.
Use QVMConnection instead.

Note that QVMConnection behaves differently than JobConnection did:
run(), run_and_measure(), wavefunction(), and expectation() all now directly
return the result of the program instead of returning a JobResult object.

This means you need to replace constructs like this:
    qvm = JobConnection()
    job = qvm.run(program, ...)
    wait_for_job(job)
    result = job.result()
with just this:
    qvm = JobConnection()
    result = qvm.run(program, ...)

For more information see https://go.rigetti.com/connections\n""")


class QPUConnection(QPU):
    def __init__(self, *args, **kwargs):
        warnings.warn("QPUConnection's semantics have changed for Forest 2. Consider using "
                      "pyquil.get_qc('...') instead of creating this object directly. "
                      "Please consult the migration guide for full details.",
                      DeprecationWarning)
        super(QPU, self).__init__(*args, **kwargs)
