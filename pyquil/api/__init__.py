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

__all__ = ['SyncConnection', 'QVMConnection', 'QPUConnection']

from pyquil.api.job import Job
from pyquil.api.qvm import QVMSyncConnection as SyncConnection
from pyquil.api.qvm import QVMConnection
from pyquil.api.qpu import QPUConnection


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
