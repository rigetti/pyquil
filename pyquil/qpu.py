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
"""
Module for facilitating connections to the QPU.
"""


class QPUConnection(object):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("""
The pyquil.qpu.QPUConnection has been moved and will be deleted in the future.
Use pyquil.api.QPUConnection instead.

Note that the new QPUConnection behaves differently:
run(), run_and_measure() all now directly return the result of the program
instead of returning a JobResult object.

This means you need to replace constructs like this:
    from pyquil.qpu import QPUConnection
    qvm = QPUConnection()
    job = qvm.run(program, ...)
    wait_for_job(job)
    result = job.result()
with just this:
    from pyquil.api import QPUConneciton
    qvm = QPUConnection()
    result = qvm.run(program, ...)

For more information see https://go.rigetti.com/connections\n""")
