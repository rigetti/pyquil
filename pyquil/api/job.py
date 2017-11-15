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

import base64
import warnings

from pyquil.wavefunction import Wavefunction


class Job(object):
    def __init__(self, raw):
        self.raw = raw

    @property
    def job_id(self):
        return self.raw['jobId']

    def is_done(self):
        return self.raw['status'] in ('FINISHED', 'ERROR')

    def result(self):
        if not self.is_done():
            raise ValueError("Cannot get a result for a program that isn't completed.")

        if self.raw['program']['type'] == 'wavefunction':
            return Wavefunction.from_bit_packed_string(
                base64.b64decode(self.raw['result']), self.raw['program']['addresses'])
        else:
            return self.raw['result']

    def is_queued(self):
        return self.raw['status'] == 'QUEUED'

    def is_running(self):
        return self.raw['status'] == 'RUNNING'

    def position_in_queue(self):
        if self.is_queued():
            return int(self.raw['position_in_queue'])
        else:
            return None

    def get(self):
        warnings.warn(DeprecationWarning("""
        Running get() on a Job is now a no-op.
        To query for updated results, use .get_job(job.job_id) on a QVMConnection/QPUConnection instead
        """))

    def decode(self):
        warnings.warn(DeprecationWarning(""".decode() on a Job result is deprecated in favor of .result()"""))
        return self.result()
