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
    """
    Represents the current status of a Job in the Forest queue.

    Job statuses are initially QUEUED when QVM/QPU resources are not available
    They transition to RUNNING when they have been started
    Finally they are marked as FINISHED or ERROR once completed
    """
    def __init__(self, raw):
        self.raw = raw

    @property
    def job_id(self):
        """
        Job id
        :rtype: str
        """
        return self.raw['jobId']

    def is_done(self):
        """
        Has the job completed yet?
        """
        return self.raw['status'] in ('FINISHED', 'ERROR')

    def result(self):
        """
        The result of the job if available, ValueError otherwise
        """
        if not self.is_done():
            raise ValueError("Cannot get a result for a program that isn't completed.")

        if self.raw['program']['type'] == 'wavefunction':
            return Wavefunction.from_bit_packed_string(
                base64.b64decode(self.raw['result']), self.raw['program']['addresses'])
        else:
            return self.raw['result']

    def is_queued(self):
        """
        Is the job still in the Forest queue?
        """
        return self.raw['status'] == 'QUEUED'

    def is_running(self):
        """
        Is the job currently running?
        """
        return self.raw['status'] == 'RUNNING'

    def position_in_queue(self):
        """
        If the job is queued, this will return how many other jobs are ahead of it.
        If the job is not queued, this will return None
        """
        if self.is_queued():
            # TODO: Remove once name of field is changed on server
            if 'position_in_queue' in self.raw:
                return int(self.raw['position_in_queue'])
            elif 'positionInQueue' in self.raw:
                return int(self.raw['positionInQueue'])

    def get(self):
        warnings.warn("""
        Running get() on a Job is now a no-op.
        To query for updated results, use .get_job(job.job_id) on a QVMConnection/QPUConnection instead
        """, stacklevel=2)

    def decode(self):
        warnings.warn(""".decode() on a Job result is deprecated in favor of .result()""", stacklevel=2)
        return self.result()
