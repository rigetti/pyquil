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

from pyquil.api.job import Job
from pyquil.device import Device, ISA, Specs
from pyquil.quil import Program
from pyquil.parser import parse_program
from ._base_connection import TYPE_MULTISHOT, get_job_id, get_session, \
    wait_for_job, post_json, get_json


class CompilerConnection(object):
    """
    Represents a connection to the Quil compiler.
    """

    def __init__(self, device=None, sync_endpoint='https://api.rigetti.com',
                 async_endpoint='https://job.rigetti.com/beta', api_key=None,
                 user_id=None, use_queue=False, ping_time=0.1, status_time=2,
                 isa_source=None, specs_source=None):
        """
        Constructor for CompilerConnection. Sets up any necessary security.

        :param Device device: A Device object to pull the ISA and Specs from.
        :param sync_endpoint: The endpoint of the server for running small jobs
        :param async_endpoint: The endpoint of the server for running large jobs
        :param api_key: The key to the Forest API Gateway (default behavior is
                        to read from config file)
        :param user_id: Your userid for Forest (default behavior is to read from
                        config file)
        :param bool use_queue: Disabling this parameter may improve performance
                               for small, quick programs. To support larger
                               programs, set it to True. (default: False)
                               NOTE: *_async methods will always use the queue.
                               See https://go.rigetti.com/connections for more
                               information.
        :param int ping_time: Time in seconds for how long to wait between
                              polling the server for updated status information
                              on a job. Note that this parameter doesn't matter
                              if use_queue is False.
        :param int status_time: Time in seconds for how long to wait between
                                printing status information. To disable printing
                                of status entirely then set status_time to
                                False. Note that this parameter doesn't matter
                                if use_queue is False.
        :param ISA isa_source: An ISA object to compile against (overwrites device ISA).
        :param Specs specs_source: A Specs object for program fidelity (overwrites device Specs).
        """
        self.async_endpoint = async_endpoint
        self.sync_endpoint = sync_endpoint
        self.session = get_session(api_key, user_id)

        self.use_queue = use_queue
        self.ping_time = ping_time
        self.status_time = status_time

        self.specs = None
        if device is None and isa_source is None:
            raise ValueError('Must provide at least one of device and isa_source arguments.')

        if isinstance(device, Device):
            self.custom_isa = device.isa
            self.specs = device.specs
        elif device is not None:
            raise TypeError('device argument must be a Device.')

        # this will overwrite the ISA from device if both are provided
        if isinstance(isa_source, ISA):
            self.custom_isa = isa_source
        elif isa_source is not None:
            raise TypeError('isa_source argument must be an ISA.')

        # this will overwrite the specs from device if both are provided
        if isinstance(specs_source, Specs):
            self.specs = specs_source
        elif specs_source is not None:
            raise TypeError('specs_source argument must be a Specs.')

    def compile(self, quil_program, priority=0):
        """
        Sends a Quil program to the Forest compiler and returns the resulting
        compiled Program.

        :param Program quil_program: Quil program to be compiled.
        :param ISA isa: ISA to target.
        :param int priority: Sets a desired priority for the job. Larger numbers are higher priority, default is 0 (highest priority available to average user).
        :returns: The compiled Program object.
        :rtype: Program
        """
        payload = self._compile_payload(quil_program)
        if self.use_queue:
            response = post_json(self.session, self.async_endpoint + "/job",
                                 {"machine": "QUILC", "program": payload, "priority": priority})
            job = self.wait_for_job(get_job_id(response))
            return job.compiled_quil()
        else:
            response = post_json(self.session, self.sync_endpoint + "/quilc",
                                 payload)
            return parse_program(response.json()['compiled-quil'])

    def compile_async(self, quil_program, priority=0):
        """
        Similar to compile except that it returns a job id and doesn't wait for
        the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._compile_payload(quil_program)
        response = post_json(self.session, self.async_endpoint + "/job",
                             {"machine": "QUILC", "program": payload, "priority": priority})
        return get_job_id(response)

    def _compile_payload(self, quil_program):
        payload = {"type": TYPE_MULTISHOT,
                   "qubits": [],
                   "uncompiled-quil": quil_program.out(),
                   "target-device": {"isa": self.custom_isa.to_dict()}}

        if self.specs is not None:
            payload["target-device"]["specs"] = self.specs.to_dict()

        return payload

    def get_job(self, job_id):
        """
        Given a job id, return information about the status of the job

        :param str job_id: job id
        :return: Job object with the status and potentially results of the job
        :rtype: Job
        """
        response = get_json(self.session, self.async_endpoint + "/job/" + job_id)
        return Job(response.json(), 'QUILC')

    def wait_for_job(self, job_id, ping_time=None, status_time=None):
        """
        Wait for the results of a job and periodically print status

        :param job_id: Job id
        :param ping_time: How often to poll the server.
                          Defaults to the value specified in the constructor.
                          (0.1 seconds)
        :param status_time: How often to print status, set to False to never
                            print status.
                            Defaults to the value specified in the constructor
                            (2 seconds)
        :return: Completed Job
        """
        def get_job_fn():
            return self.get_job(job_id)
        return wait_for_job(get_job_fn,
                            ping_time if ping_time else self.ping_time,
                            status_time if status_time else self.status_time)
