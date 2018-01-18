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
import warnings

from six import integer_types

from pyquil.api import Job
from pyquil.quil import Program
from ._base_connection import validate_run_items, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE, get_job_id, \
    get_session, wait_for_job, post_json, get_json, parse_error


def get_devices(async_endpoint='https://job.rigetti.com/beta', api_key=None, user_id=None):
    """
    Get a list of currently available devices. The arguments for this method are the same as those for QPUConnection.
    Note that this method will only work for accounts that have QPU access.

    :return: set of online and offline devices
    :rtype: set
    """
    session = get_session(api_key, user_id)
    response = session.get(async_endpoint + '/devices')
    if response.status_code >= 400:
        raise parse_error(response)
    return {Device(name, device) for (name, device) in response.json()['devices'].items()}


class Device(object):
    """
    A device (quantum chip) that can accept programs. Only devices that are online will actively be accepting new
    programs.
    """
    def __init__(self, name, raw):
        """
        :param name: name of the device
        :param raw: raw JSON response from the server with additional information about this device
        """
        self.name = name
        self.raw = raw

    def is_online(self):
        """
        Whether or not the device is online and accepting new programs.

        :rtype: bool
        """
        return self.raw['is_online']

    def is_retuning(self):
        """
        Whether or not the device is currently retuning.

        :rtype: bool
        """
        return self.raw['is_retuning']

    @property
    def status(self):
        """Returns a string describing the device's status

            - online: The device is online and ready for use
            - retuning : The device is not accepting new jobs because it is re-calibrating
            - offline: The device is not available for use, potentially because you don't
              have the right permissions.
        """
        if self.is_online():
            return 'online'
        elif self.is_retuning():
            return 'retuning'
        else:
            return 'offline'

    def __str__(self):
        return '<Device {} {}>'.format(self.name, self.status)

    def __repr__(self):
        return str(self)


class QPUConnection(object):
    """
    Represents a connection to the QPU (Quantum Processing Unit)
    """

    def __init__(self, device_name=None, async_endpoint='https://job.rigetti.com/beta', api_key=None, user_id=None,
                 ping_time=0.1, status_time=2):
        """
        Constructor for QPUConnection. Sets up necessary security and picks a device to run on.

        :param str device_name: Name of the device to send programs too, should be one of the devices returned from
                                a call to get_devices()
        :param async_endpoint: The endpoint of the server for running QPU jobs
        :param api_key: The key to the Forest API Gateway (default behavior is to read from config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param int ping_time: Time in seconds for how long to wait between polling the server for updated status
                              information on a job.
        :param int status_time: Time in seconds for how long to wait between printing status information.
                                To disable printing of status entirely then set status_time to False.
        """
        if not device_name:
            warnings.warn("""
You created a QPUConnection without specificying a device name. This means that
your program will be sent to a random, online device. This is probably not what
you want. Instead, pass a device name to the constructor of QPUConnection:

    qpu = QPUConnection('the_name')

To get a list of available devices, use the get_devices method, for instance:

    from pyquil.api import get_devices
    for device in get_devices():
        if device.is_online():
            print('Device {} is online'.format(device.name)

Note that in order to use QPUConnection or get_devices() you must have a valid
API key with QPU access. See https://forest.rigetti.com for more details.

To suppress this warning, see Python's warning module.
""")
        self.async_endpoint = async_endpoint
        self.session = get_session(api_key, user_id)

        self.ping_time = ping_time
        self.status_time = status_time

        self.device_name = device_name

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a pyQuil program on the QPU. This functionality is in beta.

        :param Program quil_program: Quil program to run on the QPU
        :param list|range classical_addresses: Currently unused
        :param int trials: Number of shots to take
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        raise DeprecationWarning("""
The QPU does not currently support arbitrary measure operations. For now, the
only supported operation on the QPU is run_and_measure.""")

    def run_async(self, quil_program, classical_addresses, trials=1):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        # NB: Throw the same deprecation warning as in run
        return self.run(quil_program, classical_addresses, trials)

    def _run_payload(self, quil_program, classical_addresses, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": list(classical_addresses),
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        return payload

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a pyQuil program on the QPU multiple times, measuring all the qubits in the QPU
        simultaneously at the end of the program each time. This functionality is in beta.

        :param Program quil_program: A Quil program.
        :param list|range qubits: The list of qubits to measure
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)

        response = post_json(self.session, self.async_endpoint + "/job", self._wrap_program(payload))
        job = self.wait_for_job(get_job_id(response))
        return job.result()

    def run_and_measure_async(self, quil_program, qubits, trials):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = post_json(self.session, self.async_endpoint + "/job", self._wrap_program(payload))
        return get_job_id(response)

    def _run_and_measure_payload(self, quil_program, qubits, trials):
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(qubits)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT_MEASURE,
                   'qubits': list(qubits),
                   'trials': trials,
                   'quil-instructions': quil_program.out()}

        return payload

    def get_job(self, job_id):
        """
        Given a job id, return information about the status of the job

        :param str job_id: job id
        :return: Job object with the status and potentially results of the job
        :rtype: Job
        """
        response = get_json(self.session, self.async_endpoint + "/job/" + job_id)
        return Job(response.json(), 'QPU')

    def wait_for_job(self, job_id, ping_time=None, status_time=None):
        """
        Wait for the results of a job and periodically print status

        :param job_id: Job id
        :param ping_time: How often to poll the server.
                          Defaults to the value specified in the constructor. (0.1 seconds)
        :param status_time: How often to print status, set to False to never print status.
                            Defaults to the value specified in the constructor (2 seconds)
        :return: Completed Job
        """
        def get_job_fn():
            return self.get_job(job_id)
        return wait_for_job(get_job_fn,
                            ping_time if ping_time else self.ping_time,
                            status_time if status_time else self.status_time)

    def _wrap_program(self, program):
        return {
            "machine": "QPU",
            "program": program,
            "device": self.device_name
        }
