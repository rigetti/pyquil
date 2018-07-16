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
import time
import warnings
from typing import Union

import numpy as np
from six import integer_types

from pyquil.api import errors
from pyquil.api._qam import QAM
from pyquil.api.job import Job
from pyquil.device import Device
from pyquil.gates import MEASURE
from pyquil.quil import Program, get_classical_addresses_from_program
from ._base_connection import (validate_run_items, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE,
                               get_job_id, get_session, wait_for_job, post_json, get_json,
                               parse_error, ASYNC_ENDPOINT, ForestConnection)


def get_devices(async_endpoint=ASYNC_ENDPOINT, api_key=None, user_id=None,
                as_dict=False):
    """
    Get a list of currently available devices. The arguments for this method are the same as those for QPUConnection.
    Note that this method will only work for accounts that have QPU access.

    :return: Set or Dictionary (keyed by device name) of all available devices.
    :rtype: Set|Dict
    """
    session = get_session(api_key, user_id)
    response = session.get(async_endpoint + '/devices')
    if response.status_code >= 400:
        raise parse_error(response)

    if not as_dict:
        warnings.warn("""
Warning: The return type Set for get_devices() is being deprecated for Dict. This will eventually
return the following:

    get_devices()
    # {'19Q-Acorn': <Device 19Q-Acorn online>, '8Q-Agave': <Device 8Q-Agave offline>}
    acorn = get_devices()['19Q-Acorn']

To use this Dict return type now, you may optionally pass the flag get_devices(as_dict=True). This
will become the default behavior in a future pyQuil release.
""", DeprecationWarning, stacklevel=2)
        return {Device(name, device) for (name, device) in response.json()['devices'].items()}

    return {name: Device(name, device) for (name, device) in response.json()['devices'].items()}


def append_measures_to_program(gate_program, qubits):
    """
    For run_and_measure programs, append MEASURE instructions to the
    program, on all provided qubits.

    :param Program gate_program: Program without MEASURE instructions
    :param list qubits: Qubits to measure
    :return: Full pyquil program with MEASUREs
    :rtype: Program
    """
    meas_program = Program([MEASURE(q, q) for q in qubits])
    return gate_program + meas_program


class QPUConnection(object):
    """
    Represents a connection to the QPU (Quantum Processing Unit)
    """

    def __init__(self, device=None, async_endpoint=ASYNC_ENDPOINT, api_key=None,
                 user_id=None, ping_time=0.1, status_time=2, device_name=None):
        """
        Constructor for QPUConnection. Sets up necessary security and picks a device to run on.

        :param Device device: The device to send programs to. It should be one of the values in the
                              dictionary returned from get_devices().
        :param async_endpoint: The endpoint of the server for running QPU jobs
        :param api_key: The key to the Forest API Gateway (default behavior is to read from config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param int ping_time: Time in seconds for how long to wait between polling the server for updated status
                              information on a job.
        :param int status_time: Time in seconds for how long to wait between printing status information.
                                To disable printing of status entirely then set status_time to False.
        """
        if isinstance(device, Device):
            device_dot_name = device.name
        elif isinstance(device, str):
            device_dot_name = device
        else:
            device_dot_name = None

        if device_dot_name is None and device_name is None:
            warnings.warn("""
You created a QPUConnection without specifying a device name. This means that
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

        if device_name is not None:
            warnings.warn("""
Warning: The keyword argument device_name is being deprecated in favor of the keyword argument
device, which may take either a Device object or a string. For example:

    acorn = get_devices(as_dict=True)['19Q-Acorn']
    # Alternative, correct implementations
    qpu = QPUConnection(device=acorn)
    qpu = QPUConnection(device='19Q-Acorn')
    qpu = QPUConnection(acorn)
    qpu = QPUConnection('19Q-Acorn')

The device_name kwarg implementation, qpu = QPUConnection(device_name='19Q-Acorn'), will eventually
be removed in a future release of pyQuil.
""", DeprecationWarning, stacklevel=2)

        if device_dot_name and device_name is not None:
            warnings.warn("""
Warning: You have supplied both a device ({}) and a device_name ({}). The QPU is being initialized
with the former, the device.
""".format(str(device), device_name))

        if device_dot_name is not None:
            self.device_name = device_dot_name
        elif device_name is not None:
            self.device_name = device_name
        else:
            self.device_name = None

        self.async_endpoint = async_endpoint

        self.ping_time = ping_time
        self.status_time = status_time

        self._connection = ForestConnection(sync_endpoint=None, async_endpoint=async_endpoint,
                                            api_key=api_key, user_id=user_id, use_queue=True,
                                            ping_time=ping_time, status_time=status_time)
        self.session = self._connection.session  # backwards compatibility

    def run(self, quil_program, classical_addresses=None, trials=1, needs_compilation=True, isa=None):
        """
        Run a pyQuil program on the QPU and return the values stored in the classical registers
        designated by the classical_addresses parameter. The program is repeated according to
        the number of trials provided to the run method. This functionality is in beta.

        It is important to note that our QPUs currently only allow a single set of simultaneous
        readout pulses on all qubits in the QPU at the end of the program. This means that
        missing or duplicate MEASURE instructions do not change the pulse program, but instead
        only contribute to making a less rich or richer mapping, respectively, between classical
        and qubit addresses.

        :param Program quil_program: Pyquil program to run on the QPU
        :param list|range classical_addresses: Classical register addresses to return
        :param int trials: Number of times to run the program (a.k.a. number of shots)
        :param bool needs_compilation: If True, preprocesses the job with the compiler.
        :param ISA isa: If set, specifies a custom ISA to compile to. If left unset,
                    Forest uses the default ISA associated to this QPU device.
        :return: A list of a list of classical registers (each register contains a bit)
        :rtype: list
        """
        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(quil_program)

        return self._connection._qpu_run(quil_program, classical_addresses, trials,
                                         needs_compilation, isa, device_name=self.device_name)

    def run_async(self, quil_program, classical_addresses=None, trials=1, needs_compilation=True, isa=None):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to
        be executed. See https://go.rigetti.com/connections for reasons to use this method.
        """
        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(quil_program)

        return self._connection._qpu_run_async(quil_program, classical_addresses, trials,
                                               needs_compilation, isa, device_name=self.device_name)

    def run_and_measure(self, quil_program, qubits, trials=1, needs_compilation=True, isa=None):
        """
        Similar to run, except for how MEASURE operations are dealt with. With run, users are
        expected to include MEASURE operations in the program if they want results back. With
        run_and_measure, users provide a pyquil program that does not have MEASURE instructions,
        and also provide a list of qubits to measure. All qubits in this list will be measured
        at the end of the program, and their results stored in corresponding classical registers.

        :param Program quil_program: Pyquil program to run on the QPU
        :param list|range qubits: The list of qubits to measure
        :param int trials: Number of times to run the program (a.k.a. number of shots)
        :param bool needs_compilation: If True, preprocesses the job with the compiler.
        :param ISA isa: If set, specifies a custom ISA to compile to. If left unset,
                    Forest uses the default ISA associated to this QPU device.
        :return: A list of a list of classical registers (each register contains a bit)
        :rtype: list
        """
        job = self.wait_for_job(self.run_and_measure_async(quil_program, qubits, trials, needs_compilation, isa))
        return job.result()

    def run_and_measure_async(self, quil_program, qubits, trials, needs_compilation=True, isa=None):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the program
        to be executed. See https://go.rigetti.com/connections for reasons to use this method.
        """
        full_program = append_measures_to_program(quil_program, qubits)
        payload = self._run_and_measure_payload(full_program, qubits, trials, needs_compilation=needs_compilation, isa=isa)
        response = post_json(self.session, self.async_endpoint + "/job", self._wrap_program(payload))
        return get_job_id(response)

    def _run_and_measure_payload(self, quil_program, qubits, trials, needs_compilation, isa):
        # Developer note: Don't migrate this code to `ForestConnection`. The QPU run_and_measure
        # web endpoint is deprecated. If run_and_measure-type functionality is desired,
        # the client (ie PyQuil) should add measure instructions and hit the `run` endpoint. See
        # `QuantumComputer.run_and_measure` for an example.
        if not quil_program:
            raise ValueError("You have attempted to run an empty program."
                             " Please provide gates or measure instructions to your program.")

        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(qubits)
        if not isinstance(trials, integer_types):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT_MEASURE,
                   'qubits': list(qubits),
                   'trials': trials}

        if needs_compilation:
            payload['uncompiled-quil'] = quil_program.out()
            if isa:
                payload['target-device'] = {"isa": isa.to_dict()}
        else:
            payload['compiled-quil'] = quil_program.out()

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
        return self._connection._wait_for_job(job_id, 'QPU', ping_time, status_time)

    def _wrap_program(self, program):
        return {
            "machine": "QPU",
            "program": program,
            "device": self.device_name
        }


class QPU(QAM):
    def __init__(self, connection: ForestConnection, device: Union[Device, str] = None):
        """
        A physical quantum device that can run Quil programs.

        :param connection: A connection to the Forest web API.
        :param device: The name of the device to send programs to. It can be either the
            string device name or a :py:class:`Device` object, from whence the name
            will be extracted.
        """
        if isinstance(device, Device):
            device_dot_name = device.name
        elif isinstance(device, str):
            device_dot_name = device
        else:
            raise ValueError("Unknown device {}".format(device))

        self.device_name = device_dot_name
        self.connection = connection

    def run(self, quil_program, classical_addresses, trials):
        """
        Run a pyQuil program on the QPU and return the values stored in the classical registers
        designated by the classical_addresses parameter. The program is repeated according to
        the number of trials provided to the run method. This functionality is in beta.

        It is important to note that our QPUs currently only allow a single set of simultaneous
        readout pulses on all qubits in the QPU at the end of the program. This means that
        missing or duplicate MEASURE instructions do not change the pulse program, but instead
        only contribute to making a less rich or richer mapping, respectively, between classical
        and qubit addresses.

        :param Program quil_program: Pyquil program to run on the QPU
        :param list|range classical_addresses: Classical register addresses to return
        :param int trials: Number of times to run the program (a.k.a. number of shots)
        :return: A list of a list of classical registers (each register contains a bit)
        :rtype: list
        """
        return np.asarray(self.connection._qpu_run(quil_program=quil_program,
                                                   classical_addresses=classical_addresses,
                                                   trials=trials, needs_compilation=False, isa=None,
                                                   device_name=self.device_name))

    def run_async(self, quil_program, classical_addresses, trials):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to
        be executed. See https://go.rigetti.com/connections for reasons to use this method.
        """
        return self.connection._qpu_run_async(quil_program=quil_program,
                                              classical_addresses=classical_addresses,
                                              trials=trials, needs_compilation=False, isa=None,
                                              device_name=self.device_name)

    def wait_for_job(self, job_id, ping_time=None, status_time=None):
        return self.connection._wait_for_job(job_id=job_id, ping_time=ping_time,
                                             status_time=status_time, machine='QPU')
