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
from pyquil.paulis import PauliTerm
from ._base_connection import TYPE_MULTISHOT, get_job_id, get_session, \
    wait_for_job, post_json, get_json


class CompilerConnectionError(ValueError):
    def __init__(self):
        super().__init__("You must provide at least either device or isa_source when constructing the compiler"
                         " connection to use this method")


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
        self.custom_isa = None

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

    def compile(self, quil_program):
        """
        Sends a Quil program to the Forest compiler and returns the resulting
        compiled Program.

        :param Program quil_program: Quil program to be compiled.
        :returns: The compiled Program object.
        :rtype: Program
        """
        if self.specs is None and self.custom_isa is None:
            raise CompilerConnectionError()

        payload = self._compile_payload(quil_program)
        if self.use_queue:
            response = post_json(self.session, self.async_endpoint + "/job",
                                 {"machine": "QUILC", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.compiled_quil()
        else:
            response = post_json(self.session, self.sync_endpoint + "/quilc",
                                 payload)
            return parse_program(response.json()['compiled-quil'])

    def compile_async(self, quil_program):
        """
        Similar to compile except that it returns a job id and doesn't wait for
        the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if self.specs is None and self.custom_isa is None:
            raise CompilerConnectionError()

        payload = self._compile_payload(quil_program)
        response = post_json(self.session, self.async_endpoint + "/job",
                             {"machine": "QUILC", "program": payload})
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

    def _clifford_application_payload(self, clifford, pauli):
        """
        Prepares a JSON payload for conjugating a Pauli by a Clifford - see apply_clifford_to_pauli.

        :param Program clifford: A Program that consists only of Clifford operations.
        :param PauliTerm pauli: A PauliTerm to be acted on by clifford via conjugation.
        :return: The JSON payload, with keys "clifford" and "pauli".
        """
        indices_and_terms = zip(*list(pauli.operations_as_set()))
        payload = {"clifford": clifford.out(),
                   "pauli": list(indices_and_terms)}
        return payload

    def apply_clifford_to_pauli(self, clifford, pauli_in):
        """
        Given a circuit that consists only of elements of the Clifford group, return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm representing PCP^{\dagger}.

        :param Program clifford: A Program that consists only of Clifford operations.
        :param PauliTerm pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to pauli_in * clifford * pauli_in^{\dagger}
        """
        payload = self._clifford_application_payload(clifford, pauli_in)
        phase_factor, paulis = post_json(self.session, self.sync_endpoint + "/apply-clifford", payload).json()
        pauli_out = PauliTerm("I", 0, 1.j ** phase_factor)
        clifford_qubits = clifford.get_qubits()
        pauli_qubits = pauli_in.get_qubits()
        all_qubits = sorted(set(pauli_qubits).union(set(clifford_qubits)))
        # The returned pauli will have specified its value on all_qubits, sorted by index. This is maximal set of
        # qubits that can be affected by this conjugation.
        for i, pauli in enumerate(paulis):
            pauli_out *= PauliTerm(pauli, all_qubits[i])
        return pauli_out * pauli_in.coefficient

    def _rb_sequence_payload(self, depth, qubits, gateset):
        """
        Prepares a JSON payload for generating a randomized benchmarking sequence - see generate_rb_sequence.

        :param int depth: The number of cliffords per rb sequences to generate.
        :param int qubits: The number of qubits to perform rb on.
        :param list gateset: A list of Gate objects that make up the gateset to decompose the Cliffords into.
        :return: The JSON payload, with keys "depth", "qubits", and "gateset".
        """
        payload = {"depth": depth,
                   "qubits": qubits,
                   "gateset": [gate.out() for gate in gateset]}
        return payload

    def generate_rb_sequence(self, depth, qubits, gateset):
        """
        Construct a randomized benchmarking experiment on the given qubits,
        decomposing into gateset.

        The JSON payload that is parsed is a list of lists of indices, or Nones. In the
        former case, they are the index of the gate in the gateset.

        :param int depth: The number of Clifford gates to include in the randomized benchmarking experiement.
        :param int qubits: The number of qubits to generate a randomized benchmarking sequence for.
        :param list gateset: A list of pyquil gates to decompose the Clifford elements into.
        """
        payload = self._rb_sequence_payload(depth, qubits, gateset)
        response = post_json(self.session, self.sync_endpoint + "/rb", payload).json()
        programs = []
        for clifford in response:
            clifford_program = Program()
            for index in clifford:
                clifford_program.inst(gateset[index])
            programs.append(clifford_program)
        return programs
