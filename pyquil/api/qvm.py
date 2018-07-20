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
from typing import Iterable

import numpy as np
from six import integer_types

from pyquil.api._qam import QAM
from pyquil.api.compiler import CompilerConnection
from pyquil.api.job import Job
from pyquil.noise import apply_noise_model
from pyquil.paulis import PauliSum
from pyquil.quil import Program, get_classical_addresses_from_program
from pyquil.wavefunction import Wavefunction
from ._base_connection import validate_noise_probabilities, validate_run_items, TYPE_MULTISHOT, \
    TYPE_MULTISHOT_MEASURE, TYPE_WAVEFUNCTION, TYPE_EXPECTATION, get_job_id, get_session, \
    wait_for_job, post_json, get_json, SYNC_ENDPOINT, ASYNC_ENDPOINT, ForestConnection


class QVMConnection(object):
    """
    Represents a connection to the QVM.
    """

    def __init__(self, device=None, sync_endpoint=SYNC_ENDPOINT,
                 async_endpoint=ASYNC_ENDPOINT, api_key=None, user_id=None,
                 use_queue=False, ping_time=0.1, status_time=2, gate_noise=None,
                 measurement_noise=None, random_seed=None):
        """
        Constructor for QVMConnection. Sets up any necessary security, and establishes the noise
        model to use.

        :param Device device: The optional device, from which noise will be added by default to all
                              programs run on this instance.
        :param sync_endpoint: The endpoint of the server for running small jobs
        :param async_endpoint: The endpoint of the server for running large jobs
        :param api_key: The key to the Forest API Gateway (default behavior is to read from config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param bool use_queue: Disabling this parameter may improve performance for small, quick programs.
                               To support larger programs, set it to True. (default: False)
                               *_async methods will always use the queue
                               See https://go.rigetti.com/connections for more information.
        :param int ping_time: Time in seconds for how long to wait between polling the server for updated status
                              information on a job. Note that this parameter doesn't matter if use_queue is False.
        :param int status_time: Time in seconds for how long to wait between printing status information.
                                To disable printing of status entirely then set status_time to False.
                                Note that this parameter doesn't matter if use_queue is False.
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
                           Y, or Z gate getting applied to each qubit after a gate application or
                           reset. (default None)
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability of
                                  an X, Y, or Z gate getting applied before a a measurement.
                                  (default None)
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
                            automatically generated seed) or a non-negative integer.
        """
        if (device is not None and device.noise_model is not None) and \
                (gate_noise is not None or measurement_noise is not None):
            raise ValueError("""
You have attempted to supply the QVM with both a device noise model
(by having supplied a device argument), as well as either gate_noise
or measurement_noise. At this time, only one may be supplied.

To read more about supplying noise to the QVM, see http://pyquil.readthedocs.io/en/latest/noise_models.html#support-for-noisy-gates-on-the-rigetti-qvm.
""")

        if device is not None and device.noise_model is None:
            warnings.warn("""
You have supplied the QVM with a device that does not have a noise model. No noise will be added to
programs run on this QVM.
""")

        self.noise_model = device.noise_model if device is not None else None
        self.compiler = CompilerConnection(device=device) if device is not None else None

        self.async_endpoint = async_endpoint
        self.sync_endpoint = sync_endpoint

        self.use_queue = use_queue
        self.ping_time = ping_time
        self.status_time = status_time

        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

        self._connection = ForestConnection(sync_endpoint=sync_endpoint,
                                            async_endpoint=async_endpoint,
                                            api_key=api_key, user_id=user_id, use_queue=use_queue,
                                            ping_time=ping_time, status_time=status_time)
        self.session = self._connection.session  # backwards compatibility

    def ping(self):
        raise DeprecationWarning("ping() function is deprecated")

    def run(self, quil_program, classical_addresses=None, trials=1, needs_compilation=False, isa=None):
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.

        :param Program quil_program: A Quil program.
        :param list|range classical_addresses: A list of addresses.
        :param int trials: Number of shots to collect.
        :param bool needs_compilation: If True, preprocesses the job with the compiler.
        :param ISA isa: If set, compiles to this target ISA.
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(quil_program)

        return self._connection._qvm_run(quil_program, classical_addresses, trials,
                                         needs_compilation, isa, self.measurement_noise,
                                         self.gate_noise, self.random_seed).tolist()

    def run_async(self, quil_program, classical_addresses=None, trials=1, needs_compilation=False, isa=None):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(quil_program)

        return self._connection._qvm_run_async(quil_program, classical_addresses, trials,
                                               needs_compilation, isa, self.measurement_noise,
                                               self.gate_noise, self.random_seed)

    def run_and_measure(self, quil_program, qubits, trials=1, needs_compilation=False, isa=None):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the outcomes sampled from *different* ``run_and_measure`` calls *generally sample
            different bitstring distributions*.

        :param Program quil_program: A Quil program.
        :param list|range qubits: A list of qubits.
        :param int trials: Number of shots to collect.
        :param bool needs_compilation: If True, preprocesses the job with the compiler.
        :param ISA isa: If set, compiles to this target ISA.
        :return: A list of a list of bits.
        :rtype: list
        """
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._run_and_measure because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)

        payload = self._run_and_measure_payload(quil_program, qubits, trials, needs_compilation, isa)
        if self.use_queue or needs_compilation:
            if needs_compilation and not self.use_queue:
                warnings.warn('Synchronous QVM connection does not support compilation preprocessing. Running this job over the asynchronous endpoint, as if use_queue were set to True.')

            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result().tolist()
        else:
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def run_and_measure_async(self, quil_program, qubits, trials=1, needs_compilation=False, isa=None):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials, needs_compilation, isa)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _run_and_measure_payload(self, quil_program, qubits, trials, needs_compilation, isa):
        if not quil_program:
            raise ValueError("You have attempted to run an empty program."
                             " Please provide gates or measure instructions to your program.")

        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(qubits)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")
        if needs_compilation and not isa:
            raise TypeError("ISA cannot be None if QVM program needs compilation preprocessing.")

        if self.noise_model is not None:
            compiled_program = self.compiler.compile(quil_program)
            quil_program = apply_noise_model(compiled_program, self.noise_model)

        payload = {"type": TYPE_MULTISHOT_MEASURE,
                   "qubits": list(qubits),
                   "trials": trials}
        if needs_compilation:
            payload["uncompiled-quil"] = quil_program.out()
            payload["target-device"] = {"isa": isa.to_dict()}
        else:
            payload["compiled-quil"] = quil_program.out()

        self._maybe_add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

    def wavefunction(self, quil_program, classical_addresses=None, needs_compilation=False, isa=None):
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param Program quil_program: A Quil program.
        :param list|range classical_addresses: An optional list of classical addresses.
        :param needs_compilation: If True, preprocesses the job with the compiler.
        :param isa: If set, compiles to this target ISA.
        :return: A Wavefunction object representing the state of the QVM.
        :rtype: Wavefunction
        """
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._wavefunction because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)

        if classical_addresses is None:
            classical_addresses = []

        if self.use_queue or needs_compilation:
            if needs_compilation and not self.use_queue:
                warnings.warn('Synchronous QVM connection does not support compilation preprocessing. Running this job over the asynchronous endpoint, as if use_queue were set to True.')

            payload = self._wavefunction_payload(quil_program, classical_addresses, needs_compilation, isa)
            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._wavefunction_payload(quil_program, classical_addresses, needs_compilation, isa)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return Wavefunction.from_bit_packed_string(response.content, classical_addresses)

    def wavefunction_async(self, quil_program, classical_addresses=None, needs_compilation=False, isa=None):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._wavefunction because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)
        if classical_addresses is None:
            classical_addresses = []

        payload = self._wavefunction_payload(quil_program, classical_addresses, needs_compilation, isa)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _wavefunction_payload(self, quil_program, classical_addresses, needs_compilation, isa):
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # _base_connection._wavefunction_payload because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if needs_compilation and not isa:
            raise TypeError("ISA cannot be None if QVM program requires compilation preprocessing.")

        payload = {'type': TYPE_WAVEFUNCTION,
                   'addresses': list(classical_addresses)}

        if needs_compilation:
            payload['uncompiled-quil'] = quil_program.out()
            payload['target-device'] = {"isa": isa.to_dict()}
        else:
            payload['compiled-quil'] = quil_program.out()

        self._maybe_add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

    def expectation(self, prep_prog, operator_programs=None, needs_compilation=False, isa=None):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        To measure the expectation of a PauliSum, you probably want to
        do something like this::

                progs, coefs = hamiltonian.get_programs()
                expect_coeffs = np.array(cxn.expectation(prep_program, operator_programs=progs))
                return np.real_if_close(np.dot(coefs, expect_coeffs))

        :param Program prep_prog: Quil program for state preparation.
        :param list operator_programs: A list of Programs, each specifying an operator whose expectation to compute.
            Default is a list containing only the empty Program.
        :param bool needs_compilation: If True, preprocesses the job with the compiler.
        :param ISA isa: If set, compiles to this target ISA.
        :return: Expectation values of the operators.
        :rtype: List[float]
        """
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._expectation because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)

        if isinstance(operator_programs, Program):
            warnings.warn("You have provided a Program rather than a list of Programs. The results from expectation "
                          "will be line-wise expectation values of the operator_programs.", SyntaxWarning)
        if needs_compilation:
            raise TypeError("Expectation QVM programs do not support compilation preprocessing."
                            "  Make a separate CompilerConnection job first.")
        if self.use_queue:
            payload = self._expectation_payload(prep_prog, operator_programs)
            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result().tolist()
        else:
            payload = self._expectation_payload(prep_prog, operator_programs)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def pauli_expectation(self, prep_prog, pauli_terms):
        """
        Calculate the expectation value of Pauli operators given a state prepared by prep_program.

        If ``pauli_terms`` is a ``PauliSum`` then the returned value is a single ``float``,
        otherwise the returned value is a list of ``float``s, one for each ``PauliTerm`` in the
        list.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        :param Program prep_prog: Quil program for state preparation.
        :param Sequence[PauliTerm]|PauliSum pauli_terms: A list of PauliTerms or a PauliSum.
        :return: If ``pauli_terms`` is a PauliSum return its expectation value. Otherwise return
          a list of expectation values.
        :rtype: float|List[float]
        """

        is_pauli_sum = False
        if isinstance(pauli_terms, PauliSum):
            progs, coeffs = pauli_terms.get_programs()
            is_pauli_sum = True
        else:
            coeffs = [pt.coefficient for pt in pauli_terms]
            progs = [pt.program for pt in pauli_terms]

        bare_results = self.expectation(prep_prog, progs, needs_compilation=False, isa=False)
        results = [c * r for c, r in zip(coeffs, bare_results)]
        if is_pauli_sum:
            return sum(results)
        return results

    def expectation_async(self, prep_prog, operator_programs=None, needs_compilation=False, isa=None):
        """
        Similar to expectation except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if needs_compilation:
            raise TypeError("Expectation QVM programs do not support compilation preprocessing.  Make a separate CompilerConnection job first.")

        payload = self._expectation_payload(prep_prog, operator_programs)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _expectation_payload(self, prep_prog, operator_programs):
        if operator_programs is None:
            operator_programs = [Program()]

        if not isinstance(prep_prog, Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {'type': TYPE_EXPECTATION,
                   'state-preparation': prep_prog.out(),
                   'operators': [x.out() for x in operator_programs]}

        self._add_rng_seed_to_payload(payload)

        return payload

    def get_job(self, job_id):
        """
        Given a job id, return information about the status of the job

        :param str job_id: job id
        :return: Job object with the status and potentially results of the job
        :rtype: Job
        """
        return self._connection._get_job(job_id, machine='QVM')

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
        return self._connection._wait_for_job(job_id, 'QVM', ping_time, status_time)

    def _maybe_add_noise_to_payload(self, payload):
        """
        Set the gate noise and measurement noise of a payload.
        """
        if self.measurement_noise is not None:
            payload["measurement-noise"] = self.measurement_noise
        if self.gate_noise is not None:
            payload["gate-noise"] = self.gate_noise

    def _add_rng_seed_to_payload(self, payload):
        """
        Add a random seed to the payload.
        """
        if self.random_seed is not None:
            payload['rng-seed'] = self.random_seed


class QVM(QAM):
    def __init__(self, connection: ForestConnection, noise_model=None, gate_noise=None,
                 measurement_noise=None, random_seed=None):
        """
        A virtual machine that classically emulates the execution of Quil programs.

        :param connection: A connection to the Forest web API.
        :param noise_model: A noise model that describes noise to apply when emulating a program's
            execution.
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
           Y, or Z gate getting applied to each qubit after a gate application or reset. The
           default value of None indicates no noise.
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability
            of an X, Y, or Z gate getting applied before a measurement. The default value of
            None indicates no noise.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        """
        if (noise_model is not None) and (gate_noise is not None or measurement_noise is not None):
            raise ValueError("""
You have attempted to supply the QVM with both a Kraus noise model
(by supplying a `noise_model` argument), as well as either `gate_noise`
or `measurement_noise`. At this time, only one may be supplied.

To read more about supplying noise to the QVM, see http://pyquil.readthedocs.io/en/latest/noise_models.html#support-for-noisy-gates-on-the-rigetti-qvm.
""")

        self.noise_model = noise_model
        self.connection = connection

        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

    def run(self, quil_program: Program, classical_addresses: Iterable[int],
            trials: int) -> np.ndarray:
        """
        Run a Quil program on the QVM multiple times and return the values stored in the
        classical registers designated by the classical_addresses parameter.

        :param quil_program: A program to run
        :param classical_addresses: Classical register addresses to return
        :param int trials: Number of times to repeatedly run the program. This is sometimes called
            the number of shots.
        :return: An array of bitstrings of shape ``(trials, len(classical_addresses))``
        """
        if self.noise_model is not None:
            quil_program = apply_noise_model(quil_program, self.noise_model)

        return self.connection._qvm_run(quil_program=quil_program,
                                        classical_addresses=classical_addresses,
                                        trials=trials, needs_compilation=False, isa=None,
                                        measurement_noise=self.measurement_noise,
                                        gate_noise=self.gate_noise,
                                        random_seed=self.random_seed)

    def run_async(self, quil_program, classical_addresses, trials):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if self.noise_model is not None:
            quil_program = apply_noise_model(quil_program, self.noise_model)

        return self.connection._qvm_run_async(quil_program=quil_program,
                                              classical_addresses=classical_addresses,
                                              trials=trials, needs_compilation=False, isa=None,
                                              measurement_noise=self.measurement_noise,
                                              gate_noise=self.gate_noise,
                                              random_seed=self.random_seed)

    def wait_for_job(self, job_id, ping_time=None, status_time=None) -> Job:
        """
        For async functions, wait for the specified job to be done and return the completed job.

        :param job_id: The id of the job returned by ``_async`` methods.
        :param ping_time: An optional time in seconds to poll for job completion.
        :param status_time: An optional time in seconds to print the status of a job.
        :return: The completed job.
        """
        return self.connection._wait_for_job(job_id=job_id, ping_time=ping_time,
                                             status_time=status_time, machine='QVM')
