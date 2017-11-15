from six import integer_types

from pyquil.api import Job
from pyquil.quil import Program
from ._base_connection import validate_run_items, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE, get_job_id, AsyncConnection


class QPUConnection(AsyncConnection):

    def __init__(self, endpoint='https://job.rigetti.com/beta', api_key=None, user_id=None):
        super(QPUConnection, self).__init__(endpoint=endpoint, api_key=api_key, user_id=user_id)

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a pyQuil program on the QPU. This functionality is in beta.

        :param Program quil_program: Quil program to run on the QPU
        :param list classical_addresses: Currently unused
        :param int trials: Number of shots to take
        :return: A job result
        :rtype: JobResult
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)

        response = self._post_json({"machine": "QPU", "program": payload}, route="/job")
        job = self.wait_for_job(get_job_id(response))
        return job.result()

    def run_async(self, quil_program, classical_addresses, trials=1):
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = self._post_json({"machine": "QPU", "program": payload}, route="/job")
        return get_job_id(response)

    def _run_payload(self, quil_program, classical_addresses, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": classical_addresses,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        return payload

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a pyQuil program on the QPU multiple times, measuring all the qubits in the QPU
        simultaneously at the end of the program each time. This functionality is in beta.

        :param Program quil_program: Quil program to run on the QPU
        :param list qubits: The list of qubits to return results for
        :param int trials: Number of shots to take
        :return: A job result
        :rtype: JobResult
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)

        response = self._post_json({"machine": "QPU", "program": payload}, route="/job")
        job = self.wait_for_job(get_job_id(response))
        return job.result()

    def run_and_measure_async(self, quil_program, qubits, trials):
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = self._post_json({"machine": "QPU", "program": payload}, route="/job")
        return get_job_id(response)

    def _run_and_measure_payload(self, quil_program, qubits, trials):
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(qubits)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT_MEASURE,
                   'qubits': qubits,
                   'trials': trials,
                   'quil-instructions': quil_program.out()}

        return payload

    def get_job(self, job_id):
        response = self._get_json(route="/job/" + job_id)
        return Job(response.json())
