from pyquil.quil import Program
from ._base_connection import BaseConnection, validate_run_items, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE


class QPUConnection(BaseConnection):

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
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(classical_addresses)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT,
                   'addresses': classical_addresses,
                   'trials': trials,
                   'quil-instructions': quil_program.out()}

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)

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
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(qubits)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT_MEASURE,
                   'qubits': qubits,
                   'trials': trials,
                   'quil-instructions': quil_program.out()}

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)
