import base64
import json
import struct
import time

from six import integer_types
import numpy as np

from pyquil.wavefunction import Wavefunction

OCTETS_PER_DOUBLE_FLOAT = 8
OCTETS_PER_COMPLEX_DOUBLE = 2 * OCTETS_PER_DOUBLE_FLOAT


def wait_for_job(res, ping_time=0.5):
    """
    Blocks execution and waits for an async Forest Job to complete.

    :param JobResult res: The JobResult object to wait for.
    :param ping_time: The interval (in seconds) at which to ping the server.
    :return: The completed JobResult
    """
    while not res.is_done():
        res.get()
        time.sleep(ping_time)
    return res


class JobResult(object):
    def __init__(self, qpu, success, result=None, payload=None):
        """
        :param QPUConnection qpu:
        :param bool success:
        :param dict result: JSON dictionary of the result message
        """
        self.qpu = qpu
        self.success = success
        self.result = result
        self.payload = payload

    def is_done(self):
        """
        :return: Returns True if the Job is completed and False otherwise.
        :rtype: bool
        """
        return 'result' in self.result

    def job_id(self):
        """
        :return: Returns the id of this job
        :rtype: str
        """
        return self.result['jobId']

    def get(self):
        """
        Gets an update from the Forest API on the status of this job.
        :return: The JobResult object with fields updated.
        :rtype: JobResult
        """
        return self.qpu.get_job(self)

    def _update(self, success, result):
        self.success = success
        self.result = result
        return self

    def decode(self):
        """
        Decodes the result of the job.
        :return: Depends on the type of job. A JSON object.
        """
        return json.loads(self.result['result'])

    def __str__(self):
        return str(self.result)

    @classmethod
    def load_res(cls, qpu, response):
        result = json.loads(response.content.decode("utf-8"))
        return cls(qpu, response.ok, result)


class WavefunctionResult(JobResult):

    def decode(self):
        string_result = self.result['result']
        coef_string = base64.b64decode(string_result.encode("utf-8"))
        wvf, classical_mem = recover_complexes(coef_string=coef_string,
                                               classical_addresses=self.payload['addresses'])
        return wvf, classical_mem


def recover_complexes(coef_string, classical_addresses):
    """
    From a bit packed string, unpacks to get the wavefunction and classical measurement results
    :param coef_string:
    :param classical_addresses:
    :return:
    """
    num_octets = len(coef_string)
    num_addresses = len(classical_addresses)
    num_memory_octets = _round_to_next_multiple(num_addresses, 8) / 8
    num_wavefunction_octets = num_octets - num_memory_octets

    # Parse the classical memory
    mem = []
    for i in range(num_memory_octets):
        octet = struct.unpack('B', coef_string[i])[0]
        mem.extend(_octet_bits(octet))

    mem = mem[0:num_addresses]

    # Parse the wavefunction
    wf = np.zeros(num_wavefunction_octets / OCTETS_PER_COMPLEX_DOUBLE, dtype=np.cfloat)
    for i, p in enumerate(range(num_memory_octets, num_octets, OCTETS_PER_COMPLEX_DOUBLE)):
        re_be = coef_string[p: p + OCTETS_PER_DOUBLE_FLOAT]
        im_be = coef_string[p + OCTETS_PER_DOUBLE_FLOAT: p + OCTETS_PER_COMPLEX_DOUBLE]
        re = struct.unpack('>d', re_be)[0]
        im = struct.unpack('>d', im_be)[0]
        wf[i] = complex(re, im)

    return Wavefunction(wf), mem


class RamseyResult(JobResult):
    pass


class RabiResult(JobResult):
    pass


class T1Result(JobResult):
    pass


def _round_to_next_multiple(n, m):
    """
    Round up the the next multiple.

    :param n: The number to round up.
    :param m: The multiple.
    :return: The rounded number
    """
    return n if n % m == 0 else n + m - n % m


def _octet_bits(o):
    """
    Get the bits of an octet.

    :param o: The octets.
    :return: The bits as a list in LSB-to-MSB order.
    :rtype: list
    """
    if not isinstance(o, integer_types):
        raise TypeError("o should be an int")
    if not (0 <= o <= 255):
        raise ValueError("o should be between 0 and 255 inclusive")
    bits = [0] * 8
    for i in range(8):
        if 1 == o & 1:
            bits[i] = 1
        o = o >> 1
    return bits