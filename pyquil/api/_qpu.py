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
import uuid
from typing import Dict

import numpy as np
from rpcq.core_messages import QPURequest
from rpcq.json_rpc import Shim

from pyquil.api._qam import QAM
from pyquil.api._error_reporting import _record_call

DEVICES_ENDPOINT = 'todo'


def get_devices(async_endpoint=DEVICES_ENDPOINT):
    """
    Get a list of currently available devices. The arguments for this method are the same as those
    for QPUConnection.

    Note that this method will only work for accounts that have QPU access.

    :return: Set or Dictionary (keyed by device name) of all available devices.
    :rtype: Set|Dict
    """
    # TODO: implement
    return {}

    # session = get_session()
    # response = session.get(async_endpoint + '/devices')
    # if response.status_code >= 400:
    #     raise parse_error(response)
    #
    # return {name: Device(name, device) for (name, device) in response.json()['devices'].items()}


def decode_buffer(buffer: dict) -> np.ndarray:
    """
    Translate a DataBuffer into a numpy array.

    :param buffer: Dictionary with 'data' byte array, 'dtype', and 'shape' fields
    :return: NumPy array of decoded data
    """
    buf = np.frombuffer(buffer['data'], dtype=buffer['dtype'])
    return buf.reshape(buffer['shape'])


class QPU(QAM):
    @_record_call
    def __init__(self, endpoint: str, user: str = "pyquil-user"):
        """
        A connection to the QPU.

        :param endpoint: Address to connect to the QPU server.
        :param user: A string identifying who's running jobs.
        """
        super().__init__()
        self.shim = Shim(endpoint)
        self.user = user
        self._last_results = {}

    @_record_call
    def run(self):
        """
        Run a pyquil program on the QPU.

        This formats the classified data from the QPU server by stacking measured bits into
        an array of shape (trials, classical_addresses). The mapping of qubit to
        classical address is backed out from MEASURE instructions in the program, so
        only do measurements where there is a 1-to-1 mapping between qubits and classical
        addresses.

        :return: A numpy array of classified (0/1) measurement results of
            shape (trials, size of "ro")
        """
        super().run()

        request = QPURequest(program=self.binary.program,
                             patch_values=self._build_patch_values(),
                             id=str(uuid.uuid4()))

        job_id = self.shim.call('execute_qpu_request', request=request, user=self.user)
        results = self._get_buffers(job_id)

        # reorder the results and zip them up
        self.bitstrings = np.vstack([results[f"q{qubit}"] for qubit in self.binary.ro_sources]).T
        self._last_results = results
        return self

    def _get_buffers(self, job_id: str) -> Dict[str, np.ndarray]:
        """
        Return the decoded result buffers for particular job_id.

        :param job_id: Unique identifier for the job in question
        :return: Decoded buffers or throw an error
        """
        buffers = self.shim.call('get_buffers', job_id, wait=True)
        return {k: decode_buffer(v) for k, v in buffers.items()}

    def _build_patch_values(self) -> dict:
        patch_table = {}

        for name, spec in self.binary.memory_descriptors.items():
            # NOTE: right now we fake reading out measurement values into classical memory
            if name == "ro":
                continue
            patch_table[name] = [0] * spec.length

        for k, v in self.variables_shim.items():
            # NOTE: right now we fake reading out measurement values into classical memory
            if k.name == "ro":
                continue

            # floats stored in tsunami memory are expected to be in revolutions rather than radians.
            if isinstance(v, float):
                v /= 2 * np.pi

            patch_table[k.name][k.index] = v

        return patch_table
