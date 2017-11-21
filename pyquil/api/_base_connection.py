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

from __future__ import print_function

import requests
import sys

import time
from requests.adapters import HTTPAdapter
from six import integer_types
from urllib3 import Retry

from .job import Job
from ._config import PyquilConfig

TYPE_EXPECTATION = "expectation"
TYPE_MULTISHOT = "multishot"
TYPE_MULTISHOT_MEASURE = "multishot-measure"
TYPE_WAVEFUNCTION = "wavefunction"


class BaseConnection(object):
    def __init__(self, async_endpoint=None, api_key=None, user_id=None):
        self._session = requests.Session()
        retry_adapter = HTTPAdapter(max_retries=Retry(total=3,
                                                      method_whitelist=['POST'],
                                                      status_forcelist=[502, 503, 504, 521, 523],
                                                      backoff_factor=0.2,
                                                      raise_on_status=False))

        # We need this to get binary payload for the wavefunction call.
        self._session.headers.update({"Accept": "application/octet-stream"})

        self._session.mount("http://", retry_adapter)
        self._session.mount("https://", retry_adapter)

        config = PyquilConfig()
        self.api_key = api_key if api_key else config.api_key
        self.user_id = user_id if user_id else config.user_id

        self.async_endpoint = async_endpoint

    def get_job(self, job_id):
        """
        Given a job id, return information about the status of the job

        :param str job_id: job id
        :return: Job object with the status and potentially results of the job
        :rtype: Job
        """
        response = self._get_json(self.async_endpoint + "/job/" + job_id)
        return Job(response.json())

    def wait_for_job(self, job_id, ping_time=0.1, status_time=2):
        """
        Wait for the results of a job and periodically print status

        :param job_id: Job id
        :param ping_time: How often to poll the server
        :param status_time: How often to print status, set to False to never print status
        :return: Completed Job
        """
        count = 0
        while True:
            job = self.get_job(job_id)
            if job.is_done():
                break

            if status_time and count % int(status_time / ping_time) == 0:
                if job.is_queued():
                    print("job {} is currently queued at position {}".format(job.job_id, job.position_in_queue()))
                elif job.is_running():
                    print("job {} is currently running".format(job.job_id))

            time.sleep(ping_time)
            count += 1

        return job

    def _post_json(self, url, json):
        """
        Post JSON to the Forest endpoint.

        :param str url: The full url to post to
        :param dict json: JSON.
        :return: A non-error response.
        """
        headers = {
            'X-Api-Key': self.api_key,
            'X-User-Id': self.user_id,
            'Content-Type': 'application/json; charset=utf-8'
        }
        res = self._session.post(url, json=json, headers=headers)

        # Print some nice info for unauthorized/permission errors.
        if res.status_code == 401 or res.status_code == 403:
            print("! ERROR:\n"
                  "!   There was an issue validating your forest account.\n"
                  "!   Have you run the pyquil-config-setup command yet?\n"
                  "! The server came back with the following information:\n"
                  "%s\n%s\n%s" % ("=" * 80, res.text, "=" * 80), file=sys.stderr)
            print("! If you suspect this to be a bug in pyQuil or Rigetti Forest,\n"
                  "! then please describe the problem in a GitHub issue at:\n!\n"
                  "!      https://github.com/rigetticomputing/pyquil/issues\n", file=sys.stderr)

        # Print some nice info for invalid input or internal server errors.
        if res.status_code == 400 or res.status_code >= 500:
            print("! ERROR:\n"
                  "!   Server caught an error. This could be due to a bug in the server\n"
                  "!   or a bug in your code. The server came back with the following\n"
                  "!   information:\n"
                  "%s\n%s\n%s" % ("=" * 80, res.text, "=" * 80), file=sys.stderr)
            print("! If you suspect this to be a bug in pyQuil or Rigetti Forest,\n"
                  "! then please describe the problem in a GitHub issue at:\n!\n"
                  "!      https://github.com/rigetticomputing/pyquil/issues\n", file=sys.stderr)

        res.raise_for_status()
        return res

    def _get_json(self, url):
        """
        Get JSON from a Forest endpoint.

        :param str url: The full url to fetch
        :return: Response object
        """
        headers = {
            'X-Api-Key': self.api_key,
            'X-User-Id': self.user_id,
            'Content-Type': 'application/json; charset=utf-8'
        }
        return requests.get(url, headers=headers)


def validate_noise_probabilities(noise_parameter):
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?

    :param list noise_parameter: List of noise parameter values to be validated.
    """
    if not noise_parameter:
        return
    if not isinstance(noise_parameter, list):
        raise TypeError("noise_parameter must be a list")
    if any([not isinstance(value, float) for value in noise_parameter]):
        raise TypeError("noise_parameter values should all be floats")
    if len(noise_parameter) != 3:
        raise ValueError("noise_parameter lists must be of length 3")
    if sum(noise_parameter) > 1 or sum(noise_parameter) < 0:
        raise ValueError("sum of entries in noise_parameter must be between 0 and 1 (inclusive)")
    if any([value < 0 for value in noise_parameter]):
        raise ValueError("noise_parameter values should all be non-negative")


def validate_run_items(run_items):
    """
    Check the validity of classical addresses / qubits for the payload.

    :param list run_items: List of classical addresses or qubits to be validated.
    """
    if not isinstance(run_items, list):
        raise TypeError("run_items must be a list")
    if any([not isinstance(i, integer_types) for i in run_items]):
        raise TypeError("run_items list must contain integer values")


def get_job_id(response):
    return response.json()['jobId']
