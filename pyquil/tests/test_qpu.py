import numpy as np
import pytest

from pyquil import Program
from pyquil.api import QPU, get_devices, ForestConnection
from pyquil.gates import MEASURE, X


def test_qpu_run(forest: ForestConnection):
    devices = get_devices(async_endpoint=forest.async_endpoint, api_key=forest.api_key,
                          user_id=forest.user_id, as_dict=True)

    for name, dev in devices.items():
        if not dev.is_online:
            continue

        # TODO: gh-372. No way to query whether a device is available for running
        pytest.xfail("Please fix after gh-372")
        qpu = QPU(connection=forest, device_name=name)
        bitstrings = qpu.run(
            quil_program=Program(X(0), MEASURE(0, 0)),
            classical_addresses=[0],
            trials=1000,
        )
        assert bitstrings.shape == (1000, 1)
        assert np.mean(bitstrings) > 0.8


def test_qpu_run_async(forest: ForestConnection):
    devices = get_devices(async_endpoint=forest.async_endpoint, api_key=forest.api_key,
                          user_id=forest.user_id, as_dict=True)

    for name, dev in devices.items():
        if not dev.is_online:
            continue

        # TODO: gh-372. No way to query whether a device is available for running
        pytest.xfail("Please fix after gh-372")

        qpu = QPU(connection=forest, device_name=name)
        job_id = qpu.run_async(
            quil_program=Program(X(0), MEASURE(0, 0)),
            classical_addresses=[0],
            trials=1000,
        )
        print(job_id)
        assert isinstance(job_id, str)

        bitstrings = qpu.wait_for_job(job_id).result()
        assert bitstrings.shape == (1000, 1)
        assert np.mean(bitstrings) > 0.8
