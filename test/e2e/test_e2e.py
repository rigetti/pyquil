##############################################################################
# Copyright 2016-2021 Rigetti Computing
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
from multiprocessing.pool import Pool, ThreadPool
import numpy as np

from qcs_api_client.client import QCSClientConfiguration

from pyquil import get_qc, Program
from pyquil.api import EngagementManager
from pyquil.gates import H, CNOT, MEASURE
from pyquil.quilbase import Declare

TEST_PROGRAM = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(1000)


def test_basic_program(quantum_processor_id: str, client_configuration: QCSClientConfiguration):
    qc = get_qc(quantum_processor_id, client_configuration=client_configuration)

    results = qc.run(qc.compile(TEST_PROGRAM))

    assert results.shape == (1000, 2)


def test_multithreading(
    quantum_processor_id: str, client_configuration: QCSClientConfiguration, engagement_manager: EngagementManager
):
    args = [(TEST_PROGRAM, quantum_processor_id, client_configuration, engagement_manager) for _ in range(20)]
    with ThreadPool(10) as pool:
        results = pool.starmap(run_program, args)

    assert len(results) == 20
    for result in results:
        assert result.shape == (1000, 2)


def test_multiprocessing(
    quantum_processor_id: str, client_configuration: QCSClientConfiguration, engagement_manager: EngagementManager
):
    args = [(TEST_PROGRAM, quantum_processor_id, client_configuration, engagement_manager) for _ in range(20)]
    with Pool(10) as pool:
        results = pool.starmap(run_program, args)

    assert len(results) == 20
    for result in results:
        assert result.shape == (1000, 2)


# NOTE: This must be outside of the test function, or multiprocessing complains that it can't be pickled
def run_program(
    program: Program,
    quantum_processor_id: str,
    client_configuration: QCSClientConfiguration,
    engagement_manager: EngagementManager,
)-> np.ndarray:
    qc = get_qc(
        quantum_processor_id,
        client_configuration=client_configuration,
        engagement_manager=engagement_manager,
    )
    return qc.run(qc.compile(program)).readout_data['ro']
