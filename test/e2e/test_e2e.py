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
from multiprocessing.pool import ThreadPool

import numpy as np

from pyquil import Program
from pyquil.api import QuantumComputer, QPU
from pyquil.gates import H, CNOT, MEASURE, RX
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare

TEST_PROGRAM = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(1000)


def test_basic_program(qc: QuantumComputer):
    results = qc.run(qc.compile(TEST_PROGRAM)).readout_data.get("ro")

    assert results.shape == (1000, 2)


def test_parametric_program(qc: QuantumComputer):
    compiled = qc.compile(
        Program(
            Declare("ro", "BIT", 1),
            Declare("theta", "REAL", 1),
            RX(MemoryReference("theta"), 0),
            MEASURE(0, ("ro", 0)),
        ).wrap_in_numshots_loop(1000),
    )

    all_results = []
    for theta in [0, np.pi, 2 * np.pi]:
        compiled.write_memory(region_name="theta", value=theta)
        results = qc.run(compiled).readout_data.get("ro")
        all_results.append(np.mean(results))

    if isinstance(qc.qam, QPU):
        assert all_results[0] < 0.2
        assert all_results[1] > 0.8
        assert all_results[2] < 0.2
    else:
        assert all_results[0] == 0.0
        assert all_results[1] > 0.8
        assert all_results[2] == 0.0


def test_multithreading(qc: QuantumComputer):
    def run_program(
            program: Program,
            qc: QuantumComputer,
    ) -> np.ndarray:
        return qc.run(qc.compile(program)).readout_data.get('ro')

    args = [(TEST_PROGRAM, qc) for _ in range(20)]
    with ThreadPool(10) as pool:
        results = pool.starmap(run_program, args)

    assert len(results) == 20
    for result in results:
        assert result.shape == (1000, 2)
