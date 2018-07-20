import numpy as np

from pyquil import Program
from pyquil.api import QVM, ForestConnection
from pyquil.gates import MEASURE, X


def test_qvm_run(forest: ForestConnection):
    qvm = QVM(connection=forest, gate_noise=[0.01] * 3)
    bitstrings = qvm.run(
        quil_program=Program(X(0), MEASURE(0, 0)),
        classical_addresses=[0],
        trials=1000,
    )
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qpu_run_async(forest: ForestConnection):
    qvm = QVM(connection=forest, gate_noise=[0.01] * 3)
    job_id = qvm.run_async(
        quil_program=Program(X(0), MEASURE(0, 0)),
        classical_addresses=[0],
        trials=1000,
    )
    print(job_id)
    assert isinstance(job_id, str)
    job = qvm.wait_for_job(job_id)
    bitstrings = job.result()
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8
