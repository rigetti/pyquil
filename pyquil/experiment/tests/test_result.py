import numpy as np

from pyquil.experiment import plusX
from pyquil.experiment._result import (
    bitstrings_to_expectations,
    ExperimentResult,
    ExperimentSetting,
)
from pyquil.paulis import sZ


def test_bitstrings_to_expectations():
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    assert np.allclose(
        bitstrings_to_expectations(bitstrings), np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    )
    assert np.allclose(
        bitstrings_to_expectations(bitstrings, joint_expectations=[[0]]),
        np.array([[1], [1], [-1], [-1]]),
    )
    assert np.allclose(
        bitstrings_to_expectations(bitstrings, joint_expectations=[[1]]),
        np.array([[1], [-1], [1], [-1]]),
    )
    assert np.allclose(
        bitstrings_to_expectations(bitstrings, joint_expectations=[[0, 1]]),
        np.array([[1], [-1], [-1], [1]]),
    )


def test_experiment_result_compat():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.9, std_err=0.05, total_counts=100
    )
    assert str(er) == "X0_0→(1+0j)*Z0: 0.9 +- 0.05"


def test_experiment_result():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.9, std_err=0.05, total_counts=100
    )
    assert str(er) == "X0_0→(1+0j)*Z0: 0.9 +- 0.05"
