import numpy as np

from pyquil.experiment import plusX
from pyquil.experiment._result import (
    bitstrings_to_expectations,
    correct_experiment_result,
    ratio_variance,
    ExperimentResult,
    ExperimentSetting,
)
from pyquil.paulis import sZ


def test_bitstrings_to_expectations():
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    assert np.allclose(bitstrings_to_expectations(bitstrings), np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]))
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
    er = ExperimentResult(setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.9, std_err=0.05, total_counts=100)
    assert str(er) == "X0_0→(1+0j)*Z0: 0.9 +- 0.05"


def test_experiment_result():
    er = ExperimentResult(setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.9, std_err=0.05, total_counts=100)
    assert str(er) == "X0_0→(1+0j)*Z0: 0.9 +- 0.05"


def test_ratio_variance_float():
    a, b, var_a, var_b = 1.0, 2.0, 0.1, 0.05
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    assert ab_ratio_var == 0.028125


def test_ratio_variance_numerator_zero():
    # denominator can't be zero, but numerator can be
    a, b, var_a, var_b = 0.0, 2.0, 0.1, 0.05
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    assert ab_ratio_var == 0.025


def test_ratio_variance_array():
    a = np.array([1.0, 10.0, 100.0])
    b = np.array([2.0, 20.0, 200.0])
    var_a = np.array([0.1, 1.0, 10.0])
    var_b = np.array([0.05, 0.5, 5.0])
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    np.testing.assert_allclose(ab_ratio_var, np.array([0.028125, 0.0028125, 0.00028125]))


def test_correct_experiment_result():
    e = ExperimentResult(setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.9, std_err=0.05, total_counts=100)
    cal = ExperimentResult(setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.95, std_err=0.01, total_counts=100)
    corrected = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9473684210526316,
        std_err=np.sqrt(ratio_variance(0.9, 0.05 ** 2, 0.95, 0.01 ** 2)),
        raw_expectation=0.9,
        raw_std_err=0.05,
        calibration_expectation=0.95,
        calibration_std_err=0.01,
        calibration_counts=100,
        total_counts=100,
    )
    assert corrected == correct_experiment_result(e, cal)
