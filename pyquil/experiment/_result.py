##############################################################################
# Copyright 2016-2019 Rigetti Computing
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
"""Definition of an ExperimentResult.

An ExperimentResult encapsulates the outcome of a collection of measurements that are aimed at estimating the
expectation value of some observable.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from pyquil.experiment._setting import ExperimentSetting

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentResult:
    """An expectation and standard deviation for the measurement of one experiment setting in a tomographic experiment.

    In the case of readout error calibration, we also include
    expectation, standard deviation and count for the calibration results, as well as the
    expectation and standard deviation for the corrected results.
    """

    setting: ExperimentSetting
    expectation: Union[float, complex]
    total_counts: int
    std_err: Optional[Union[float, complex]] = None
    raw_expectation: Optional[Union[float, complex]] = None
    raw_std_err: Optional[float] = None
    calibration_expectation: Optional[Union[float, complex]] = None
    calibration_std_err: Optional[Union[float, complex]] = None
    calibration_counts: Optional[int] = None
    additional_results: Optional[list["ExperimentResult"]] = None

    def __init__(
        self,
        setting: ExperimentSetting,
        expectation: Union[float, complex],
        total_counts: int,
        std_err: Optional[Union[float, complex]] = None,
        raw_expectation: Optional[Union[float, complex]] = None,
        raw_std_err: Optional[Union[float, complex]] = None,
        calibration_expectation: Optional[Union[float, complex]] = None,
        calibration_std_err: Optional[Union[float, complex]] = None,
        calibration_counts: Optional[int] = None,
        additional_results: Optional[list["ExperimentResult"]] = None,
    ):
        object.__setattr__(self, "setting", setting)
        object.__setattr__(self, "expectation", expectation)
        object.__setattr__(self, "total_counts", total_counts)
        object.__setattr__(self, "raw_expectation", raw_expectation)
        object.__setattr__(self, "calibration_expectation", calibration_expectation)
        object.__setattr__(self, "calibration_counts", calibration_counts)
        object.__setattr__(self, "additional_results", additional_results)
        object.__setattr__(self, "std_err", std_err)
        object.__setattr__(self, "raw_std_err", raw_std_err)
        object.__setattr__(self, "calibration_std_err", calibration_std_err)

    def __str__(self) -> str:
        return f"{self.setting}: {self.expectation} +- {self.std_err}"

    def __repr__(self) -> str:
        return f"ExperimentResult[{self}]"

    def serializable(self) -> dict[str, Any]:
        return {
            "type": "ExperimentResult",
            "setting": self.setting,
            "expectation": self.expectation,
            "std_err": self.std_err,
            "total_counts": self.total_counts,
            "raw_expectation": self.raw_expectation,
            "raw_std_err": self.raw_std_err,
            "calibration_expectation": self.calibration_expectation,
            "calibration_std_err": self.calibration_std_err,
            "calibration_counts": self.calibration_counts,
        }


def bitstrings_to_expectations(
    bitstrings: np.ndarray, joint_expectations: Optional[list[list[int]]] = None
) -> np.ndarray:
    """Given an array of bitstrings, map them to expectation values and return the desired joint expectation values.

    If no joint expectations are desired, then just the 1 -> -1, 0 -> 1 mapping is performed.

    :param bitstrings: Array of bitstrings to map.
    :param joint_expectations: Joint expectation values to calculate. Each entry is a list which
        contains the qubits to use in calculating the joint expectation value. Entries of length
        one just calculate single-qubit expectation values. Defaults to None, which is equivalent
        to the list of single-qubit expectations [[0], [1], ..., [n-1]] for bitstrings of length n.
    :return: An array of expectation values, of the same length as the array of bitstrings. The
        "width" could be different than the length of an individual bitstring (n) depending on
        the value of the ``joint_expectations`` parameter.
    """
    expectations: np.ndarray = 1 - 2 * bitstrings

    if joint_expectations is None:
        return expectations

    region_size = len(expectations[0])

    e = []
    for c in joint_expectations:
        where = np.zeros(region_size, dtype=bool)
        where[c] = True
        e.append(np.prod(expectations[:, where], axis=1))
    return np.stack(e, axis=-1)


def correct_experiment_result(
    result: ExperimentResult,
    calibration: ExperimentResult,
) -> ExperimentResult:
    """Given a raw, unmitigated result and its associated readout calibration, produce the result absent readout error.

    :param result: An ``ExperimentResult`` object with unmitigated readout error.
    :param calibration: An ``ExperimentResult`` object resulting from running readout calibration
        on the ``ExperimentSetting`` associated with the ``result`` parameter.
    :return: An ``ExperimentResult`` object corrected for symmetric readout error.
    """
    corrected_expectation = result.expectation / calibration.expectation

    if result.std_err is None or calibration.std_err is None:
        raise ValueError("Standard error not present in result or calibration.")

    corrected_variance = ratio_variance(
        result.expectation,
        result.std_err**2,
        calibration.expectation,
        calibration.std_err**2,
    )

    # recursively apply to additional results
    additional_results = None
    if result.additional_results is not None and calibration.additional_results:
        if len(result.additional_results) != len(calibration.additional_results):
            len_result = len(result.additional_results)
            len_calibration = len(calibration.additional_results)
            raise ValueError(f"Length of results ({len_result}) should match calibration ({len_calibration}).")
        additional_results = [
            correct_experiment_result(r, c) for r, c in zip(result.additional_results, calibration.additional_results)
        ]

    return ExperimentResult(
        setting=result.setting,
        expectation=corrected_expectation,
        std_err=np.sqrt(corrected_variance).item(),
        total_counts=result.total_counts,
        raw_expectation=result.expectation,
        raw_std_err=result.std_err,
        calibration_expectation=calibration.expectation,
        calibration_std_err=calibration.std_err,
        calibration_counts=calibration.total_counts,
        additional_results=additional_results,
    )


def ratio_variance(
    a: Union[float, complex, np.number, np.ndarray],
    var_a: Union[float, complex, np.number, np.ndarray],
    b: Union[float, complex, np.number, np.ndarray],
    var_b: Union[float, complex, np.number, np.ndarray],
) -> Union[float, complex, np.number, np.ndarray]:
    r"""Compute the variance on the ratio Y = A/B.

    Given random variables 'A' and 'B', compute the variance on the ratio Y = A/B. Denote the
    mean of the random variables as a = E[A] and b = E[B] while the variances are var_a = Var[A]
    and var_b = Var[B] and the covariance as Cov[A,B]. The following expression approximates the
    variance of Y

    Var[Y] \approx (a/b) ^2 * ( var_a /a^2 + var_b / b^2 - 2 * Cov[A,B]/(a*b) )

    We assume the covariance of A and B is negligible, resting on the assumption that A and B
    are independently measured. The expression above rests on the assumption that B is non-zero,
    an assumption which we expect to hold true in most cases, but makes no such assumptions
    about A. If we allow E[A] = 0, then calculating the expression above via numpy would complain
    about dividing by zero. Instead, we can re-write the above expression as

    Var[Y] \approx var_a /b^2 + (a^2 * var_b) / b^4

    where we have dropped the covariance term as noted above.

    See the following for more details:
      - https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4<300::AID-CYTO8>3.0.CO;2-O
      - http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
      - https://w.wiki/EMh

    :param a: Mean of 'A', to be used as the numerator in a ratio.
    :param var_a: Variance in 'A'
    :param b: Mean of 'B', to be used as the numerator in a ratio.
    :param var_b: Variance in 'B'
    """
    result = var_a / b**2 + (a**2 * var_b) / b**4
    return result
