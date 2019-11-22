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
"""
Schema definition of an ExperimentResult, which encapsulates the outcome of a collection of
measurements that are aimed at estimating the expectation value of some observable.
"""
import logging
import sys
import warnings
from typing import List, Optional, Union

import numpy as np

from pyquil.experiment._setting import ExperimentSetting

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass


log = logging.getLogger(__name__)


def bitstrings_to_expectations(
        bitstrings: np.ndarray,
        joint_expectations: Optional[List[List[int]]] = None
) -> np.ndarray:
    """
    Given an array of bitstrings (each of which is represented as an array of bits), map them to
    expectation values and return the desired correlations. If no correlations are given, then just
    the 1 -> -1, 0 -> 1 mapping is performed.

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


@dataclass(frozen=True)
class ExperimentResult:
    """An expectation and standard deviation for the measurement of one experiment setting
    in a tomographic experiment.

    In the case of readout error calibration, we also include
    expectation, standard deviation and count for the calibration results, as well as the
    expectation and standard deviation for the corrected results.
    """

    setting: ExperimentSetting
    expectation: Union[float, complex]
    total_counts: int
    std_err: Union[float, complex] = None
    raw_expectation: Union[float, complex] = None
    raw_std_err: float = None
    calibration_expectation: Union[float, complex] = None
    calibration_std_err: Union[float, complex] = None
    calibration_counts: int = None

    def __init__(self, setting: ExperimentSetting,
                 expectation: Union[float, complex],
                 total_counts: int,
                 stddev: Union[float, complex] = None,
                 std_err: Union[float, complex] = None,
                 raw_expectation: Union[float, complex] = None,
                 raw_stddev: float = None,
                 raw_std_err: float = None,
                 calibration_expectation: Union[float, complex] = None,
                 calibration_stddev: Union[float, complex] = None,
                 calibration_std_err: Union[float, complex] = None,
                 calibration_counts: int = None):

        object.__setattr__(self, 'setting', setting)
        object.__setattr__(self, 'expectation', expectation)
        object.__setattr__(self, 'total_counts', total_counts)
        object.__setattr__(self, 'raw_expectation', raw_expectation)
        object.__setattr__(self, 'calibration_expectation', calibration_expectation)
        object.__setattr__(self, 'calibration_counts', calibration_counts)

        if stddev is not None:
            warnings.warn("'stddev' has been renamed to 'std_err'")
            std_err = stddev
        object.__setattr__(self, 'std_err', std_err)

        if raw_stddev is not None:
            warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
            raw_std_err = raw_stddev
        object.__setattr__(self, 'raw_std_err', raw_std_err)

        if calibration_stddev is not None:
            warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
            calibration_std_err = calibration_stddev
        object.__setattr__(self, 'calibration_std_err', calibration_std_err)

    def get_stddev(self) -> Union[float, complex]:
        warnings.warn("'stddev' has been renamed to 'std_err'")
        return self.std_err

    def set_stddev(self, value: Union[float, complex]):
        warnings.warn("'stddev' has been renamed to 'std_err'")
        object.__setattr__(self, 'std_err', value)

    stddev = property(get_stddev, set_stddev)

    def get_raw_stddev(self) -> float:
        warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
        return self.raw_std_err

    def set_raw_stddev(self, value: float):
        warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
        object.__setattr__(self, 'raw_std_err', value)

    raw_stddev = property(get_raw_stddev, set_raw_stddev)

    def get_calibration_stddev(self) -> Union[float, complex]:
        warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
        return self.calibration_std_err

    def set_calibration_stddev(self, value: Union[float, complex]):
        warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
        object.__setattr__(self, 'calibration_std_err', value)

    calibration_stddev = property(get_calibration_stddev, set_calibration_stddev)

    def __str__(self):
        return f'{self.setting}: {self.expectation} +- {self.std_err}'

    def __repr__(self):
        return f'ExperimentResult[{self}]'

    def serializable(self):
        return {
            'type': 'ExperimentResult',
            'setting': self.setting,
            'expectation': self.expectation,
            'std_err': self.std_err,
            'total_counts': self.total_counts,
            'raw_expectation': self.raw_expectation,
            'raw_std_err': self.raw_std_err,
            'calibration_expectation': self.calibration_expectation,
            'calibration_std_err': self.calibration_std_err,
            'calibration_counts': self.calibration_counts,
        }
