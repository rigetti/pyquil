from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from pyquil import Program


class QAM(ABC):
    @abstractmethod
    def run(self, quil_program: Program, classical_addresses: Iterable[int],
            trials: int) -> np.ndarray:
        """
        Run a Quil program on the QVM multiple times and return the values stored in the
        classical registers designated by the classical_addresses parameter.

        :param quil_program: A program to run
        :param classical_addresses: Classical register addresses to return
        :param int trials: Number of times to repeatedly run the program. This is sometimes called
            the number of shots.
        :return: An array of bitstrings of shape ``(trials, len(classical_addresses))``
        """

    @abstractmethod
    def run_async(self, quil_program: Program, classical_addresses: Iterable[int], trials: int):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to
        be executed. See https://go.rigetti.com/connections for reasons to use this method.
        """

    @abstractmethod
    def wait_for_job(self, job_id, ping_time=None, status_time=None):
        pass
