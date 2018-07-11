from abc import ABC
from typing import Iterable

import numpy as np

from pyquil import Program


class QAM(ABC):
    def run(self, quil_program: Program, classical_addresses: Iterable[int],
            trials: int) -> np.ndarray:
        """
        Run a pyQuil program on the QAM and return the values stored in the classical registers
        designated by the classical_addresses parameter. The program is repeated according to
        the number of trials provided to the run method.

        :param quil_program: Program to run on the QPU
        :param classical_addresses: Classical register addresses to return
        :param int trials: Number of times to run the program (a.k.a. number of shots)
        :return: A list of a list of classical registers (each register contains a bit)
        """
        raise NotImplementedError()

    def run_async(self, quil_program: Program, classical_addresses: Iterable[int], trials:int ):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to
        be executed. See https://go.rigetti.com/connections for reasons to use this method.
        """
        raise NotImplementedError()

    def wait_for_job(self, job_id, ping_time=None, status_time=None):
        raise NotImplementedError()
