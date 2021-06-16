##############################################################################
# Copyright 2018 Rigetti Computing
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, TypeVar

import numpy as np
from pyquil.api._abstract_compiler import QuantumExecutable
from pyquil.experiment._main import Experiment


class QAMError(RuntimeError):
    pass


T = TypeVar("T")
"""A generic parameter describing the opaque job handle returned from QAM#execute and subclasses."""


@dataclass
class QAMExecutionResult:
    executable: QuantumExecutable
    memory: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)

    def read_memory(self, *, region_name: str) -> Optional[np.ndarray]:
        """
        Reads from a memory region named ``region_name`` returned from the QAM.

        :param region_name: The string naming the declared memory region.
        :return: A list of values of the appropriate type.
        """
        assert self.memory is not None, "No memory results available"
        return self.memory.get(region_name)


class QAM(ABC, Generic[T]):
    """
    Quantum Abstract Machine: This class acts as a generic interface describing how a classical
    computer interacts with a live quantum computer.
    """

    def __init__(self) -> None:
        self.experiment: Optional[Experiment]

    @abstractmethod
    def execute(self, executable: QuantumExecutable) -> T:
        """
        Run an executable on a QAM, returning a handle to be used to retrieve
        results.

        :param executable: The executable program to be executed by the QAM.
        """

    @abstractmethod
    def get_result(self, execute_response: T) -> QAMExecutionResult:
        """
        Retrieve the results associated with a previous call to ``QAM#execute``.

        :param execute_response: The return value from a call to ``execute``.
        """

    def run(self, executable: QuantumExecutable) -> QAMExecutionResult:
        """
        Run an executable to completion on the QAM.
        """
        return self.get_result(self.execute(executable))
