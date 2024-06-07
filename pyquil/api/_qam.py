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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np
from deprecated import deprecated
from qcs_sdk import ExecutionData
from qcs_sdk.qpu import MemoryValues, RawQPUReadoutData
from qcs_sdk.qvm import RawQVMReadoutData

from pyquil.api._abstract_compiler import QuantumExecutable


class QAMError(RuntimeError):
    pass


T = TypeVar("T")
"""A generic parameter describing the opaque job handle returned from QAM#execute and subclasses."""

MemoryMap = Mapping[str, Union[Sequence[int], Sequence[float]]]
"""A mapping of memory regions to a list containing the values to be written into that memory region."""


@dataclass
class QAMExecutionResult:
    executable: QuantumExecutable
    """The executable corresponding to this result."""

    data: ExecutionData
    """
    The ``ExecutionData`` returned from the job. Consider using
    ``QAMExecutionResult#register_map`` or ``QAMExecutionResult#raw_readout_data``
    to get at the data in a more convenient format.
    """

    def get_register_map(self) -> dict[str, Optional[np.ndarray]]:
        """Map a register name (ie. "ro") to a ``np.ndarray`` containing the values for the register.

        Raises a ``RegisterMatrixConversionError`` if the inner execution data for any of the
        registers would result in a jagged matrix. QPU result data is captured per measure,
        meaning a value is returned for every measure to a memory reference, not just once per shot.
        This is often the case in programs that re-use qubits or dynamic control flow, where
        measurements to the same memory reference might occur multiple times in a shot, or be skipped
        conditionally. In these cases, building a matrix with one value per memory reference, per shot
        would necessitate making assumptions about the data that could skew the data in undesirable
        ways. Instead, it's recommended to manually build a matrix from the ``QPUResultData`` available
        on the ``raw_readout_data`` property.

        .. warning::

            An exception will _not_ be raised if the result data happens to fit a rectangular matrix, since
            it's possible the register map is valid for some number of shots. Users should be aware of this
            possibility, especially when running programs that utilize qubit reuse or dynamic control flow.

        """
        register_map = self.data.result_data.to_register_map()
        return {key: matrix.to_ndarray() for key, matrix in register_map.items()}

    def get_raw_readout_data(self) -> Union[RawQVMReadoutData, RawQPUReadoutData]:
        """Get the raw result data.

        This will be a flattened structure derived
        from :class:`qcs_sdk.qvm.QVMResultData` or :class:`qcs_sdk.qpu.QPUResultData`
        depending on where the job was run. See their respective documentation
        for more information on the data format.

        This property should be used when running programs that use features like
        mid-circuit measurement and dynamic control flow on a QPU, since they can
        produce irregular result shapes that don't necessarily fit in a
        rectangular matrix. If the program was run on a QVM, or doesn't use those
        features, consider using the ``register_map`` property instead.
        """
        return self.data.result_data.to_raw_readout_data()

    def get_memory_values(self) -> Mapping[str, MemoryValues]:
        """Get the final memory values for any memory region that was both read from and written to during execution.

        This method will only return the final value in memory after the job has completed. Because of this, memory
        values should not be used to get readout data. Instead, use `get_register_map()` or `get_raw_readout_data()`.
        """
        if self.data.result_data.is_qpu():
            return self.data.result_data.to_qpu().memory_values
        return {}

    @property
    @deprecated(
        version="4.0.0",
        reason=(
            "This property is ambiguous now that the `get_raw_readout_data()` method exists "
            "and will be removed in future versions. Use the `get_register_map()` method instead."
        ),
    )
    def readout_data(self) -> Mapping[str, Optional[np.ndarray]]:
        """Readout data returned from the QAM, keyed on the name of the readout register or post-processing node."""
        return self.get_register_map()

    @property
    def execution_duration_microseconds(self) -> Optional[float]:
        """Duration job held exclusive hardware access. Defaults to ``None`` when information is not available."""
        if isinstance(self.data.duration, timedelta):
            return self.data.duration.total_seconds() * 1e6
        return None


class QAM(ABC, Generic[T]):
    """Quantum Abstract Machine: An interface describing how a classical computer interacts with a quantum computer."""

    @abstractmethod
    def execute(
        self,
        executable: QuantumExecutable,
        memory_map: Optional[MemoryMap] = None,
        **kwargs: Any,
    ) -> T:
        """Run an executable on a QAM, returning a handle to be used to retrieve results.

        :param executable: The executable program to be executed by the QAM.
        :param memory_map: A mapping of memory regions to a list containing the values to be written into that memory
            region for the run.
        """

    @abstractmethod
    def execute_with_memory_map_batch(
        self,
        executable: QuantumExecutable,
        memory_maps: Iterable[MemoryMap],
        **kwargs: Any,
    ) -> list[T]:
        """Execute a QuantumExecutable with one or more memory_maps, returning handles to be used to retrieve results.

        How these programs are batched and executed is determined by the executor. See their respective documentation
        for details.

        Returns a list of handles that can be used to fetch results with ``QAM#get_result``.
        """

    @abstractmethod
    def get_result(self, execute_response: T) -> QAMExecutionResult:
        """Retrieve the results associated with a previous call to ``QAM#execute``.

        :param execute_response: The return value from a call to ``execute``.
        """

    def run(
        self, executable: QuantumExecutable, memory_map: Optional[MemoryMap] = None, **kwargs: Any
    ) -> QAMExecutionResult:
        """Run an executable to completion on the QAM."""
        return self.get_result(self.execute(executable, memory_map, **kwargs))
