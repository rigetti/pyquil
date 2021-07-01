from typing import Optional, Sequence, TypeVar, Union, cast
import numpy as np

from rpcq.messages import ParameterAref
from pyquil.api._qam import QAM, QAMExecutionResult, QuantumExecutable

T = TypeVar("T")


class StatefulQAM(QAM[T]):
    _loaded_executable: Optional[QuantumExecutable]
    _result: Optional[QAMExecutionResult]

    @classmethod
    def wrap(cls, qam: QAM[T]) -> None:
        """
        Mutate the provided QAM to add methods and data for backwards compatibility,
        by dynamically mixing in this wrapper class.
        """
        if not isinstance(qam, StatefulQAM):
            qam.__class__ = type(str(qam.__class__.__name__), (StatefulQAM, qam.__class__), {})
            qam = cast(StatefulQAM[T], qam)
            qam.reset()

    def load(self, executable: QuantumExecutable) -> "StatefulQAM[T]":
        self._loaded_executable = executable.copy()  # copy here because calls to self.write_memory() will mutate it
        return self

    def read_memory(self, region_name: str) -> Optional[np.ndarray]:
        assert self._result is not None, "QAM#run must be called before QAM#read_memory"
        data = self._result.readout_data.get(region_name)
        return data

    def reset(self) -> "StatefulQAM[T]":
        self._loaded_executable = None
        self._result = None
        return self

    def run(self) -> "StatefulQAM[T]":  # type: ignore
        assert self._loaded_executable is not None
        self._result = super().run(self._loaded_executable)
        return self

    def wait(self) -> "StatefulQAM[T]":
        return self

    def write_memory(
        self,
        *,
        region_name: str,
        value: Union[int, float, Sequence[int], Sequence[float]],
        offset: Optional[int] = None,
    ) -> "StatefulQAM[T]":
        assert self._loaded_executable is not None, "Executable has not been loaded yet. Call QAM#load first"
        parameter_aref = ParameterAref(name=region_name, index=offset or 0)
        self._loaded_executable._memory._write_value(parameter=parameter_aref, value=value)
        return self
