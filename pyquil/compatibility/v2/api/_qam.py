from typing import Optional, Sequence, Union

from rpcq.messages import ParameterAref
from pyquil.api._qam import QAM, QAMExecutionResult, QuantumExecutable


class StatefulQAM(QAM):
    _loaded_executable: Optional[QuantumExecutable]
    _result: Optional[QAMExecutionResult]

    @classmethod
    def wrap(cls, qam: QAM) -> None:
        """
        Mutate the provided QAM to add methods and data for backwards compatibility,
        by dynamically mixing in this wrapper class.
        """
        qam.__class__ = type(str(qam.__class__.__name__), (StatefulQAM, qam.__class__), {})
        qam.reset()

    def load(self, executable: QuantumExecutable) -> "QAM":
        self._loaded_executable = executable
        return self

    def read_memory(self, region_name: str) -> "QAM":
        assert self._result is not None, "QAM#run must be called before QAM#read_memory"
        return self._result.read_memory(region_name=region_name)

    def reset(self) -> "QAM":
        self._loaded_executable = None
        self._result = None
        return self

    def run(self) -> "QAM":
        self._result = super().run(self._loaded_executable)
        return self

    def wait(self) -> "QAM":
        return self

    def write_memory(
        self,
        *,
        region_name: str,
        value: Union[int, float, Sequence[int], Sequence[float]],
        offset: Optional[int] = None,
    ) -> "QAM":
        assert self._loaded_executable is not None, "Executable has not been loaded yet. Call QAM#load first"
        parameter_aref = ParameterAref(name=region_name, offset=offset or 0)
        self._loaded_executable.set_parameter_value(parameter=parameter_aref, value=value)
        return self
