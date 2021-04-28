from typing import Optional, Sequence, Union
from pyquil.api._qam import QAM, QAMMemory, QAMExecutionResult, QuantumExecutable


class StatefulQAM:
    _memory: Optional[QAMMemory]
    _loaded_executable: Optional[QuantumExecutable]
    _result: Optional[QAMExecutionResult]

    @classmethod
    def wrap(cls, qam: QAM) -> "StatefulQAM":
        """
        Mutate the provided QAM to add methods and data for backwards compatibility.
        """
        qam.__class__ = type("QAM", (qam.__class__, StatefulQAM), {})
        qam.reset()

    def load(self, executable: QuantumExecutable):
        self._loaded_executable = executable
        return self

    def run(self):
        execute_response = self.execute(executable=self._loaded_executable, memory=self._memory)
        self._result = self.get_results(execute_response)
        self._memory = self._result.memory
        return self

    def read_memory(self, region_name: str):
        return self._memory.read_memory(region_name=region_name)

    def reset(self):
        self._memory = QAMMemory()
        self._loaded_executable = None
        self._result = None
        return self

    def wait(self):
        return self

    def write_memory(
        self,
        *,
        region_name: str,
        value: Union[int, float, Sequence[int], Sequence[float]],
        offset: Optional[int] = None,
    ):
        self._memory.write_memory(region_name=region_name, value=value, offset=offset)
        return self
