from dataclasses import dataclass, field
from typing import Dict, Sequence, Union

from rpcq.messages import ParameterAref


@dataclass
class Memory:
    """
    Memory encapsulates the values to be sent as parameters alongside a program at time of
    execution, and read back afterwards.
    """

    values: Dict[ParameterAref, Union[int, float]] = field(default_factory=dict)

    def copy(self) -> "Memory":
        """
        Return a deep copy of this Memory object.
        """
        return Memory(values={k.replace(): v for k, v in self.values.items()})

    def write(self, parameter_values: Dict[Union[str, ParameterAref], Union[int, float]]) -> "Memory":
        """
        Set the given values for the given parameters.
        """
        for parameter, parameter_value in parameter_values.items():
            self._write_value(parameter=parameter, value=parameter_value)
        return self

    def _write_value(
        self,
        *,
        parameter: Union[ParameterAref, str],
        value: Union[int, float, Sequence[int], Sequence[float]],
    ) -> "Memory":
        """
        Set the given parameter to the given value.

        :param ParameterAref|str parameter: Name of the memory region, or parameter reference with offset.
        :param int|float|Sequence[int]|Sequence[float] value: the value or values to set for this parameter. If a list
        is provided, parameter must be a ``str`` or ``parameter.offset == 0``.
        """
        if isinstance(parameter, str):
            parameter = ParameterAref(name=parameter, index=0)

        if isinstance(value, (int, float)):
            self.values[parameter] = value
        elif isinstance(value, Sequence):
            if parameter.index != 0:
                raise ValueError("Parameter may not have a non-zero index when its value is a sequence")

            for index, v in enumerate(value):
                aref = ParameterAref(name=parameter.name, index=index)
                self.values[aref] = v

        return self
