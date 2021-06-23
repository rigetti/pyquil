from typing import Any
from pyquil.api._qpu import QPU as QPUV3
from ._qam import StatefulQAM


class QPU(QPUV3):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        StatefulQAM.wrap(self)
