from pyquil.api._qvm import QVM as QVMV3
from ._qam import StatefulQAM


class QVM(QVMV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        StatefulQAM.wrap(self)
