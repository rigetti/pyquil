from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3, get_qc as get_qc_v3
from ._qam import StatefulQAM


class QuantumComputer(QuantumComputerV3):
    pass


def get_qc(*args, **kwargs) -> QuantumComputer:
    qc = get_qc_v3(*args, **kwargs)
    StatefulQAM.wrap(qc.qam)
    return qc
