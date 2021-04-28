from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3, get_qc as get_qc_v3


class QuantumComputer(QuantumComputerV3):
    pass


def get_qc(*args, **kwargs) -> QuantumComputer:
    return get_qc_v3(*args, **kwargs)
