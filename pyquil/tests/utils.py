from pyquil.parser import parse
from pyquil.api._qac import AbstractCompiler
from pyquil import Program


def parse_equals(quil_string, *instructions):
    expected = list(instructions)
    actual = parse(quil_string)
    assert expected == actual


def is_qvm_version_string(version_string: str) -> bool:
    parts = version_string.split('.')
    try:
        map(int, parts)
    except ValueError:
        return False
    return True


class DummyCompiler(AbstractCompiler):
    def get_version_info(self):
        return {}

    def quil_to_native_quil(self, program: Program, *, protoquil=None):
        return program

    def native_quil_to_executable(self, nq_program: Program):
        return nq_program
