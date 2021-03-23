import os

from pyquil.api import Client
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.parser import parse
from pyquil.api._abstract_compiler import AbstractCompiler
from pyquil import Program


def api_fixture_path(path: str) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "../api/tests/data", path)


def parse_equals(quil_string, *instructions):
    expected = list(instructions)
    actual = parse(quil_string)
    assert expected == actual


class DummyCompiler(AbstractCompiler):
    def __init__(self, quantum_processor: AbstractQuantumProcessor, client: Client):
        super().__init__(quantum_processor=quantum_processor, client=client, timeout=10)  # type: ignore

    def get_version_info(self):
        return {}

    def quil_to_native_quil(self, program: Program, *, protoquil=None):
        return program

    def native_quil_to_executable(self, nq_program: Program):
        return nq_program
