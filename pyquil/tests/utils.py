import os

from rpcq.messages import PyQuilExecutableResponse

from pyquil.parser import parse
from pyquil.api._qac import AbstractCompiler
from pyquil.api._compiler import _extract_attribute_dictionary_from_program
from pyquil import Program


def api_fixture_path(path: str) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "../api/tests/data", path)


def parse_equals(quil_string, *instructions):
    expected = list(instructions)
    actual = parse(quil_string)
    assert expected == actual


class DummyCompiler(AbstractCompiler):
    def get_version_info(self):
        return {}

    def quil_to_native_quil(self, program: Program, *, protoquil=None):
        return program

    def native_quil_to_executable(self, nq_program: Program):
        return PyQuilExecutableResponse(
            program=nq_program.out(),
            attributes=_extract_attribute_dictionary_from_program(nq_program),
        )
