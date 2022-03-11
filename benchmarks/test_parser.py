import pytest
import os
import subprocess
from pyquil._parser.parser import run_parser
from pyquil.quil import Program


def from_corpus():
    programs = []
    dir = os.path.join(os.path.dirname(__file__), "quilc", "tests", "good-test-files")
    if not os.path.exists(dir):
        subprocess.Popen(["git", "submodule", "update", "--init", "--recursive"]).wait()

    for path in os.listdir(dir):
        filepath = os.path.join(dir, path)
        if os.path.isfile(filepath):
            file = open(filepath, "r")
            program = file.read()
            try:
                run_parser(program)
                programs.append((path, program))
            except:
                continue
            finally:
                file.close()

    return programs


@pytest.mark.parametrize("quil", from_corpus())
def test_parser(benchmark, quil):
    benchmark.group = "Parse: %s" % quil[0]
    benchmark((lambda: Program(run_parser(quil[1]))))
