import pytest
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple
from pyquil._parser.parser import run_parser
from pyquil.quil import Program


def from_corpus() -> List[Tuple[str, str]]:
    programs = []
    DIR = Path(__file__).parent / "quilc" / "tests" / "good-test-files"
    if not DIR.exists():
        subprocess.run(shlex.split("git submodule update --init --recursive"))
    for path in DIR.glob("*.quil"):
        with path.open() as f:
            program = f.read()
            try:
                run_parser(program)
                programs.append((path.name, program))
            except:
                continue  # TODO log these or something?
    return programs


@pytest.mark.parametrize("quil", from_corpus())
def test_parser(benchmark, quil):
    benchmark.group = "Parse: %s" % quil[0]
    benchmark((lambda: Program(run_parser(quil[1]))))
