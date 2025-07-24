from pathlib import Path
from typing import Callable

import pytest

from pyquil.quil import Program

@pytest.fixture(scope="session")
def over_9000_program_text() -> str:
    return Path("test/benchmarks/fixtures/over-9000.quil").read_text()

@pytest.fixture
def large_program_text() -> str:
    return Path("test/benchmarks/fixtures/large_with_calibrations.quil").read_text()

@pytest.fixture
def over_9000_program(over_9000_program_text: str) -> Program:
    return Program(over_9000_program_text)

@pytest.fixture
def large_program(large_program_text: str) -> Program:
    return Program(large_program_text)

def test_build_program(benchmark, over_9000_program_text: str):
    benchmark(Program, over_9000_program_text)

def test_iteration(benchmark: Callable, over_9000_program: Program) -> None:
    """Benchmark how long it takes the compatibility layer to iterate through a programs instructions."""
    def iteration(program: Program):
        for _ in program:
            continue

    benchmark(iteration, over_9000_program)


def test_instructions(benchmark: Callable, over_9000_program: Program) -> None:
    """Benchmark how long it takes the compatibility layer to return the instructions property."""
    def instructions(program: Program) -> None:
        _ = program.instructions

    benchmark(instructions, over_9000_program)

@pytest.mark.skip("this is too slow")
def test_copy_everything_except_instructions(benchmark: Callable, large_program: Program) -> None:
    benchmark(large_program.copy_everything_except_instructions)
