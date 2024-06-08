import pytest
from typing_extensions import Callable

from pyquil.quil import Program


@pytest.fixture
def over_9000_program() -> Program:
    with open("test/benchmarks/fixtures/over-9000.quil") as f:
        return Program(f.read())


def test_iteration(benchmark: Callable, over_9000_program: Program) -> None:
    """Benchmark how long it takes the compatibility layer to iterate through a programs instructions."""
    def iteration(program: Program):
        for _ in range(100):
            for instruction in program:
                continue

    benchmark(iteration, over_9000_program)


def test_instructions(benchmark: Callable, over_9000_program: Program) -> None:
    """Benchmark how long it takes the compatibility layer to return the instructions property."""
    def instructions(program: Program) -> None:
        for _ in range(100):
            _ = program.instructions

    benchmark(instructions, over_9000_program)
