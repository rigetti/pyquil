from typing import Optional
from pyquil.quil import InstructionDesignator, Program


class ExperimentalProgram(Program):
    """
    An ExperimentalProgram is identical to a Program except that ``num_shots`` is optional.
    """

    num_shots: Optional[int]

    def __init__(self, *instructions: InstructionDesignator):
        super().__init__(*instructions)
        self.num_shots = None
