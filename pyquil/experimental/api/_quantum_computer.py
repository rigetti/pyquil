from typing import Optional
from pyquil.api._quantum_computer import QuantumComputer
from pyquil.experimental._program import ExperimentalProgram
from pyquil.experimental.api._compiler import ExperimentalQPUCompiler, ExperimentalExecutable
from pyquil.experimental.api._qpu import ExperimentalQPU, ExperimentalQPUExecutionResult


class ExperimentalQuantumComputer:
    _compiler: ExperimentalQPUCompiler
    qam: ExperimentalQPU

    def __init__(self, *, qam: ExperimentalQPU, compiler: ExperimentalQPUCompiler) -> None:
        self.qam = qam
        self.compiler = compiler

    async def compile(self, program: ExperimentalProgram) -> ExperimentalExecutable:
        return await self.compiler.native_quil_to_executable(native_quil_program=program)

    async def run(self, executable: ExperimentalExecutable) -> ExperimentalQPUExecutionResult:
        return await self.qam.run(executable)


def get_experimental_qc(
    quantum_processor_id: str, *, compilation_timeout: Optional[int] = None, execution_timeout: Optional[int] = None
) -> ExperimentalQuantumComputer:
    compiler = ExperimentalQPUCompiler(quantum_processor_id=quantum_processor_id, timeout=compilation_timeout)
    qpu = ExperimentalQPU(quantum_processor_id=quantum_processor_id, timeout=execution_timeout)
    return ExperimentalQuantumComputer(qam=qpu, compiler=compiler)
