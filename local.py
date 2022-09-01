import os
import asyncio
import numpy as np
from unittest import result
from pyquil.api._compiler import LocalCompiler
from pyquil.api._qpu import QPU
from pyquil import Program, get_qc


QPID = os.getenv("QPID", "Aspen-12")


async def main():
    qc = get_qc(QPID)
    p = Program(
        "DECLARE ro BIT[2]",
        "DECLARE theta REAL",
        "RX(theta) 0",
        "X 0",
        "CNOT 0 1",
        "MEASURE 0 ro[0]",
        "MEASURE 1 ro[1]"
    )
    p.wrap_in_numshots_loop(10)
    lc = LocalCompiler(quantum_processor=qc.quantum_processor, quantum_processor_id=QPID)

    native = await lc.quil_to_native_quil(p)
    executable = await lc.native_quil_to_executable(native)
    executable.write_memory(region_name='theta', value=np.pi)

    qpu = QPU(quantum_processor_id=QPID)
    execute_response = qpu.execute(executable=executable)
    print(execute_response)
    # result_response = qpu.get_result(execute_response=execute_response)
    # print(result_response.execution_duration_microseconds, result_response.readout_data)

if __name__ == "__main__":
    asyncio.run(main())
