import os
import numpy as np

from pyquil.api._qpu import QPU
from pyquil import Program, get_qc


QPID = os.getenv("QPID", "Aspen-11")


qc = get_qc(QPID)
program = Program(
    "DECLARE ro BIT[2]",
    "DECLARE theta REAL",
    "RX(theta) 0",
    "X 0",
    "CNOT 0 1",
    "MEASURE 0 ro[0]",
    "MEASURE 1 ro[1]",
)
program.wrap_in_numshots_loop(10)
executable = qc.compile(program)
executable.write_memory(region_name="theta", value=np.pi)

# qpu = QPU(quantum_processor_id=QPID)
# execute_response = qpu.execute(executable=executable)
execute_response = qc.run(executable=executable)
print(execute_response)
