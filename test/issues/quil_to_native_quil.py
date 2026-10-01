from pyquil import get_qc, Program
from pyquil.gates import MEASURE, H, X

# Declare our memory spaces
branching_prog = Program()
ro = branching_prog.declare("ro", "BIT")
test_register = branching_prog.declare("test_register", "BIT")

# Construct each branch of our if-statement. We can have empty branches
# simply by having empty programs.
then_branch = Program(X(0))
else_branch = Program()

# Construct our program so that the result in test_register is equally likely to be a 0 or 1
branching_prog += H(1)
branching_prog += MEASURE(1, test_register)

# Add the conditional branching
branching_prog.if_then(test_register, then_branch, else_branch)

# Measure qubit 0 into our readout register
branching_prog += MEASURE(0, ro)
branching_prog.resolve_label_placeholders()

# print(branching_prog)

# qpu_arch = "Cepheus-1-36Q-2"
# qpu_arch = "Aspen-M-3"
qpu_arch = "9q-square-qvm"
qc = get_qc(qpu_arch)

from quil.program import Program as RsProgram

basic = Program(
    RsProgram.parse(r"""
DECLARE ro BIT[1]
LABEL @top
JUMP @top
""")
)

print(branching_prog)
n = qc.compiler.quil_to_native_quil(branching_prog, protoquil=False)
print("----\n", n, type(n))
n = qc.compiler.native_quil_to_executable(n, protoquil=False)
print("----\n", n, type(n))

# executable = qc.compile(branching_prog, protoquil=False)
# print("----\n", executable)

# result = qc.run(branching_prog)
# print(result.get_register_map()['test_register'])
