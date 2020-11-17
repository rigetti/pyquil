import pkgutil
import operator

from lark import Lark, Transformer, v_args

import numpy as np

from pyquil.quil import Program
from pyquil.quilbase import (
    DefGate,
    DefPermutationGate,
    DefGateByPaulis,
    # TODO DefCircuit
    DefWaveform,
    Qubit,
    FormalArgument,
    Frame,
    Pulse,
    Fence,
    FenceAll,
    DefCalibration,
    DefMeasureCalibration,
    DefFrame,
    Parameter,
    Declare,
    Capture,
    MemoryReference,
    Pragma,
    RawInstr,
    Nop,
    JumpTarget,
    Jump,
    JumpWhen,
    JumpUnless,
    Reset,
    ResetQubit,
    Wait,
    ClassicalStore,
    ClassicalLoad,
    ClassicalConvert,
    ClassicalExchange,
    ClassicalMove,
    ClassicalNeg,
    ClassicalNot,
    ClassicalTrue,
    ClassicalFalse,
    ClassicalAnd,
    ClassicalOr,
    ClassicalInclusiveOr,
    ClassicalExclusiveOr,
    ClassicalAdd,
    ClassicalSub,
    ClassicalMul,
    ClassicalDiv,
    ClassicalEqual,
    ClassicalGreaterEqual,
    ClassicalGreaterThan,
    ClassicalLessThan,
    ClassicalLessEqual,
)
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
    WaveformReference,
    Expression,
    quil_sqrt,
    quil_sin,
    quil_cos,
    quil_cis,
    quil_exp,
    Label,
)
from pyquil.gates import DELAY, SHIFT_PHASE, QUANTUM_GATES, MEASURE, Gate
from pyquil.paulis import PauliTerm


grammar = pkgutil.get_data("pyquil", "grammar.lark").decode()


class QuilTransformer(Transformer):
    def quil(self, instructions):
        return instructions

    indented_instrs = list

    @v_args(inline=True)
    def def_gate_matrix(self, name, variables, matrix):
        print(name)
        print(matrix)
        return DefGate(name, matrix=matrix, parameters=variables)

    @v_args(inline=True)
    def def_gate_as_permutation(self, name, matrix):
        return DefPermutationGate(name, permutation=matrix)

    # TODO(notmgsk): Remove?
    @v_args(inline=True)
    def gate_type(self, gate_type):
        valid_gate_types = (
            "MATRIX",
            "PERMUTATION",
        )
        if str(gate_type) in valid_gate_types:
            return str(gate_type)
        else:
            raise ValueError(
                f"Unsupported gate type in DEFGATE: {str(gate_type)}. "
                f"Supported gate types: {valid_gate_types}."
            )

    @v_args(inline=True)
    def def_pauli_gate(self, name, variables, qubits, terms):
        pg = DefGateByPaulis(name, parameters=variables, arguments=qubits, body=terms)
        return pg

    pauli_terms = list

    @v_args(inline=True)
    def pauli_term(self, name, expression, qubits):
        return PauliTerm.from_list(list(zip(name, qubits)), expression)

    @v_args(inline=True)
    def def_circuit(self, name, variables, qubits, instrs):
        # TODO(notmgsk): yuck
        qubits = qubits if qubits else []
        space = " " if qubits else ""
        if variables:
            raw_defcircuit = "DEFCIRCUIT {}({}){}{}:".format(
                name, ", ".join(variables), space, " ".join(map(str, qubits))
            )
        else:
            raw_defcircuit = "DEFCIRCUIT {}{}{}:".format(name, space, " ".join(map(str, qubits)))

        raw_defcircuit += "\n    ".join([""] + [str(instr) for instr in instrs])
        return RawInstr(raw_defcircuit)

    @v_args(inline=True)
    def def_frame(self, frame, *specs):
        names = {
            "DIRECTION": "direction",
            "HARDWARE-OBJECT": "hardware_object",
            "INITIAL-FREQUENCY": "initial_frequency",
            "SAMPLE-RATE": "sample_rate",
            "CENTER-FREQUENCY": "center_frequency",
        }
        options = {}

        for (spec_name, spec_value) in specs:
            name = names.get(spec_name, None)
            if name:
                options[name] = spec_value
            else:
                raise ValueError(
                    f"Unexpectected attribute {spec_name} in definition of frame {frame}. "
                    f"{frame}, {specs}"
                )

        f = DefFrame(frame, **options)
        return f

    frame_spec = list
    frame_attr = v_args(inline=True)(str)

    @v_args(inline=True)
    def def_waveform(self, name, params, matrix):
        return DefWaveform(name, params, matrix[0])

    @v_args(inline=True)
    def def_calibration(self, name, params, qubits, instructions):
        dc = DefCalibration(name, params, qubits, instructions)
        return dc

    @v_args(inline=True)
    def def_measure_calibration(self, qubit, name, instructions):
        mref = FormalArgument(name) if name else None
        dmc = DefMeasureCalibration(qubit, mref, instructions)
        return dmc

    @v_args(inline=True)
    def gate(self, modifiers, name, params, qubits):
        # TODO Don't like this.
        modifiers = modifiers or []
        params = params or []

        # Some gate modifiers increase the arity of the base gate. The
        # new qubit arguments prefix the old ones.
        modifier_qubits = []
        for m in modifiers:
            if m in ["CONTROLLED", "FORKED"]:
                modifier_qubits.append(qubits[len(modifier_qubits)])

        base_qubits = qubits[len(modifier_qubits):]
        forked_offset = len(params) >> modifiers.count("FORKED")
        base_params = params[:forked_offset]

        if name in QUANTUM_GATES:
            if base_params:
                gate = QUANTUM_GATES[name](*base_params, *base_qubits)
            else:
                gate = QUANTUM_GATES[name](*base_qubits)
        else:
            gate = Gate(name, base_params, base_qubits)

        for modifier in modifiers[::-1]:
            if modifier == "CONTROLLED":
                gate.controlled(modifier_qubits.pop())
            elif modifier == "DAGGER":
                gate.dagger()
            elif modifier == "FORKED":
                gate.forked(modifier_qubits.pop(), params[forked_offset : (2 * forked_offset)])
                forked_offset *= 2
            else:
                raise ValueError(f"Unsupported gate modifier {modifier}.")

        return gate

    modifiers = list
    modifier = v_args(inline=True)(str)

    @v_args(inline=True)
    def frame(self, qubits, name):
        f = Frame(qubits, name)
        return f

    @v_args(inline=True)
    def pulse(self, nonblocking, frame, waveform):
        p = Pulse(frame, waveform, nonblocking=bool(nonblocking))
        return p

    @v_args(inline=True)
    def fence_some(self, qubits):
        f = Fence(list(qubits))
        return f

    fence_all = v_args(inline=True)(FenceAll)

    @v_args(inline=True)
    def declare(self, name, memory_type, memory_size, *sharing):
        shared, *offsets = sharing
        d = Declare(
            str(name),
            str(memory_type),
            int(memory_size),
            shared_region=str(shared) if shared else None,
            offsets=offsets if shared else None,
        )
        return d

    @v_args(inline=True)
    def capture(self, nonblocking, frame, waveform, addr):
        c = Capture(frame, waveform, addr)
        return c

    @v_args(inline=True)
    def addr(self, name):
        return MemoryReference(str(name))

    @v_args(inline=True)
    def addr_subscript(self, name, subscript):
        return MemoryReference(str(name), int(subscript))

    @v_args(inline=True)
    def offset_descriptor(self, offset, name):
        return (int(offset), str(name))

    @v_args(inline=True)
    def delay_qubits(self, qubits, delay_amount):
        d = DELAY(*[*qubits, delay_amount])
        return d

    @v_args(inline=True)
    def delay_frames(self, qubit, *frames_and_delay_amount):
        *frame_names, delay_amount = frames_and_delay_amount
        frames = [Frame([qubit], name) for name in frame_names]
        d = DELAY(*[*frames, delay_amount])
        return d

    @v_args(inline=True)
    def shift_phase(self, frame, expression):
        return SHIFT_PHASE(frame, expression)

    @v_args(inline=True)
    def pragma(self, name, *pragma_names_and_string):
        if len(pragma_names_and_string) == 1:
            freeform_string = pragma_names_and_string[0]
            args = ()
        else:
            *args_identifiers, freeform_string = pragma_names_and_string
            args = map(str, args_identifiers)

        p = Pragma(str(name), args=args, freeform_string=freeform_string)
        return p

    @v_args(inline=True)
    def measure(self, qubit, address):
        return MEASURE(qubit, address)

    @v_args(inline=True)
    def nop(self):
        return Nop()

    @v_args(inline=True)
    def def_label(self, label):
        return JumpTarget(label)

    @v_args(inline=True)
    def jump(self, label):
        return Jump(label)

    @v_args(inline=True)
    def jump_when(self, label, address):
        return JumpWhen(label, address)

    @v_args(inline=True)
    def jump_unless(self, label, address):
        return JumpUnless(label, address)

    label = v_args(inline=True)(Label)

    @v_args(inline=True)
    def reset(self, qubit):
        if qubit:
            return ResetQubit(qubit)
        else:
            return Reset()

    @v_args(inline=True)
    def wait(self):
        return Wait()

    @v_args(inline=True)
    def store(self, left, subscript, right):
        return ClassicalStore(left, subscript, right)

    @v_args(inline=True)
    def load(self, left, right, subscript):
        return ClassicalLoad(left, right, subscript)

    @v_args(inline=True)
    def convert(self, left, right):
        return ClassicalConvert(left, right)

    @v_args(inline=True)
    def exchange(self, left, right):
        return ClassicalExchange(left, right)

    @v_args(inline=True)
    def move(self, left, right):
        return ClassicalMove(left, right)

    @v_args(inline=True)
    def classical_unary(self, op, target):
        if op == "TRUE":
            return ClassicalTrue(target)
        elif op == "FALSE":
            return ClassicalFalse(target)
        elif op == "NEG":
            return ClassicalNeg(target)
        elif op == "NOT":
            return ClassicalNot(target)

    @v_args(inline=True)
    def logical_binary_op(self, op, left, right):
        if op == "AND":
            return ClassicalAnd(left, right)
        elif op == "OR":
            return ClassicalOr(left, right)
        elif op == "IOR":
            return ClassicalInclusiveOr(left, right)
        elif op == "XOR":
            return ClassicalExclusiveOr(left, right)

    @v_args(inline=True)
    def arithmetic_binary_op(self, op, left, right):
        if op == "ADD":
            return ClassicalAdd(left, right)
        elif op == "SUB":
            return ClassicalSub(left, right)
        elif op == "MUL":
            return ClassicalMul(left, right)
        elif op == "DIV":
            return ClassicalDiv(left, right)

    @v_args(inline=True)
    def classical_comparison(self, op, target, left, right):
        if op == "EQ":
            return ClassicalEqual(target, left, right)
        elif op == "GT":
            return ClassicalGreaterThan(target, left, right)
        elif op == "GE":
            return ClassicalGreaterEqual(target, left, right)
        elif op == "LT":
            return ClassicalLessThan(target, left, right)
        elif op == "LE":
            return ClassicalLessEqual(target, left, right)

    @v_args(inline=True)
    def waveform(self, name, *params):
        param_dict = {k: v for (k, v) in params}
        if param_dict:
            return _wf_from_dict(name, param_dict)
        else:
            return WaveformReference(name)

    @v_args(inline=True)
    def waveform_name(self, prefix, suffix=None):
        return f"{prefix}/{suffix}" if suffix else prefix

    def matrix(self, rows):
        return list(rows)

    def matrix_row(self, expressions):
        return list(expressions)

    def params(self, params):
        return list(params)

    @v_args(inline=True)
    def named_param(self, name, val):
        return (str(name), val)

    def qubit_designators(self, qubits):
        return list(qubits)

    qubit = v_args(inline=True)(Qubit)
    qubits = list
    qubit_variable = v_args(inline=True)(FormalArgument)
    qubit_variables = list

    @v_args(inline=True)
    def variable(self, var):
        variable = Parameter(str(var))
        return variable

    def variables(self, variables):
        return list(variables)

    @v_args(inline=True)
    def i(self):
        return 1j

    @v_args(inline=True)
    def imag(self, number):
        return number * 1j

    @v_args(inline=True)
    def pi(self):
        return np.pi

    int_n = v_args(inline=True)(int)
    float_n = v_args(inline=True)(float)

    name = v_args(inline=True)(str)
    string = v_args(inline=True)(str)

    @v_args(inline=True)
    def signed_number(self, sign, number):
        if sign and sign == "-":
            return -number
        else:
            return number

    # TODO(notmgsk): These were originally lowercase. Is that required
    # by the language spec?
    @v_args(inline=True)
    def apply_fun(self, fun, arg):
        if fun == "SIN":
            return quil_sin(arg) if isinstance(arg, Expression) else np.sin(arg)
        if fun == "COS":
            return quil_cos(arg) if isinstance(arg, Expression) else np.cos(arg)
        if fun == "SQRT":
            return quil_sqrt(arg) if isinstance(arg, Expression) else np.sqrt(arg)
        if fun == "EXP":
            return quil_exp(arg) if isinstance(arg, Expression) else np.exp(arg)
        if fun == "CIS":
            return quil_cis(arg) if isinstance(arg, Expression) else np.cos(arg) + 1j * np.sin(arg)

    add = v_args(inline=True)(operator.add)
    sub = v_args(inline=True)(operator.sub)
    mul = v_args(inline=True)(operator.mul)
    div = v_args(inline=True)(operator.truediv)
    pow = v_args(inline=True)(operator.pow)
    neg = v_args(inline=True)(operator.neg)
    pos = v_args(inline=True)(operator.pos)


parser = Lark(
    grammar,
    start="quil",
    parser="lalr",
    transformer=QuilTransformer(),
    maybe_placeholders=True,
    #    debug=True,
)


def parse(program: str) -> Program:
    p = parser.parse(program)
    return p


# import json

# with open("/tmp/aspen-8-quilt-calibrations.txt") as f:
#     data = json.load(f)["quilt"]

# try:

#     from time import time

#     t = time()
#     p = Program(parser.parse(data))
#     print(f"{time() - t}s")
#     t = time()
#     p = Program(data)
#     print(f"{time() - t}s")
# except UnexpectedToken as e:
#     print(data.splitlines()[e.line - 1])
#     raise RuntimeError(e)
