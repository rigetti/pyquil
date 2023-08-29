import json
import pkgutil
import operator
from typing import List

from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args

import numpy as np

from pyquil.quilbase import (
    AbstractInstruction,
    DefGate,
    DefPermutationGate,
    DefGateByPaulis,
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
    RawCapture,
    MemoryReference,
    Pragma,
    RawInstr,
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
    ClassicalAnd,
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
    _contained_mrefs,
)
from pyquil.gates import (
    DELAY,
    SHIFT_PHASE,
    SET_PHASE,
    SWAP_PHASES,
    SET_SCALE,
    SET_FREQUENCY,
    SHIFT_FREQUENCY,
    QUANTUM_GATES,
    MEASURE,
    HALT,
    NOP,
    Gate,
)


class QuilTransformer(Transformer):  # type: ignore
    def quil(self, instructions):
        return instructions

    indented_instrs = list

    @v_args(inline=True)
    def def_gate_matrix(self, name, variables, matrix):
        return DefGate(name, matrix=matrix, parameters=variables)

    @v_args(inline=True)
    def def_gate_as_permutation(self, name, matrix):
        return DefPermutationGate(name, permutation=matrix)

    @v_args(inline=True)
    def def_pauli_gate(self, name, variables, qubits, terms):
        pg = DefGateByPaulis(name, parameters=variables, arguments=qubits, body=terms)
        return pg

    pauli_terms = list

    @v_args(inline=True)
    def pauli_term(self, name, expression, qubits):
        from pyquil.paulis import PauliTerm

        return PauliTerm.from_list(list(zip(name, qubits)), expression)

    @v_args(inline=True)
    def def_circuit(self, name, variables, qubits, instrs):
        qubits = qubits if qubits else []
        space = " " if qubits else ""
        if variables:
            raw_defcircuit = "DEFCIRCUIT {}({}){}{}:".format(
                name, ", ".join(map(str, variables)), space, " ".join(map(str, qubits))
            )
        else:
            raw_defcircuit = "DEFCIRCUIT {}{}{}:".format(name, space, " ".join(map(str, qubits)))

        raw_defcircuit += "\n    ".join([""] + [str(instr) for instr in instrs])
        return RawInstr(raw_defcircuit)

    @v_args(inline=True)
    def def_circuit_without_qubits(self, name, variables, instrs):
        return self.def_circuit_qubits(name, variables, [], instrs)

    @v_args(inline=True)
    def def_frame(self, frame, *specs):
        names = {
            "DIRECTION": "direction",
            "HARDWARE-OBJECT": "hardware_object",
            "INITIAL-FREQUENCY": "initial_frequency",
            "SAMPLE-RATE": "sample_rate",
            "CENTER-FREQUENCY": "center_frequency",
            "ENABLE-RAW-CAPTURE": "enable_raw_capture",
            "CHANNEL-DELAY": "channel_delay",
        }
        options = {}

        for spec_name, spec_value in specs:
            name = names.get(spec_name, None)
            if name:
                options[name] = json.loads(str(spec_value))
            else:
                raise ValueError(
                    f"Unexpectected attribute {spec_name} in definition of frame {frame}. " f"{frame}, {specs}"
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
        for p in params:
            mrefs = _contained_mrefs(p)
            if mrefs:
                raise ValueError(f"Unexpected memory references {mrefs} in DEFCAL {name}. Did you forget a '%'?")
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

        base_qubits = qubits[len(modifier_qubits) :]
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

    @v_args(inline=True)
    def gate_no_qubits(self, name):
        return RawInstr(name)

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
            memory_type=str(memory_type),
            memory_size=int(memory_size) if memory_size else 1,
            shared_region=str(shared) if shared else None,
            offsets=offsets if shared else None,
        )
        return d

    @v_args(inline=True)
    def capture(self, nonblocking, frame, waveform, addr):
        c = Capture(frame, waveform, addr, nonblocking=nonblocking)
        return c

    @v_args(inline=True)
    def raw_capture(self, nonblocking, frame, expression, addr):
        c = RawCapture(frame, expression, addr, nonblocking=nonblocking)
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
    def delay_qubits(self, qubits, delay_amount=None):
        # TODO(notmgsk): This is a very nasty hack. I can't quite get
        # the Lark grammar to recognize the last token (i.e. 1) in
        # `DELAY 0 1` as the delay amount. I think it's because it
        # matches 1 as a qubit rather than an expression (in the
        # grammar). Then again I would expect look-ahead to see that
        # it matches expression too, so it should give that
        # preference. How do we encode that priority?
        if delay_amount is None:
            delay_amount = int(qubits[-1].index)
            qubits = qubits[:-1]
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
    def set_phase(self, frame, expression):
        return SET_PHASE(frame, expression)

    @v_args(inline=True)
    def set_scale(self, frame, expression):
        return SET_SCALE(frame, expression)

    @v_args(inline=True)
    def set_frequency(self, frame, expression):
        return SET_FREQUENCY(frame, expression)

    @v_args(inline=True)
    def shift_frequency(self, frame, expression):
        return SHIFT_FREQUENCY(frame, expression)

    @deprecated(version="3.5.1", reason="The correct instruction is SWAP-PHASES, not SWAP-PHASE")
    @v_args(inline=True)
    def swap_phase(self, framea, frameb):
        return SWAP_PHASES(framea, frameb)

    @versionadded(version="3.5.1", reason="The correct instruction is SWAP-PHASES, not SWAP-PHASE")
    @v_args(inline=True)
    def swap_phases(self, framea, frameb):
        return SWAP_PHASES(framea, frameb)

    @v_args(inline=True)
    def pragma(self, name, *pragma_names_and_string):
        args = list(map(str, pragma_names_and_string))
        p = Pragma(str(name), args=args)
        return p

    @v_args(inline=True)
    def pragma_freeform_string(self, name, *pragma_names_and_string):
        if len(pragma_names_and_string) == 1:
            freeform_string = pragma_names_and_string[0]
            args = ()
        else:
            *args_identifiers, freeform_string = pragma_names_and_string
            args = list(map(str, args_identifiers))
        # Strip the quotes from start/end of string which are matched
        # by the Lark grammar
        freeform_string = freeform_string[1:-1]
        p = Pragma(str(name), args=args, freeform_string=freeform_string)
        return p

    @v_args(inline=True)
    def measure(self, qubit, address):
        return MEASURE(qubit, address)

    @v_args(inline=True)
    def halt(self):
        return HALT

    @v_args(inline=True)
    def nop(self):
        return NOP

    @v_args(inline=True)
    def include(self, string):
        return RawInstr(f"INCLUDE {string}")

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
            return ClassicalMove(target, 1)
        elif op == "FALSE":
            return ClassicalMove(target, 0)
        elif op == "NEG":
            return ClassicalNeg(target)
        elif op == "NOT":
            return ClassicalNot(target)

    @v_args(inline=True)
    def logical_binary_op(self, op, left, right):
        if op == "AND":
            return ClassicalAnd(left, right)
        elif op == "OR":
            return ClassicalInclusiveOr(left, right)
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

    @v_args(inline=True)
    def apply_fun(self, fun, arg):
        if fun.upper() == "SIN":
            return quil_sin(arg) if isinstance(arg, Expression) else np.sin(arg)
        if fun.upper() == "COS":
            return quil_cos(arg) if isinstance(arg, Expression) else np.cos(arg)
        if fun.upper() == "SQRT":
            return quil_sqrt(arg) if isinstance(arg, Expression) else np.sqrt(arg)
        if fun.upper() == "EXP":
            return quil_exp(arg) if isinstance(arg, Expression) else np.exp(arg)
        if fun.upper() == "CIS":
            return quil_cis(arg) if isinstance(arg, Expression) else np.cos(arg) + 1j * np.sin(arg)

    add = v_args(inline=True)(operator.add)
    sub = v_args(inline=True)(operator.sub)
    mul = v_args(inline=True)(operator.mul)
    div = v_args(inline=True)(operator.truediv)
    pow = v_args(inline=True)(operator.pow)
    neg = v_args(inline=True)(operator.neg)
    pos = v_args(inline=True)(operator.pos)
    function = v_args(inline=True)(str)
    keyword = v_args(inline=True)(str)


grammar = pkgutil.get_data("pyquil._parser", "grammar.lark").decode()
parser = Lark(
    grammar,
    start="quil",
    parser="lalr",
    transformer=QuilTransformer(),
    maybe_placeholders=True,
)


def run_parser(program: str) -> List[AbstractInstruction]:
    """
    Parse a raw Quil program and return a corresponding list of PyQuil objects.

    :param str quil: a single or multiline Quil program
    :return: list of instructions
    """
    p = parser.parse(program)
    return p
