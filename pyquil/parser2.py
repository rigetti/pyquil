from typing import List
import operator

from lark import Lark, Transformer, v_args
from lark.exceptions import UnexpectedToken

import numpy as np

from pyquil.quil import Program
from pyquil.quilbase import (
    AbstractInstruction,
    DefWaveform,
    Qubit,
    FormalArgument,
    Frame,
    Pulse,
    Fence,
    DefCalibration,
    DefFrame,
    Parameter,
    Declare,
    Capture,
    MemoryReference,
    Pragma,
)
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import WaveformReference
from pyquil.gates import DELAY, SHIFT_PHASE

grammar = """
quil : all_instr*

?all_instr : def_frame
           | def_waveform
           | def_calibration

?instr : fence
       | pulse
       | delay
       | shift_phase
       | declare
       | capture
       | pragma

def_frame : "DEFFRAME" frame ( ":" frame_spec+ )?
frame_spec : _NEWLINE_TAB frame_attr ":" (expression | "\\"" name "\\"" )
!frame_attr : "SAMPLE-RATE"
            | "INITIAL-FREQUENCY"
            | "DIRECTION"
            | "HARDWARE-OBJECT"
            | "CENTER-FREQUENCY"

def_waveform : "DEFWAVEFORM" waveform_name params ":" matrix

def_calibration : "DEFCAL" name params qubit_designators ":" indented_instrs

indented_instrs : ( (_NEWLINE_TAB instr)* )?

params : ( "(" param ("," param)* ")" )?
?param : expression

matrix : ( _NEWLINE_TAB matrix_row )*
matrix_row : expression ( "," expression )*

?expression         : product
                    | expression "+" product -> add
                    | expression "-" product -> sub
?product            : power
                    | product "*" power -> mul
                    | product "/" power -> div
?power              : atom
                    | atom "^" power -> pow
?atom               : number
                    | "-" atom -> neg
                    | "+" atom -> pos
                    | FUNCTION "(" expression ")" -> apply_fun
                    | "(" expression ")"
                    | variable

variable : "%" IDENTIFIER

// Instructions

fence : "FENCE" qubit_designators

pulse : [ NONBLOCKING ] "PULSE" frame waveform

?delay : delay_qubits | delay_frames
delay_qubits : "DELAY" qubit_designators expression
delay_frames : "DELAY" qubit_designator ( "\\"" name "\\"" )+ expression

shift_phase : "SHIFT-PHASE" frame expression

declare : "DECLARE" IDENTIFIER IDENTIFIER [ "[" INT "]" ] [ "SHARING" IDENTIFIER ( offset_descriptor )* ]
?offset_descriptor : "OFFSET" INT IDENTIFIER

capture : [ NONBLOCKING ] "CAPTURE" frame waveform addr
addr : IDENTIFIER -> addr | ( [ IDENTIFIER ] "[" INT "]" ) -> addr_subscript

pragma : "PRAGMA" ( IDENTIFIER | keyword ) pragma_name* string
// Why can't I int_n here?
!pragma_name : IDENTIFIER | keyword | INT

// Qubits, frames, waveforms
qubit_designators : ( qubit_designator+ )?
?qubit_designator : qubit | qubit_variable
qubit : int_n
qubit_variable : IDENTIFIER
named_param : IDENTIFIER ":" expression
waveform : waveform_name ( "(" named_param ( "," named_param )* ")" )?
waveform_name : name ( "/" name )?
frame : qubit_designators+ "\\"" name "\\""

FUNCTION            : "sin" | "cos" | "sqrt" | "exp" | "cis"
// Numbers
?number             : (int_n|float_n) "i" -> imag
                    | int_n
                    | float_n
                    | "i" -> i
                    | "pi" -> pi

int_n               : INT
float_n             : FLOAT


// Lexer
keyword             : DEFGATE | DEFCIRCUIT | MEASURE | LABEL | HALT | JUMP | JUMPWHEN | JUMPUNLESS
                    | RESET | WAIT | NOP | INCLUDE | PRAGMA | DECLARE | SHARING | OFFSET | AS | MATRIX
                    | PERMUTATION | NEG | NOT | TRUE | FALSE | AND | IOR | XOR | OR | ADD | SUB | MUL
                    | DIV | MOVE | EXCHANGE | CONVERT | EQ | GT | GE | LT | LE | LOAD | STORE | PI | I
                    | SIN | COS | SQRT | EXP | CIS | CAPTURE | DEFCAL | DEFFRAME | DEFWAVEFORM
                    | DELAY | FENCE | INITIALFREQUENCY | CENTERFREQUENCY | NONBLOCKING | PULSE | SAMPLERATE
                    | SETFREQUENCY | SETPHASE | SETSCALE | SHIFTPHASE | SWAPPHASE | RAWCAPTURE | FILTERNODE
                    | CONTROLLED | DAGGER | FORKED

// Keywords

DEFGATE             : "DEFGATE"
DEFCIRCUIT          : "DEFCIRCUIT"
MEASURE             : "MEASURE"

LABEL               : "LABEL"
HALT                : "HALT"
JUMP                : "JUMP"
JUMPWHEN            : "JUMP-WHEN"
JUMPUNLESS          : "JUMP-UNLESS"

RESET               : "RESET"
WAIT                : "WAIT"
NOP                 : "NOP"
INCLUDE             : "INCLUDE"
PRAGMA              : "PRAGMA"

DECLARE             : "DECLARE"
SHARING             : "SHARING"
OFFSET              : "OFFSET"

AS                  : "AS"
MATRIX              : "MATRIX"
PERMUTATION         : "PERMUTATION"
PAULISUM            : "PAULI-SUM"

NEG                 : "NEG"
NOT                 : "NOT"
TRUE                : "TRUE"  // Deprecated
FALSE               : "FALSE"  // Deprecated

AND                 : "AND"
IOR                 : "IOR"
XOR                 : "XOR"
OR                  : "OR"    // Deprecated

ADD                 : "ADD"
SUB                 : "SUB"
MUL                 : "MUL"
DIV                 : "DIV"

MOVE                : "MOVE"
EXCHANGE            : "EXCHANGE"
CONVERT             : "CONVERT"

EQ                  : "EQ"
GT                  : "GT"
GE                  : "GE"
LT                  : "LT"
LE                  : "LE"

LOAD                : "LOAD"
STORE               : "STORE"

PI                  : "pi"
I                   : "i"

SIN                 : "SIN"
COS                 : "COS"
SQRT                : "SQRT"
EXP                 : "EXP"
CIS                 : "CIS"

// Operators

PLUS                : "+"
MINUS               : "-"
TIMES               : "*"
DIVIDE              : "/"
POWER               : "^"

// analog keywords

CAPTURE             : "CAPTURE"
DEFCAL              : "DEFCAL"
DEFFRAME            : "DEFFRAME"
DEFWAVEFORM         : "DEFWAVEFORM"
DELAY               : "DELAY"
FENCE               : "FENCE"
HARDWAREOBJECT      : "HARDWARE-OBJECT"
INITIALFREQUENCY    : "INITIAL-FREQUENCY"
CENTERFREQUENCY     : "CENTER-FREQUENCY"
NONBLOCKING         : "NONBLOCKING"
PULSE               : "PULSE"
SAMPLERATE          : "SAMPLE-RATE"
SETFREQUENCY        : "SET-FREQUENCY"
SHIFTFREQUENCY      : "SHIFT-FREQUENCY"
SETPHASE            : "SET-PHASE"
SETSCALE            : "SET-SCALE"
SHIFTPHASE          : "SHIFT-PHASE"
SWAPPHASE           : "SWAP-PHASE"
RAWCAPTURE          : "RAW-CAPTURE"
FILTERNODE          : "FILTER-NODE"

// Modifiers

CONTROLLED          : "CONTROLLED"
DAGGER              : "DAGGER"
FORKED              : "FORKED"

// Common
name                : IDENTIFIER
IDENTIFIER          : ("_"|LETTER) [ ("_"|"-"|LETTER|DIGIT)* ("_"|LETTER|DIGIT) ]
string              : ESCAPED_STRING
_NEWLINE_TAB        : NEWLINE "    "
%import common.DIGIT
%import common.ESCAPED_STRING
%import common._STRING_INNER
%import common.FLOAT
%import common.INT
%import common.LETTER
%import common.NEWLINE
%import common.WS
%ignore WS

"""


class QuilTransformer(Transformer):
    def quil(self, instructions):
        return instructions

    def indented_instrs(self, instrs):
        return list(instrs)

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
    def frame(self, qubits, name):
        f = Frame(qubits, name)
        return f

    @v_args(inline=True)
    def pulse(self, nonblocking, frame, waveform):
        p = Pulse(frame, waveform, nonblocking=bool(nonblocking))
        return p

    @v_args(inline=True)
    def fence(self, qubits):
        f = Fence(list(qubits))
        return f

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
    qubit_variable = v_args(inline=True)(FormalArgument)

    @v_args(inline=True)
    def variable(self, var):
        variable = Parameter(str(var))
        return variable

    @v_args(inline=True)
    def i(self):
        return 1j

    @v_args(inline=True)
    def pi(self):
        return np.pi

    int_n = v_args(inline=True)(int)
    float_n = v_args(inline=True)(float)

    name = v_args(inline=True)(str)
    string = v_args(inline=True)(str)

    def apply_fun(self, fun, arg):
        if fun == "sin":
            return np.sin(arg)
        if fun == "cos":
            return np.cos(arg)
        if fun == "sqrt":
            return np.sqrt(arg)
        if fun == "exp":
            return np.exp(arg)
        if fun == "cis":
            return np.cos(arg) + 1j * np.sin(arg)

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
    debug=True,
)


def parse(program: str) -> Program:
    p = Program(parser.parse(program))
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
