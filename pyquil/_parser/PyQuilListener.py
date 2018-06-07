##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import operator
from typing import Any, List, Iterator, Callable
from numbers import Number
import sys

import numpy as np
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import CommonToken
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import InputMismatchException
from numpy.ma import sin, cos, sqrt, exp

from pyquil import parameters
from pyquil.gates import STANDARD_GATES
from pyquil.parameters import Parameter, Expression, Segment
from pyquil.quilbase import Gate, DefGate, Measurement, Addr, JumpTarget, Label, Halt, Jump, JumpWhen, JumpUnless, \
    Reset, Wait, ClassicalTrue, ClassicalFalse, ClassicalNot, ClassicalAnd, ClassicalOr, ClassicalMove, \
    ClassicalExchange, Nop, RawInstr, Qubit, Pragma, AbstractInstruction

if sys.version_info.major == 2:
    from .gen2.QuilLexer import QuilLexer
    from .gen2.QuilListener import QuilListener
    from .gen2 .QuilParser import QuilParser
elif sys.version_info.major == 3:
    from .gen3.QuilLexer import QuilLexer
    from .gen3.QuilListener import QuilListener
    from .gen3.QuilParser import QuilParser


def run_parser(quil):
    # type: (str) -> List[AbstractInstruction]
    """
    Run the ANTLR parser.

    :param str quil: a single or multiline Quil program
    :return: list of instructions that were parsed
    """
    # Step 1: Run the Lexer
    input_stream = InputStream(quil)
    lexer = QuilLexer(input_stream)
    stream = CommonTokenStream(lexer)

    # Step 2: Run the Parser
    parser = QuilParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(CustomErrorListener())
    tree = parser.quil()

    # Step 3: Run the Listener
    pyquil_listener = PyQuilListener()
    walker = ParseTreeWalker()
    walker.walk(pyquil_listener, tree)

    return pyquil_listener.result


class CustomErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        # type: (QuilParser, CommonToken, int, int, str, InputMismatchException) -> None
        expected_tokens = self.get_expected_tokens(recognizer, e.getExpectedTokens()) if e else []

        raise RuntimeError(
            "Error encountered while parsing the quil program at line {} and column {}\n".format(line, column + 1) +
            "Received an '{}' but was expecting one of [ {} ]".format(offendingSymbol.text, ', '.join(expected_tokens))
        )

    def get_expected_tokens(self, parser, interval_set):
        # type: (QuilParser, IntervalSet) -> Iterator
        """
        Like the default getExpectedTokens method except that it will fallback to the rule name if the token isn't a
        literal. For instance, instead of <INVALID> for  integer it will return the rule name: INT
        """
        for tok in interval_set:
            literal_name = parser.literalNames[tok]
            symbolic_name = parser.symbolicNames[tok]

            if literal_name != '<INVALID>':
                yield literal_name
            else:
                yield symbolic_name


class PyQuilListener(QuilListener):
    """
    Functions are invoked when the parser reaches the various different constructs in Quil.
    """
    def __init__(self):
        # type: () -> None
        self.result = []    # type: List[AbstractInstruction]

    def exitDefGate(self, ctx):
        # type: (QuilParser.DefGateContext) -> None
        gate_name = ctx.name().getText()
        matrix = _matrix(ctx.matrix())
        parameters = list(map(_variable, ctx.variable()))
        self.result.append(DefGate(gate_name, matrix, parameters))

    def exitDefCircuit(self, ctx):
        # type: (QuilParser.DefCircuitContext) -> None
        self.result.append(RawInstr(ctx.getText()))

    def exitGate(self, ctx):
        # type: (QuilParser.GateContext) -> None
        gate_name = ctx.name().getText()
        params = list(map(_param, ctx.param()))
        qubits = list(map(_qubit, ctx.qubit()))

        if gate_name in STANDARD_GATES:
            if params:
                self.result.append(STANDARD_GATES[gate_name](*params)(*qubits))
            else:
                self.result.append(STANDARD_GATES[gate_name](*qubits))
        else:
            self.result.append(Gate(gate_name, params, qubits))

    def exitMeasure(self, ctx):
        # type: (QuilParser.MeasureContext) -> None
        qubit = _qubit(ctx.qubit())
        classical = None
        if ctx.addr():
            classical = _addr(ctx.addr())
        self.result.append(Measurement(qubit, classical))

    def exitDefLabel(self, ctx):
        # type: (QuilParser.DefLabelContext) -> None
        self.result.append(JumpTarget(_label(ctx.label())))

    def exitHalt(self, ctx):
        # type: (QuilParser.HaltContext) -> None
        self.result.append(Halt())

    def exitJump(self, ctx):
        # type: (QuilParser.JumpContext) -> None
        self.result.append(Jump(_label(ctx.label())))

    def exitJumpWhen(self, ctx):
        # type: (QuilParser.JumpWhenContext) -> None
        self.result.append(JumpWhen(_label(ctx.label()), _addr(ctx.addr())))

    def exitJumpUnless(self, ctx):
        # type: (QuilParser.JumpUnlessContext) -> None
        self.result.append(JumpUnless(_label(ctx.label()), _addr(ctx.addr())))

    def exitResetState(self, ctx):
        # type: (QuilParser.ResetStateContext) -> None
        self.result.append(Reset())

    def exitWait(self, ctx):
        # type: (QuilParser.WaitContext) -> None
        self.result.append(Wait())

    def exitClassicalUnary(self, ctx):
        # type: (QuilParser.ClassicalUnaryContext) -> None
        if ctx.TRUE():
            self.result.append(ClassicalTrue(_addr(ctx.addr())))
        elif ctx.FALSE():
            self.result.append(ClassicalFalse(_addr(ctx.addr())))
        elif ctx.NOT():
            self.result.append(ClassicalNot(_addr(ctx.addr())))

    def exitClassicalBinary(self, ctx):
        # type: (QuilParser.ClassicalBinaryContext) -> None
        if ctx.AND():
            self.result.append(ClassicalAnd(_addr(ctx.addr(0)), _addr(ctx.addr(1))))
        elif ctx.OR():
            self.result.append(ClassicalOr(_addr(ctx.addr(0)), _addr(ctx.addr(1))))
        elif ctx.MOVE():
            self.result.append(ClassicalMove(_addr(ctx.addr(0)), _addr(ctx.addr(1))))
        elif ctx.EXCHANGE():
            self.result.append(ClassicalExchange(_addr(ctx.addr(0)), _addr(ctx.addr(1))))

    def exitNop(self, ctx):
        # type: (QuilParser.NopContext) -> None
        self.result.append(Nop())

    def exitInclude(self, ctx):
        # type: (QuilParser.IncludeContext) -> None
        self.result.append(RawInstr(ctx.INCLUDE().getText() + ' ' + ctx.STRING().getText()))

    def exitPragma(self, ctx):
        # type: (QuilParser.PragmaContext) -> None
        args = list(map(lambda x: x.getText(), ctx.pragma_name()))
        if ctx.STRING():
            # [1:-1] is used to strip the quotes from the parsed string
            self.result.append(Pragma(ctx.IDENTIFIER().getText(), args, ctx.STRING().getText()[1:-1]))
        else:
            self.result.append(Pragma(ctx.IDENTIFIER().getText(), args))


"""
Helper functions for converting from ANTLR internals to PyQuil objects
"""


def _qubit(qubit):
    # type: (QuilParser.QubitContext) -> Qubit
    return Qubit(int(qubit.getText()))


def _param(param):
    # type: (QuilParser.ParamContext) -> Any
    if param.expression():
        return _expression(param.expression())
    else:
        raise RuntimeError("Unexpected param: " + param.getText())


def _variable(variable):
    # type: (QuilParser.VariableContext) -> Parameter
    return Parameter(variable.IDENTIFIER().getText())


def _matrix(matrix):
    # type: (QuilParser.MatrixContext) -> List[List[Any]]
    out = []
    for row in matrix.matrixRow():
        out.append(list(map(_expression, row.expression())))
    return out


def _addr(classical):
    # type: (QuilParser.AddrContext) -> Addr
    return Addr(int(classical.classicalBit().getText()))


def _segment(segment):
    # type: (QuilParser.SegmentContext) -> Segment
    ints = segment.INT()
    return Segment(int(ints[0].getText()), int(ints[1].getText()))


def _label(label):
    # type: (QuilParser.LabelContext) -> Label
    return Label(label.IDENTIFIER().getText())


def _expression(expression):
    # type: (QuilParser.ExpressionContext) -> Any
    """
    NB: Order of operations is already dealt with by the grammar. Here we can simply match on the type.
    """
    if isinstance(expression, QuilParser.ParenthesisExpContext):
        return _expression(expression.expression())
    elif isinstance(expression, QuilParser.PowerExpContext):
        if expression.POWER():
            return _binary_exp(expression, operator.pow)
    elif isinstance(expression, QuilParser.MulDivExpContext):
        if expression.TIMES():
            return _binary_exp(expression, operator.mul)
        elif expression.DIVIDE():
            return _binary_exp(expression, operator.truediv)
    elif isinstance(expression, QuilParser.AddSubExpContext):
        if expression.PLUS():
            return _binary_exp(expression, operator.add)
        elif expression.MINUS():
            return _binary_exp(expression, operator.sub)
    elif isinstance(expression, QuilParser.SignedExpContext):
        if expression.sign().PLUS():
            return _expression(expression.expression())
        elif expression.sign().MINUS():
            return -1 * _expression(expression.expression())
    elif isinstance(expression, QuilParser.FunctionExpContext):
        return _apply_function(expression.function(), _expression(expression.expression()))
    elif isinstance(expression, QuilParser.SegmentExpContext):
        return _segment(expression.segment())
    elif isinstance(expression, QuilParser.NumberExpContext):
        return _number(expression.number())
    elif isinstance(expression, QuilParser.VariableExpContext):
        return _variable(expression.variable())

    raise RuntimeError("Unexpected expression type:" + expression.getText())


def _binary_exp(expression, op):
    # type: (QuilParser.ExpressionContext, Callable) -> Number
    """
    Apply an operator to two expressions. Start by evaluating both sides of the operator.
    """
    [arg1, arg2] = expression.expression()
    return op(_expression(arg1), _expression(arg2))


def _apply_function(func, arg):
    # type: (QuilParser.FunctionContext, Any) -> Any
    if isinstance(arg, Expression):
        if func.SIN():
            return parameters.quil_sin(arg)
        elif func.COS():
            return parameters.quil_cos(arg)
        elif func.SQRT():
            return parameters.quil_sqrt(arg)
        elif func.EXP():
            return parameters.quil_exp(arg)
        elif func.CIS():
            return parameters.quil_cis(arg)
        else:
            raise RuntimeError("Unexpected function to apply: " + func.getText())
    else:
        if func.SIN():
            return sin(arg)
        elif func.COS():
            return cos(arg)
        elif func.SQRT():
            return sqrt(arg)
        elif func.EXP():
            return exp(arg)
        elif func.CIS():
            return cos(arg) + complex(0, 1) * sin(arg)
        else:
            raise RuntimeError("Unexpected function to apply: " + func.getText())


def _number(number):
    # type: (QuilParser.NumberContext) -> Any
    if number.realN():
        return _real(number.realN())
    elif number.imaginaryN():
        return complex(0, _real(number.imaginaryN().realN()))
    elif number.I():
        return complex(0, 1)
    elif number.PI():
        return np.pi
    else:
        raise RuntimeError("Unexpected number: " + number.getText())


def _real(real):
    # type: (QuilParser.RealNContext) -> Any
    if real.FLOAT():
        return float(real.getText())
    elif real.INT():
        return int(real.getText())
    else:
        raise RuntimeError("Unexpected real: " + real.getText())
