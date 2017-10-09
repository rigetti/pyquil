import operator

from antlr4 import *
from numpy.ma import sin, cos, sqrt, exp

from pyquil.gates import STANDARD_GATES
from .gen3.QuilLexer import QuilLexer
from .gen3.QuilListener import QuilListener
from .gen3.QuilParser import QuilParser
from pyquil.quil import Program
from pyquil.quilbase import Gate, DefGate, Measurement, Addr, JumpTarget, Label, Halt, Jump, JumpWhen, JumpUnless, \
    Reset, Wait, ClassicalTrue, ClassicalFalse, ClassicalNot, ClassicalAnd, ClassicalOr, ClassicalMove, \
    ClassicalExchange, Nop, RawInstr
from pyquil.resource_manager import DirectQubit


def run_parser(quil: str):
    input_stream = InputStream(quil)
    lexer = QuilLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = QuilParser(stream)
    tree = parser.quil()

    pyquil_listener = PyQuilListener()
    walker = ParseTreeWalker()
    walker.walk(pyquil_listener, tree)

    return pyquil_listener.result


class PyQuilListener(QuilListener):
    def __init__(self):
        self.result = []

    def exitDefGate(self, ctx: QuilParser.DefGateContext):
        gate_name = ctx.name().getText()
        if ctx.variable():
            raise NotImplementedError("%variables are not supported yet")
        matrix = _matrix(ctx.matrix())
        self.result.append(DefGate(gate_name, matrix))

    def exitDefCircuit(self, ctx: QuilParser.DefCircuitContext):
        raise NotImplementedError("circuits are not supported yet")

    def exitGate(self, ctx: QuilParser.GateContext):
        gate_name = ctx.name().getText()
        params = list(map(_param, ctx.param()))

        if gate_name in STANDARD_GATES:
            qubits = list(map(_qubit, ctx.qubit()))
            if params:
                self.result.append(STANDARD_GATES[gate_name](*params)(*qubits))
            else:
                self.result.append(STANDARD_GATES[gate_name](*qubits))
        else:
            qubits = list(map(_direct_qubit, ctx.qubit()))
            self.result.append(Gate(gate_name, params, qubits))

    def exitMeasure(self, ctx: QuilParser.MeasureContext):
        qubit = _direct_qubit(ctx.qubit())
        classical = None
        if ctx.addr():
            classical = _addr(ctx.addr())
        self.result.append(Measurement(qubit, classical))

    def exitDefLabel(self, ctx: QuilParser.DefLabelContext):
        self.result.append(JumpTarget(_label(ctx.label())))

    def exitHalt(self, ctx: QuilParser.HaltContext):
        self.result.append(Halt())

    def exitJump(self, ctx: QuilParser.JumpContext):
        self.result.append(Jump(_label(ctx.label())))

    def exitJumpWhen(self, ctx: QuilParser.JumpWhenContext):
        self.result.append(JumpWhen(_label(ctx.label()), _addr(ctx.addr())))

    def exitJumpUnless(self, ctx: QuilParser.JumpUnlessContext):
        self.result.append(JumpUnless(_label(ctx.label()), _addr(ctx.addr())))

    def exitResetState(self, ctx: QuilParser.ResetStateContext):
        self.result.append(Reset())

    def exitWait(self, ctx: QuilParser.WaitContext):
        self.result.append(Wait())

    def exitClassicalUnary(self, ctx: QuilParser.ClassicalUnaryContext):
        if ctx.TRUE():
            self.result.append(ClassicalTrue(_addr(ctx.addr())))
        elif ctx.FALSE():
            self.result.append(ClassicalFalse(_addr(ctx.addr())))
        elif ctx.NOT():
            self.result.append(ClassicalNot(_addr(ctx.addr())))

    def exitClassicalBinary(self, ctx: QuilParser.ClassicalBinaryContext):
        if ctx.AND():
            self.result.append(ClassicalAnd(_addr(ctx.addr(0)), _addr(ctx.addr(1))))
        elif ctx.OR():
            self.result.append(ClassicalOr(_addr(ctx.addr(0)), _addr(ctx.addr(1))))
        elif ctx.MOVE():
            self.result.append(ClassicalMove(_addr(ctx.addr(0)), _addr(ctx.addr(1))))
        elif ctx.EXCHANGE():
            self.result.append(ClassicalExchange(_addr(ctx.addr(0)), _addr(ctx.addr(1))))

    def exitNop(self, ctx: QuilParser.NopContext):
        self.result.append(Nop())

    def exitInclude(self, ctx: QuilParser.IncludeContext):
        self.result.append(RawInstr(ctx.INCLUDE().getText() + ' ' + ctx.STRING().getText()))

    def exitPragma(self, ctx: QuilParser.PragmaContext):
        pragma_names = ' '.join(map(lambda x: x.getText(), ctx.IDENTIFIER()))
        self.result.append(RawInstr(ctx.PRAGMA().getText() + ' ' + pragma_names + ' ' + ctx.STRING().getText()))


def _qubit(qubit: QuilParser.QubitContext):
    return int(qubit.getText())


def _direct_qubit(qubit: QuilParser.QubitContext):
    return DirectQubit(int(qubit.getText()))


def _param(param: QuilParser.ParamContext):
    if param.dynamicParam():
        raise NotImplementedError("dynamic parameters not supported yet")
    elif param.expression():
        return _expression(param.expression())
    else:
        raise RuntimeError("Unexpected param: " + str(param))


def _matrix(matrix: QuilParser.MatrixContext):
    out = []
    for row in matrix.matrixRow():
        out.append(list(map(_expression, row.expression())))
    return out


def _addr(classical: QuilParser.AddrContext):
    return Addr(int(classical.classicalBit().getText()))


def _label(label: QuilParser.LabelContext):
    return Label(label.IDENTIFIER().getText())


def _expression(expression):
    if type(expression) is QuilParser.ParenthesisExpContext:
        return _expression(expression.expression())
    elif type(expression) is QuilParser.PowerExpContext:
        if expression.POWER():
            return _binary_exp(expression, operator.pow)
    elif type(expression) is QuilParser.MulDivExpContext:
        if expression.TIMES():
            return _binary_exp(expression, operator.mul)
        elif expression.DIVIDE():
            return _binary_exp(expression, operator.truediv)
    elif type(expression) is QuilParser.AddSubExpContext:
        if expression.PLUS():
            return _binary_exp(expression, operator.add)
        elif expression.MINUS():
            return _binary_exp(expression, operator.sub)
    elif type(expression) is QuilParser.FunctionExpContext:
        return _apply_function(expression.function(), _expression(expression.expression()))
    elif type(expression) is QuilParser.NumberExpContext:
        return _number(expression.number())
    elif type(expression) is QuilParser.VariableContext:
        raise NotImplementedError("%variables are not supported yet")

    raise RuntimeError("Unexpected expression type: " + str(expression))


def _binary_exp(expression, op):
    [arg1, arg2] = expression.expression()
    return op(_expression(arg1), _expression(arg2))


def _apply_function(func: QuilParser.FunctionContext, arg):
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
        raise RuntimeError("Unexpected function to apply: " + str(func))


def _number(number: QuilParser.NumberContext):
    if number.realN():
        return _real(number.realN())
    elif number.imaginaryN():
        return complex(0, _real(number.imaginaryN().realN()))
    elif number.I():
        return complex(0, 1)
    else:
        raise RuntimeError("Unexpected number: " + str(number))


def _real(real: QuilParser.RealNContext):
    if real.floatN():
        return float(real.getText())
    elif real.intN():
        return int(real.getText())
    else:
        raise RuntimeError("Unexpected real: " + str(real))
