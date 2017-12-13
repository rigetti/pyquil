from fractions import Fraction

import numpy as np
from six import integer_types

from pyquil.quilatom import QuilAtom
from pyquil.slot import Slot

__all__ = ['Parameter', 'sin', 'cos', 'sqrt', 'exp', 'cis']


def format_parameter(element):
    """
    Formats a particular parameter. Essentially the same as built-in formatting except using 'i' instead of 'j' for
    the imaginary number.

    :param element: {int, float, long, complex, Parameter} Formats a parameter for Quil output.
    """
    if isinstance(element, integer_types) or isinstance(element, np.int_):
        return repr(element)
    elif isinstance(element, float):
        return _check_for_pi(element)
    elif isinstance(element, complex):
        out = ''
        r = element.real
        i = element.imag
        if i == 0:
            return repr(r)

        if r != 0:
            out += repr(r)

        if i == 1:
            out += 'i'
        elif i == -1:
            out += '-i'
        elif i < 0:
            out += repr(i) + 'i'
        else:
            out += '+' + repr(i) + 'i'

        return out
    elif isinstance(element, Expression):
        return _expression_to_string(element)
    elif isinstance(element, Slot):
        return format_parameter(element.value())
    assert False, "Invalid parameter: %r" % element


class Expression(object):
    """
    Expression involving some unbound parameters. Parameters in Quil are represented as a label like '%x' for the
    parameter named 'x'. An example expression therefore may be '%x*(%y/4)'.

    Expressions may also have function calls, supported functions in Quil are sin, cos, sqrt, exp, and cis

    This class overrides all the Python operators that are supported by Quil.
    """
    def __str__(self):
        return _expression_to_string(self)

    def __repr__(self):
        return str(self.__class__.__name__) + '(' + ','.join(map(repr, self.__dict__.values())) + ')'

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __div__(self, other):
        return Div(self, other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        return Div(other, self)

    __rtruediv__ = __rdiv__

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __neg__(self):
        return Mul(-1, self)


class Parameter(QuilAtom, Expression):
    """
    Parameters in Quil are represented as a label like '%x' for the parameter named 'x'.
    """
    def __init__(self, name):
        self.name = name

    def out(self):
        return '%' + self.name


class Function(Expression):
    """
    Supported functions in Quil are sin, cos, sqrt, exp, and cis
    """
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression


def sin(expression):
    return Function('sin', expression)


def cos(expression):
    return Function('cos', expression)


def sqrt(expression):
    return Function('sqrt', expression)


def exp(expression):
    return Function('exp', expression)


def cis(expression):
    return Function('cis', expression)


class BinaryExp(Expression):
    operator = None
    precedence = None
    associates = None

    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2


class Add(BinaryExp):
    operator = '+'
    precedence = 1
    associates = 'both'

    def __init__(self, op1, op2):
        super(Add, self).__init__(op1, op2)


class Sub(BinaryExp):
    operator = '-'
    precedence = 1
    associates = 'left'

    def __init__(self, op1, op2):
        super(Sub, self).__init__(op1, op2)


class Mul(BinaryExp):
    operator = '*'
    precedence = 2
    associates = 'both'

    def __init__(self, op1, op2):
        super(Mul, self).__init__(op1, op2)


class Div(BinaryExp):
    operator = '/'
    precedence = 2
    associates = 'left'

    def __init__(self, op1, op2):
        super(Div, self).__init__(op1, op2)


class Pow(BinaryExp):
    operator = '^'
    precedence = 3
    associates = 'right'

    def __init__(self, op1, op2):
        super(Pow, self).__init__(op1, op2)


def _expression_to_string(expression):
    """
    Recursively converts an expression to a string taking into account precedence and associativity for placing
    parenthesis

    :param Expression expression: expression involving parameters
    :return: string such as '%x*(%y-4)'
    :rtype: str
    """
    if isinstance(expression, BinaryExp):
        left = _expression_to_string(expression.op1)
        if isinstance(expression.op1, BinaryExp) and not (
                expression.op1.precedence > expression.precedence or
                expression.op1.precedence == expression.precedence and
                expression.associates in ('left', 'both')):
            left = '(' + left + ')'

        right = _expression_to_string(expression.op2)
        if isinstance(expression.op2, BinaryExp) and not (
                expression.precedence < expression.op2.precedence or
                expression.precedence == expression.op2.precedence and
                expression.associates in ('right', 'both')):
            right = '(' + right + ')'

        return left + expression.operator + right
    elif isinstance(expression, Function):
        return expression.name + '(' + _expression_to_string(expression.expression) + ')'
    elif isinstance(expression, Parameter):
        return str(expression)
    else:
        return format_parameter(expression)


def _contained_parameters(expression):
    """
    Determine which parameters are contained in this expression.

    :param Expression expression: expression involving parameters
    :return: set of parameters contained in this expression
    :rtype: set
    """
    if isinstance(expression, BinaryExp):
        return _contained_parameters(expression.op1) | _contained_parameters(expression.op2)
    elif isinstance(expression, Function):
        return _contained_parameters(expression.expression)
    elif isinstance(expression, Parameter):
        return {expression}
    else:
        return set()


def _check_for_pi(element):
    """
    Check to see if there exists a rational number r = p/q
    in reduced form for which the difference between element/np.pi
    and r is small and q <= 8.

    :param element: float
    :return element: pretty print string if true, else standard representation.
    """
    frac = Fraction(element / np.pi).limit_denominator(8)
    num, den = frac.numerator, frac.denominator
    sign = "-" if num < 0 else ""
    if num / float(den) == element / np.pi:
        if num == 0:
            return "0"
        elif abs(num) == 1 and den == 1:
            return sign + "pi"
        elif abs(num) == 1:
            return sign + "pi/" + repr(den)
        elif den == 1:
            return repr(num) + "*pi"
        else:
            return repr(num) + "*pi/" + repr(den)
    else:
        return repr(element)
