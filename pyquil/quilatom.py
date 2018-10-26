##############################################################################
# Copyright 2017-2018 Rigetti Computing
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
import numpy as np
from six import integer_types
from warnings import warn
from fractions import Fraction


class QuilAtom(object):
    """
    Abstract class for atomic elements of Quil.
    """

    def out(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        raise NotImplementedError()


class Qubit(QuilAtom):
    """
    Representation of a qubit.

    :param int index: Index of the qubit.
    """

    def __init__(self, index):
        if not (isinstance(index, integer_types) and index >= 0):
            raise TypeError("Addr index must be a non-negative int")
        self.index = index

    def out(self):
        return str(self.index)

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        return "<Qubit {0}>".format(self.index)

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return isinstance(other, Qubit) and other.index == self.index


class QubitPlaceholder(QuilAtom):
    def out(self):
        raise RuntimeError("Qubit {} has not been assigned an index".format(self))

    def __str__(self):
        return "q{}".format(id(self))

    def __repr__(self):
        return "<QubitPlaceholder {}>".format(id(self))

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return isinstance(other, QubitPlaceholder) and id(other) == id(self)

    @classmethod
    def register(cls, n):
        """Return a 'register' of ``n`` QubitPlaceholders.

        >>> qs = QubitPlaceholder.register(8) # a qubyte
        >>> prog = Program(H(q) for q in qs)
        >>> address_qubits(prog).out()
        H 0
        H 1
        ...
        >>>

        The returned register is a Python list of QubitPlaceholder objects, so all
        normal list semantics apply.

        :param n: The number of qubits in the register
        """
        return [cls() for _ in range(n)]


def unpack_qubit(qubit):
    """
    Get a qubit from an object.

    :param qubit: An int or Qubit.
    :return: A Qubit instance
    """
    if isinstance(qubit, integer_types):
        return Qubit(qubit)
    elif isinstance(qubit, Qubit):
        return qubit
    elif isinstance(qubit, QubitPlaceholder):
        return qubit
    else:
        raise TypeError("qubit should be an int or Qubit instance")


def unpack_classical_reg(c):
    """
    Get the address for a classical register.

    :param c: A list of length 2, a pair, a string (to be interpreted as name[0]), or a MemoryReference.
    :return: The address as a MemoryReference.
    """
    if isinstance(c, list) or isinstance(c, tuple):
        if len(c) > 2 or len(c) is 0:
            raise ValueError("if c is a list/tuple, it should be of length <= 2")
        if len(c) is 1:
            c = (c[0], 0)
        if not isinstance(c[0], str):
            raise ValueError("if c is a list/tuple, its first member should be a string")
        if not isinstance(c[1], int):
            raise ValueError("if c is a list/tuple, its second member should be an int")
        return MemoryReference(c[0], c[1])
    if isinstance(c, MemoryReference):
        return c
    elif isinstance(c, str):
        return MemoryReference(c, 0)
    else:
        raise TypeError("c should be a list of length 2, a pair, a string, or a MemoryReference")


class Label(QuilAtom):
    """
    Representation of a label.

    :param string label_name: The label name.
    """

    def __init__(self, label_name):
        self.name = label_name

    def out(self):
        return "@{name}".format(name=self.name)

    def __str__(self):
        return "@{name}".format(name=self.name)

    def __repr__(self):
        return "<Label {0}>".format(repr(self.name))

    def __eq__(self, other):
        return isinstance(other, Label) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class LabelPlaceholder(QuilAtom):
    def __init__(self, prefix="L"):
        self.prefix = prefix

    def out(self):
        raise RuntimeError("Label has not been assigned a name")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "<LabelPlaceholder {} {}>".format(self.prefix, id(self))

    def __eq__(self, other):
        return isinstance(other, LabelPlaceholder) and id(other) == id(self)

    def __hash__(self):
        return hash(id(self))


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
            assert np.isclose(r, 0, atol=1e-14)
            out = 'i'
        elif i == -1:
            assert np.isclose(r, 0, atol=1e-14)
            out = '-i'
        elif i < 0:
            out += repr(i) + 'i'
        elif r != 0:
            out += '+' + repr(i) + 'i'
        else:
            out += repr(i) + 'i'

        return out
    elif isinstance(element, MemoryReference):
        return str(element)
    elif isinstance(element, Expression):
        return _expression_to_string(element)
    elif isinstance(element, MemoryReference):
        return element.out()
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

    def _substitute(self, d):
        return self


def substitute(expr, d):
    """
    Using a dictionary of substitutions ``d`` try and explicitly evaluate as much of ``expr`` as
    possible.

    :param Expression expr: The expression whose parameters are substituted.
    :param Dict[Parameter,Union[int,float]] d: Numerical substitutions for parameters.
    :return: A partially simplified Expression or a number.
    :rtype: Union[Expression,int,float]
    """
    try:
        return expr._substitute(d)
    except AttributeError:
        return expr


def substitute_array(a, d):
    """
    Apply ``substitute`` to all elements of an array ``a`` and return the resulting array.

    :param Union[np.array,List] a: The expression array to substitute.
    :param Dict[Parameter,Union[int,float]] d: Numerical substitutions for parameters.
    :return: An array of partially substituted Expressions or numbers.
    :rtype: np.array
    """
    a = np.asarray(a, order="C")
    return np.array([substitute(v, d) for v in a.flat]).reshape(a.shape)


class Parameter(QuilAtom, Expression):
    """
    Parameters in Quil are represented as a label like '%x' for the parameter named 'x'.
    """

    def __init__(self, name):
        self.name = name

    def out(self):
        return '%' + self.name

    def _substitute(self, d):
        return d.get(self, self)

    def __str__(self):
        return '%' + self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Parameter) and other.name == self.name


class Function(Expression):
    """
    Supported functions in Quil are sin, cos, sqrt, exp, and cis
    """
    def __init__(self, name, expression, fn):
        self.name = name
        self.expression = expression
        self.fn = fn

    def _substitute(self, d):
        sop = substitute(self.expression, d)
        if isinstance(sop, Expression):
            return Function(self.name, sop, self.fn)
        return self.fn(sop)

    def __eq__(self, other):
        return (isinstance(other, Function)
                and self.name == other.name
                and self.expression == other.expression)

    def __neq__(self, other):
        return not self.__eq__(other)


def quil_sin(expression):
    return Function('sin', expression, np.sin)


def quil_cos(expression):
    return Function('cos', expression, np.cos)


def quil_sqrt(expression):
    return Function('sqrt', expression, np.sqrt)


def quil_exp(expression):
    return Function('exp', expression, np.exp)


def quil_cis(expression):
    return Function('cis', expression, lambda x: np.exp(1j * x))


class BinaryExp(Expression):
    operator = None     # type: str
    precedence = None   # type: int
    associates = None   # type: str

    @staticmethod
    def fn(a, b):
        raise NotImplementedError

    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def _substitute(self, d):
        sop1, sop2 = substitute(self.op1, d), substitute(self.op2, d)
        return self.fn(sop1, sop2)

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.op1 == other.op1
                and self.op2 == other.op2)

    def __neq__(self, other):
        return not self.__eq__(other)


class Add(BinaryExp):
    operator = '+'
    precedence = 1
    associates = 'both'

    @staticmethod
    def fn(a, b):
        return a + b

    def __init__(self, op1, op2):
        super(Add, self).__init__(op1, op2)


class Sub(BinaryExp):
    operator = '-'
    precedence = 1
    associates = 'left'

    @staticmethod
    def fn(a, b):
        return a - b

    def __init__(self, op1, op2):
        super(Sub, self).__init__(op1, op2)


class Mul(BinaryExp):
    operator = '*'
    precedence = 2
    associates = 'both'

    @staticmethod
    def fn(a, b):
        return a * b

    def __init__(self, op1, op2):
        super(Mul, self).__init__(op1, op2)


class Div(BinaryExp):
    operator = '/'
    precedence = 2
    associates = 'left'

    @staticmethod
    def fn(a, b):
        return a / b

    def __init__(self, op1, op2):
        super(Div, self).__init__(op1, op2)


class Pow(BinaryExp):
    operator = '^'
    precedence = 3
    associates = 'right'

    @staticmethod
    def fn(a, b):
        return a ** b

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
                expression.op1.precedence > expression.precedence
                or expression.op1.precedence == expression.precedence
                and expression.associates in ('left', 'both')):
            left = '(' + left + ')'

        right = _expression_to_string(expression.op2)
        if isinstance(expression.op2, BinaryExp) and not (
                expression.precedence < expression.op2.precedence
                or expression.precedence == expression.op2.precedence
                and expression.associates in ('right', 'both')):
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


class MemoryReference(QuilAtom, Expression):
    """
    Representation of a reference to a classical memory address.

    :param name: The name of the variable
    :param offset: Everything in Quil is a C-style array, so every memory reference has an offset.
    :param declared_size: The optional size of the named declaration. This can be used for bounds
        checking, but isn't. It is used for pretty-printing to quil by deciding whether to output
        memory references with offset 0 as either e.g. ``ro[0]`` or ``beta`` depending on whether
        the declared variable is of length >1 or 1, resp.
    """

    def __init__(self, name, offset=0, declared_size=None):
        if not isinstance(offset, integer_types) or offset < 0:
            raise TypeError("MemoryReference offset must be a non-negative int")
        self.name = name
        self.offset = offset
        self.declared_size = declared_size

    def out(self):
        if self.declared_size is not None and self.declared_size == 1 and self.offset == 0:
            return "{}".format(self.name)
        else:
            return "{}[{}]".format(self.name, self.offset)

    def __str__(self):
        if self.declared_size is not None and self.declared_size == 1 and self.offset == 0:
            return "{}".format(self.name)
        else:
            return "{}[{}]".format(self.name, self.offset)

    def __repr__(self):
        return "<MRef {}[{}]>".format(self.name, self.offset)

    def __eq__(self, other):
        return (isinstance(other, MemoryReference)
                and other.name == self.name
                and other.offset == self.offset)

    def __hash__(self):
        return hash((self.name, self.offset))

    def __getitem__(self, offset):
        if self.offset != 0:
            raise ValueError("Please only index off of the base MemoryReference (offset = 0)")

        return MemoryReference(name=self.name, offset=offset)


class Addr(MemoryReference):
    """
    Representation of a classical bit address.

    WARNING: Addr has been deprecated. Addr(c) instances are auto-replaced by MemoryReference("ro", c).
             Use MemoryReference instances.

    :param int value: The classical address.
    """

    def __init__(self, value):
        warn("Addr objects have been deprecated. Defaulting to memory region \"ro\". Use MemoryReference instead.")
        if not isinstance(value, integer_types) or value < 0:
            raise TypeError("Addr value must be a non-negative int")
        super(Addr, self).__init__("ro", offset=value, declared_size=None)
