from six import integer_types


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
        return repr(self)

    def __repr__(self):
        return "<QubitPlaceholder {}>".format(id(self))

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return isinstance(other, QubitPlaceholder) and id(other) == id(self)

    @classmethod
    def register(cls, n):
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

    :param c: A list of length 1 or an int or an Addr.
    :return: The address as an Addr.
    """
    if not (isinstance(c, integer_types) or isinstance(c, (list, Addr))):
        raise TypeError("c should be an int or list or Addr")
    if isinstance(c, list) and (len(c) != 1 or not isinstance(c[0], int)):
        raise ValueError("if c is a list, it should be of 1 int")
    if isinstance(c, Addr):
        return c
    elif isinstance(c, list):
        return Addr(c[0])
    else:
        return Addr(c)


class Addr(QuilAtom):
    """
    Representation of a classical bit address.

    :param int value: The classical address.
    """

    def __init__(self, value):
        if not isinstance(value, integer_types) or value < 0:
            raise TypeError("Addr value must be a non-negative int")
        self.address = value

    def out(self):
        return "[{}]".format(self.address)

    def __str__(self):
        return "[{}]".format(self.address)

    def __repr__(self):
        return "<Addr {0}>".format(self.address)

    def __eq__(self, other):
        return isinstance(other, Addr) and other.address == self.address

    def __hash__(self):
        return hash(self.address)


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
