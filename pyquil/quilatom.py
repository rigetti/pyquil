from six import integer_types


class QuilAtom(object):
    """
    Abstract class for atomic elements of Quil.
    """
    def out(self):
        pass

    def __str__(self):
        return self.out()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.out() == other.out()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))


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

    def __repr__(self):
        return "<Qubit {0}>".format(self.index)


class QubitPlaceholder(Qubit):
    def __init__(self):
        pass

    @property
    def index(self):
        raise RuntimeError("Qubit has not been assigned an index")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "<QubitPlaceholder {}>".format(id(self))

    def __lt__(self, other):
        return id(self) < id(other)


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
        return "[{0}]".format(self.address)

    def __repr__(self):
        return "<Addr {0}>".format(self.address)


class Label(QuilAtom):
    """
    Representation of a label.

    :param string label_name: The label name.
    """

    def __init__(self, label_name):
        self.name = label_name

    def out(self):
        return "@" + str(self.name)

    def __repr__(self):
        return "<Label {0}>".format(repr(self.name))


class LabelPlaceholder(Label):
    def __init__(self, prefix="L"):
        self.prefix = prefix

    @property
    def name(self):
        raise RuntimeError("Label has not been assigned a name")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "<LabelPlaceholder {} {}>".format(self.prefix, id(self))
