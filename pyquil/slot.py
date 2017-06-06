#!/usr/bin/python
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
"""
Contains Slot pyQuil placeholders for constructing Quil template programs.
"""

class Slot(object):
    """
    A placeholder for a parameter value.

    Arithmetic operations: ``+-*/``
    Logical: abs, max, <, >, <=, >=, !=, ==
    Arbitrary functions are not supported

    :param float value: A value to initialize to. Defaults to 0.0
    :param function func: An initial function to determine the final parameterized value.
    """

    def __init__(self, value=0.0, func=None):
        self._value = value
        self.compute_value = func if func is not None else lambda: self._value

    def value(self):
        """
        Computes the value of this Slot parameter.
        """
        return self.compute_value()

    def __repr__(self):
        return "<Slot {}>".format(self.value())

    def __str__(self):
        return str(self.value())

    def __add__(self, val):
        return Slot(self, lambda: self.value() + val)

    def __radd__(self, val):
        return Slot(self, lambda: val + self.value())

    def __sub__(self, val):
        return Slot(self, lambda: self.value() - val)

    def __rsub__(self, val):
        return Slot(self, lambda: val - self.value())

    def __mul__(self, val):
        return Slot(self, lambda: self.value() * val)

    def __rmul__(self, val):
        return Slot(self, lambda: val * self.value())

    def __div__(self, val):
        return Slot(self, lambda: self.value() / val)

    __truediv__ = __div__

    def __rdiv__(self, val):
        return Slot(self, lambda: val / self.value())

    __rtruediv__ = __rdiv__

    def __neg__(self):
        return Slot(self, lambda: -self.value())

    def __abs__(self):
        return Slot(self, lambda: abs(self.value()))

    def __max__(self, other):
        return max(other, self.value())

    def __lt__(self, other):
        return self.value() < other

    def __le__(self, other):
        return self.value() <= other

    def __eq__(self, other):
        return self.value() == other

    def __ne__(self, other):
        return self.value() != other

    def __gt__(self, other):
        return self.value() > other

    def __ge__(self, other):
        return self.value() >= other