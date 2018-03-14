Changelog
=========

v1.9
----

 - :py:class:`PauliTerm` now remembers the order of its operations. ``sX(1)*sZ(2)`` will compile
   to different Quil code than ``sZ(2)*sX(1)``, although the terms will still be equal according
   to the ``__eq__`` method. :py:func:`PauliTerm.id()` will return a string representation of
   the term's operations in the remembered order. To compare terms irrespective of operation
   order, use :py:func:`PauliTerm.operations_as_set()`. During :py:class:`PauliSum` combination
   of like terms, a warning will be emitted if two terms are combined that have different orders
   of operation.