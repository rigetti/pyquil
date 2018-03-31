Changelog
=========

v1.9
----

 - :py:class:`PauliTerm` now remembers the order of its operations. ``sX(1)*sZ(2)`` will compile
   to different Quil code than ``sZ(2)*sX(1)``, although the terms will still be equal according
   to the ``__eq__`` method. During :py:class:`PauliSum` combination
   of like terms, a warning will be emitted if two terms are combined that have different orders
   of operation.
 - :py:func:`PauliTerm.id()` takes an optional argument ``sort_ops`` which defaults to True for
   backwards compatibility. However, this function should not be used for comparing term-type like
   it has been used previously. Use :py:func:`PauliTerm.operations_as_set()` instead. In the future,
   ``sort_ops`` will default to False and will eventually be removed.
