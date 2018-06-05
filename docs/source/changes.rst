Changelog
=========

v1.9
----

 - :py:class:`Program` can be initiated with a generator expression.
 - :py:func:`Program.measure_all` (with no arguments) will measure all qubits in a program.
 - ``classical_addresses`` is now optional in QVM and QPU :py:func:`run` methods. By default,
   any classical addresses targeted by ``MEASURE`` will be returned.
 - :py:func:`QVMConnection.pauli_expectation` accepts ``PauliSum`` as arguments. This offers
   a more sensible API compared to :py:func:`QVMConnection.expectation`.
 - pyQuil will now retry jobs every 10 seconds if the QPU is re-tuning.
 - :py:class:`PauliTerm` now remembers the order of its operations. ``sX(1)*sZ(2)`` will compile
   to different Quil code than ``sZ(2)*sX(1)``, although the terms will still be equal according
   to the ``__eq__`` method. During :py:class:`PauliSum` combination
   of like terms, a warning will be emitted if two terms are combined that have different orders
   of operation.
 - :py:func:`PauliTerm.id()` takes an optional argument ``sort_ops`` which defaults to True for
   backwards compatibility. However, this function should not be used for comparing term-type like
   it has been used previously. Use :py:func:`PauliTerm.operations_as_set()` instead. In the future,
   ``sort_ops`` will default to False and will eventually be removed.
 - :py:func:`Program.alloc()` has been deprecated. Please instantiate :py:class:`QubitPlaceholder()`
   directly or request a "register" (list) of ``n`` placeholders by using the class constructor
   :py:func:`QubitPlaceholder.register(n)`.
 - Programs must contain either (1) all instantiated qubits with integer indexes or (2) all
   placeholder qubits of type :py:class:`QubitPlaceholder`. We have found that most users use
   (1) but (2) will become useful with larger and more diverse devices.
 - Programs that contain qubit placeholders must be **explicitly addressed** prior to execution.
   Previously, qubits would be assigned "under the hood" to integers 0...N. Now, you must use
   :py:func:`address_qubits` which returns a new program with all qubits indexed depending
   on the ``qubit_mapping`` argument. The original program is unaffected and can be "readdressed"
   multiple times.
 - In light of the above, ``shift_quantum_gates`` has been removed. Users who relied on this
   functionality should use :py:class:`QubitPlaceholder` and :py:func:`address_qubits` to
   achieve the same result. Users should also double-check data resulting from use of this function
   as there were several edge cases which would cause the shift to be applied incorrectly resulting
   in badly-addressed qubits.
 - :py:class:`PauliTerm` can now accept :py:class:`QubitPlaceholder` in addition to integers.
 - :py:class:`QubitPlaceholder` is no longer a subclass of :py:class:`Qubit`.
   :py:class:`LabelPlaceholder` is no longer a subclass of :py:class:`Label`.
 - :py:class:`QuilAtom` subclasses' hash functions have changed.
 - Documentation improvements
