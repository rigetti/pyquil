Changelog
=========

v1.9
----

 - :py:func:`Program.alloc()` has been deprecated. Please instantiate :py:class:`QubitPlaceholder`
   directly.
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