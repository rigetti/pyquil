Pauli Operators
===============

Quantum operators can be expressed as combinations of Pauli operators I, X, Y, Z::

    >>> operator = sZ(0)*sZ(1) + sX(2)*sY(3)
    >>> print(operator)
    (1+0j)*Z0*Z1 + (1+0j)*X2*Y3

.. currentmodule:: pyquil.paulis

Construction functions
----------------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    sX
    sY
    sZ
    sI
    ID
    ZERO

Working with operators
----------------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    simplify_pauli_sum
    check_commutation
    commuting_sets
    is_identity
    is_zero
    exponentiate
    exponential_map
    exponentiate_commuting_pauli_sum
    suzuki_trotter
    trotterize


Classes
-------

.. autoclass:: pyquil.paulis.PauliSum

    .. rubric:: Methods
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~PauliSum.get_qubits
        ~PauliSum.simplify
        ~PauliSum.get_programs


.. autoclass:: pyquil.paulis.PauliTerm

    .. rubric:: Methods
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~PauliTerm.id
        ~PauliTerm.operations_as_set
        ~PauliTerm.copy
        ~PauliTerm.program
        ~PauliTerm.from_list
        ~PauliTerm.pauli_string
