Gates and Instructions
======================

A :py:class:`~pyquil.quil.Program` is effectively a list of gates and instructions which
can be created using the function documented in this section::

    >>> p = Program()
    >>> p += H(qubit=0)
    >>> p += RY(angle=pi/3, qubit=1)
    >>> p += CNOT(0, 1)
        ...


.. currentmodule:: pyquil.gates

Native gates for Rigetti QPUs
-----------------------------

Physical quantum processors can enact a subset of all named gates. Luckily,
a small set of gates is universal for quantum computation, so all named gates
can be enacted by suitable combinations of physically realizable gates. Rigetti's
superconducting quantum processors can perform :py:func:`RX` with ``angle=+-pi/2`` or
``angle=+-pi``, :py:func:`RZ` with an arbitrary angle, and :py:func:`CZ` interactions
between neighboring qubits. Rigetti QPUs can natively measure in the computational (Z) basis.

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    RX
    RZ
    CZ
    MEASURE


All gates and instructions
--------------------------

In general, you will write a quantum program using the full suite of Quil gates and instructions
and use :ref:`the Quil compiler <compiler>` to compile programs into the target instruction
set (ISA). The full list of quantum gates and classical Quil instructions is enumerated here.

.. rubric:: Single-qubit gates
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    I
    X
    Y
    Z
    H
    S
    T
    RX
    RY
    RZ
    PHASE

.. rubric:: Multi-qubit gates
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    CZ
    CNOT
    CCNOT
    CPHASE00
    CPHASE01
    CPHASE10
    CPHASE
    SWAP
    CSWAP
    ISWAP
    PSWAP


.. rubric:: Classical instructions
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    WAIT
    RESET
    NOP
    HALT
    MEASURE
    NEG
    NOT
    AND
    OR
    IOR
    XOR
    MOVE
    EXCHANGE
    LOAD
    STORE
    CONVERT
    ADD
    SUB
    MUL
    DIV
    EQ
    LT
    LE
    GT
    GE

.. rubric:: Collections
.. autosummary::

    QUANTUM_GATES
    STANDARD_INSTRUCTIONS

