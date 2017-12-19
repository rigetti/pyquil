
The Rigetti QVM
===============

The Rigetti Quantum Virtual Machine is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_  It is implemented in ANSI Common LISP and
executes programs specified in the Quantum Instruction Language (Quil). Quil is an opinionated
quantum instruction language: its basic belief is that in the near term quantum computers will
operate as coprocessors, working in concert with traditional CPUs.  This means that Quil is
designed to execute on a Quantum Abstract Machine that has a shared classical/quantum architecture
at its core. The QVM is a wavefunction simulation of unitary evolution with classical control flow
and shared quantum classical memory.

Most API keys give access to the QVM with up to 26 qubits. If you would like access to more qubits
or help running larger jobs, then contact us at support@rigetti.com. On request we may also
provide access to a QVM that allows persistent wavefunction memory between different programs as
well as direct access to the wavefunction memory (wrapped as a ``numpy`` array) from python.


Multi-qubit basis enumeration on the QVM
----------------------------------------

The Rigetti QVM enumerates bitstrings such that qubit `0` is the least significant bit (LSB)
and therefore on the right end of a bitstring.
For example for the bitstring ``0101``, qubit `0` has the value `1`, qubit `1` has the value `0`,
qubit `2` has the value `1` and qubit `3` has the value `0`.
This convention is counter to that often found in the quantum computing literature where
bitstrings are often ordered such that the lowest-index qubit is on the left.

As a consequence there arise some subtle but important differences in the ordering of wavefunction
and multi-qubit gate matrix coefficients.
According to our conventions the matrix

.. math::

    \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0
    \end{pmatrix}

corresponds to the Quil instruction ``CNOT(1, 0)`` which is counter to how most other people in the
field order their tensor product factors (or more specifically their kronecker products).
In this convention ``CNOT(0, 1)`` is given by

.. math::

    \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0 \\
        0 & 1 & 0 & 0
    \end{pmatrix}

For additional information why we decided on this basis ordering check out our note
*Someone shouts, "|01000>!" Who is Excited?* [2]_.

.. [1] https://arxiv.org/abs/1608.03355
.. [2] https://arxiv.org/abs/1711.02086