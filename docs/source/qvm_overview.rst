
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
or help running larger jobs, then contact us at support@rigetti.com.

.. [1] https://arxiv.org/abs/1608.03355
