.. _qvm:

The Quantum Computer
====================

The Quantum Virtual Machine (QVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rigetti Quantum Virtual Machine is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_  It is implemented in ANSI Common LISP and
executes programs specified in the Quantum Instruction Language (Quil).

Quil is an opinionated quantum instruction language: its basic belief is that in the near term quantum computers will
operate as coprocessors, working in concert with traditional CPUs.  This means that Quil is designed to execute on
a Quantum Abstract Machine that has a shared classical/quantum architecture at its core.

The QVM is a wavefunction simulation of unitary evolution with classical control flow
and shared quantum classical memory.

.. _qvm_use:

Using the QVM
-------------
After :ref:`downloading the SDK <sdkinstall>`, the QVM is available on your local machine. You can initialize a local
QVM server by typing ``qvm -S`` into your terminal. You should see the following message.

.. code:: python

    $ qvm -S
    ******************************
    * Welcome to the Rigetti QVM *
    ******************************
    Copyright (c) 2018 Rigetti Computing.

    (Configured with 2048 MiB of workspace and 8 workers.)

    [2018-11-06 18:18:18] Starting server on port 5000.

For a detailed description of how to use the ``qvm`` from the command line, see :ref:`The QVM manual page <qvm_man>`.

Once the QVM is serving requests, we can run the following pyQuil program to get a ``QuantumComputer`` object which
will use the QVM.

.. code:: python

    from pyquil import get_qc, Program
    from pyquil.gates import *
    qc = get_qc('9q-square-qvm')


One executes quantum programs on the QVM using the ``.run(...)`` method, intended to closely mirror how one will
execute programs on a real QPU. We also offer a Wavefunction Simulator (formerly a part of the ``QVM`` object),
which allows users to contruct and inspect wavefunctions of quantum programs. Learn more
about the Wavefunction Simulator :ref:`here <wavefunction_simulator>`. For information on constructing quantum
programs, please refer back to :ref:`basics`.

The ``.run(...)`` method
------------------------

The ``.run(...)`` method takes in a compiled program. You are responsible for compiling your program before running it.
Remember to also start up a ``quilc`` compiler server, too, with ``quilc -S``.

.. code:: python

    p = Program()
    ro = p.declare('ro', 'BIT', 1)
    p += X(0)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    p.wrap_in_numshots_loop(5)
    executable = qc.compile(p)
    results = qvm.run(executable)
    print(results)

The results returned are a *list of lists of integers*. In the above case, that's

.. parsed-literal::

    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]

Let's unpack this. The *outer* list is an enumeration over the trials; the argument given to
``wrap_in_numshots_loop`` will match the length of ``results``.

The *inner* list, on the other hand, is an enumeration over the results stored in the memory region named ``ro``, which
we use as our readout register. We see that the result of this program is that the memory region ``ro[0]`` now stores
the state of qubit 0, which should be ``1`` after an :math:`X`-gate. See :ref:`declaring_memory` and :ref:`measurement`
for more details about declaring and accessing classical memory regions.

.. [1] https://arxiv.org/abs/1608.03355

Simulating the QPU using the QVM
--------------------------------

The QVM is a powerful tool for testing quantum programs before executing them on the QPU.

.. code:: python

    qc = get_qc("QuantumComputerName")
    qc = get_qc("QuantumComputerName-qvm")

By simply providing ``-qvm`` in the device name, all programs executed on this QVM will, have the same topology as
the named QPU. To learn how to add noise models to your virtual ``QuantumComputer`` instance, check out
:ref:`noise`.
