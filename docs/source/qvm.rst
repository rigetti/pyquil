.. _qvm:

The Quantum Computer
====================

The Quantum Virtual Machine (QVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rigetti Quantum Virtual Machine is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_  It is implemented in ANSI Common LISP and
executes programs specified in the Quantum Instruction Language (Quil). Quil is an opinionated
quantum instruction language: its basic belief is that in the near term quantum computers will
operate as coprocessors, working in concert with traditional CPUs.  This means that Quil is
designed to execute on a Quantum Abstract Machine that has a shared classical/quantum architecture
at its core. The QVM is a wavefunction simulation of unitary evolution with classical control flow
and shared quantum classical memory.

.. _qvm_use:

Using the QVM
-------------
After `downloading the SDK <rigetti.com/forest>`_,the QVM is available on your local machine. You can initialize a local
QVM instance by doing the following:


.. code:: python

    ### CONSOLE 1
    $ qvm -S
    Configured with 2048 MiB of workspace and 8 workers.)
    [2018-09-20 15:39:50] Starting server on port 5000.


.. code:: python

    from pyquil import get_qc, Program
    from pyquil.gates import *
    qvm = get_qc('9q-square-qvm')


One executes quantum programs on the QVM using a ``.run(...)`` method, intended to closely mirror how one will execute programs on a
real QPU (check out `our website to see current and legacy QPUs <rigetti.com/qpu>`_). We also offer a Wavefunction Simulator
(formerly a part of the QVM object), which allows users to contruct and inspect wavefunctions of quantum programs. Learn more
about :ref:`wavefunction_simulator`.

(For information on constructing quantum programs, please refer back to :ref:`basics`.)

The ``.run(...)`` method
------------------------

.. code:: python

    program = Program(X(0), MEASURE(0, 0))
    results = qvm.run(program)
    # results = [[1]]

The ``.run(...)`` method takes numerous arguments, several of which are optional. The most important
are

1. the ``program`` to be executed on the QVM,
2. the ``classical_addresses`` which to be returned from the QVM (not included above; by default, these are set to the addresses used in the program's ``MEASURE`` instructions), and
3. the number of ``trials`` to be executed on the machine.

The results returned are a *list of lists of integers*. In the above case, that's

.. parsed-literal::

    [[1]]

Let's unpack this. The *outer* list is an
enumeration over the trials; if you set ``trials=1`` then ``len(results)`` should equal ``1``.

The *inner* list, on the other hand, is an enumeration over the results stored in the classical
addresses. We see that the result of this program is that the classical register ``[0]`` now stores
the state of qubit 0, which should be ``1`` after an :math:`X`-gate. We
can of course ask for more classical registers:

.. code:: python

    qvm.run(p, [0, 1, 2])

.. parsed-literal::

    [[1, 0, 0]]

The classical registers are initialized to zero, so registers ``[1]``
and ``[2]`` come out as zero. If we stored the measurement in a
different classical register we would obtain:

.. code:: python

    p = Program()   # clear the old program
    p.inst(X(0)).measure(0, 1)
    qvm.run(p, [0, 1, 2])

.. parsed-literal::

    [[0, 1, 0]]

We can also run programs multiple times and accumulate all the results
in a single list.

.. code:: python

    coin_flip = Program().inst(H(0)).measure(0, 0)
    num_flips = 5
    qvm.run(coin_flip, [0], num_flips)

.. parsed-literal::

    [[0], [1], [0], [1], [0]]

Try running the above code several times. You will see that you will,
with very high probability, get different results each time.

.. [1] https://arxiv.org/abs/1608.03355

Simulating the QPU using the QVM
--------------------------------

The QVM is a powerful tool for testing quantum programs before executing them on the QPU. In
addition to the ``noise.py`` module for generating custom noise models for simulating noise on the
QVM, pyQuil provides a simple interface for loading the QVM with noise models tailored to Rigetti's
available QPUs, in just one modified line of code. This is made possible via the ``Device`` class,
which holds hardware specification information, noise model information, and instruction set
architecture (ISA) information regarding connectivity. This information is held in the ``Specs``,
``ISA`` and ``NoiseModel`` attributes of the ``Device`` class, respectively.

Specifically, to load a QVM with the ``NoiseModel`` information from a ``Device``, all that is
required is to provide a ``Device`` object to the QVM during initialization:

.. note::

    This feature is currently deprecated, in advance of a new QPU (with new noise models). For users interested in
    creating noise models for the QVM, you can do so by following the instructions in :ref:`noise`.

.. code:: python

    from pyquil.api import get_devices, QVMConnection

    device_name = get_device('quantum_device_name')
    qvm = QVMConnection(device_name)

By simply providing a device during QVM initialization, all programs executed on this QVM will, by
default, have noise applied that is characteristic of the corresponding Rigetti QPU (in the case
above, the ``agave`` device). One may then efficiently test realistic quantum algorithms on the QVM,
in advance of running those programs on the QPU.
