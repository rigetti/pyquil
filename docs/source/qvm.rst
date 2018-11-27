.. _qvm:

The Quantum Computer
====================

``QuantumComputer``
~~~~~~~~~~~~~~~~~~~

The :py:class:`~pyquil.api.QuantumComputer` abstraction offered by pyQuil provides an easy access point to the most
critical objects used in pyQuil for building and executing your quantum programs.
We will cover the main methods and attributes on this page.
The `QuantumComputer API Reference <apidocs/quantum_computer.html>`_ provides a reference for all of its methods and
options.

At a high level, the :py:class:`~pyquil.api.QuantumComputer` wraps around our favorite quantum computing tools:

  - **A compiler** ``.compiler`` : this determines how we manipulate the Quil input to something more efficient when possible,
    and then into a form which our QAM can accept as input.
  - **A device** ``.device`` : this specifies the topology and Instruction Set Architecture (ISA) of
    the targeted device by listing the supported 1Q and 2Q gates.
  - **A quantum abstract machine** ``.qam`` : this is our general purpose quantum computing device,
    which implements the required abstract methods to manipulate classical and quantum state, such as
    loading programs, writing to shared classical memory, and executing programs. It is implemented as a :py:class:`~pyquil.api.QVM`
    (quantum virtual machine) or
    :py:class:`~pyquil.api.QPU` (quantum processing unit) object in pyQuil.

The :py:class:`~pyquil.api.QuantumComputer` instance makes methods available which are built on the above objects. If
you need more fine grained controls for your work, you might try exploring what is offered by these objects.

For more information on each of the above, check out the following pages:

 - `Compiler API Reference <apidocs/compilers.html>`_
 - :ref:`Quil Compiler docs <compiler>`
 - `Device API Reference <apidocs/devices.html>`_
 - :ref:`new_topology`
 - `Quantum abstract machine (QAM) API Reference <apidocs/qam.html>`_
 - `The Quil Whitepaper <https://arxiv.org/abs/1608.03355>`_ which describes the QAM

Instantiation
-------------

A decent amount of information needs to be provided to initialize the ``compiler``, ``device``, and ``qam`` attributes,
much of which is already in your :ref:`config files <_advanced_usage>` (or provided reasonable defaults when running locally).
Typically, you will want a :py:class:`~pyquil.api.QuantumComputer` which either:

  - pertains to a real, available QPU device
  - is a QVM but mimics the topology of a QPU, with or without noise
  - is some a generic QVM

All of this can be accomplished with :py:func:`~pyquil.api.get_qc`.

.. code:: python

    def get_qc(name: str, *, as_qvm: bool = None, noisy: bool = None,
               connection: ForestConnection = None) -> QuantumComputer:

.. code:: python

    from pyquil import get_qc

    # Get a QPU
    qc = get_qc(QPU_LATTICE_NAME)  # This is just a string naming the device

    # Get a QVM with the same topology as the QPU lattice, without noise
    qc = get_qc(QPU_LATTICE_NAME, as_qvm=True)
    # or, equivalently
    qc = get_qc(f"{QPU_LATTICE_NAME}-qvm")

    # With noise
    qc = get_qc(QPU_LATTICE_NAME, as_qvm=True, noisy=True)

    # A fully connected QVM
    number_of_qubits = 10
    qc = get_qc(f"{number_of_qubits}q-qvm")

For now, you will have to join QCS to get ``QPU_LATTICE_NAME`` by running the
``qcs lattices`` command from your QMI. Access to the QPU is only possible from a QMI, during a booked reservation.
If this sounds unfamiliar, check out our `documentation for QCS <https://www.rigetti.com/qcs/docs/intro-to-qcs>`_
and `join the waitlist <https://www.rigetti.com/>`_.

For more information about creating your own noise models, check out :ref:`noise`.

.. note::
    When connecting to a QVM locally (such as with ``get_qc(..., as_qvm=True)``) you'll have to set up the QVM
    in :ref:`server mode <server>`.

Methods
-------

Now that you have your ``qc``, there's a lot you can do with it. Most users will want to use ``compile``, ``run`` or
``run_and_measure``, and ``qubits`` very regularly. The general flow of use would look like this:

.. code:: python

    from pyquil import get_qc, Program
    from pyquil.gates import *

    qc = get_qc('9q-square-qvm')            # not general to any number of qubits, 9q-square-qvm is special

    qubits = qc.qubits()                    # this information comes from qc.device
    p = Program()
    # ... build program, potentially making use of the qubits list

    compiled_program = qc.compile(p)        # this makes multiple calls to qc.compiler

    results = qc.run(compiled_program)      # this makes multiple calls to qc.qam

.. note::

    In addition to a running QVM server, you will need a running ``quilc`` server to compile your program. Setting
    up both of these is very easy, as explained :ref:`here <server>`.


The ``.run_and_measure(...)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most high level way to run your program. With this method, you are **not** responsible for compiling your program
before running it, nor do you have to specify any ``MEASURE`` instructions; all qubits will get measured.

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import X

    qc = get_qc("8q-qvm")

    p = Program(X(0))

    results = qc.run_and_measure(p, trials=5)
    print(results)

Trials specifies how many times to run this program. Let's see our results:

.. parsed-literal::

    {0: array([1, 1, 1, 1, 1]),
     1: array([0, 0, 0, 0, 0]),
     2: array([0, 0, 0, 0, 0]),
     3: array([0, 0, 0, 0, 0]),
     4: array([0, 0, 0, 0, 0]),
     5: array([0, 0, 0, 0, 0]),
     6: array([0, 0, 0, 0, 0]),
     7: array([0, 0, 0, 0, 0])}

The return value is a dictionary from qubit index to results for all trials.
Every qubit in the lattice is measured for you, and as expected, qubit 0 has been flipped to the excited state
for each trial.

The ``.run(...)`` method
^^^^^^^^^^^^^^^^^^^^^^^^

The lower-level ``.run(...)`` method gives you more control over how you want to build and compile your program than
``.run_and_measure`` does. **You are responsible for compiling your program before running it.**
The above program would be written in this way to execute with ``run``:

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import X

    qc = get_qc("8q-qvm")

    p = Program()
    ro = p.declare('ro', 'BIT', 1)
    p += X(0)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    p.wrap_in_numshots_loop(5)

    executable = qc.compile(p)
    bitstrings = qc.run(executable)  # .run takes in a compiled program, unlike .run_and_measure
    print(bitstrings)

By specifying ``MEASURE`` ourselves, we will only get the results that we are interested in. To be completely equivalent
to the previous example, we would have to measure all eight qubits.

The results returned is a *list of lists of integers*. In the above case, that's

.. parsed-literal::

    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]

Let's unpack this. The *outer* list is an enumeration over the trials; the argument given to
``wrap_in_numshots_loop`` will match the length of ``results``.

The *inner* list, on the other hand, is an enumeration over the results stored in the memory region named ``ro``, which
we use as our readout register. We see that the result of this program is that the memory region ``ro[0]`` now stores
the state of qubit 0, which should be ``1`` after an :math:`X`-gate. See :ref:`declaring_memory` and :ref:`measurement`
for more details about declaring and accessing classical memory regions.

.. tip:: Get the results for qubit 0 with ``numpy.array(bitstrings)[:,0]``.

.. _new_topology:

Providing Your Own Device Topology
----------------------------------

It is simple to provide your own device topology as long as you can give your qubits each a number,
and specify which edges exist. Here is an example, using the topology of our 16Q chip (two octagons connected by a square):

.. code:: python

    import networkx as nx

    from pyquil.device import NxDevice, gates_in_isa
    from pyquil.noise import decoherence_noise_with_asymmetric_ro

    qubits = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]  # qubits are numbered by octagon
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5) (5, 6), (6, 7), (7, 0),  # first octagon
             (1, 16), (2, 15),  # connections across the square
             (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (16, 17), (10, 17)] # second octagon

    # Build the NX graph
    topo = nx.from_edgelist(edges)
    # You would uncomment the next line if you have disconnected qubits
    # topo.add_nodes_from(qubits)
    device = NxDevice(topo)
    device.noise_model = decoherence_noise_with_asymmetric_ro(gates_in_isa(device.get_isa()))  # Optional

Now that you have your device, you could set ``qc.device`` and ``qc.compiler.device`` to point to your new device,
or use it to make new objects.

Simulating the QPU using the QVM
--------------------------------

The :py:class:`~pyquil.api.QAM` methods are intended to be used in the same way, whether a QVM or QPU is being targeted.
Everywhere on this page,
you can swap out the type of the QAM (quantum abstract machine -- remember, a QVM or a QPU) and you will still
get reasonable results back. As long as the topology of the devices are the same, programs compiled and ran on the QVM
will be able to run on the QPU and visa-versa. Since :py:class:`~pyquil.api.QuantumComputer` is built on the ``QAM``
abstract class, its methods will also work for both QAM implementations.

This makes the QVM is a powerful tool for testing quantum programs before executing them on the QPU.

.. code:: python

    qpu = get_qc(QPU_LATTICE_NAME)
    qvm = get_qc(QPU_LATTICE_NAME, as_qvm=True)

By simply providing ``as_qvm=True``, we get a QVM which will have the same topology as
the named QPU. It's a good idea to run your programs against the QVM before booking QPU time to iron out
bugs. You can also provide ``noisy=True`` to get a noisy QVM. To learn more about how to add noise models to your virtual ``QuantumComputer`` instance, check out
:ref:`noise`.

.. _qvm_use:

The Quantum Virtual Machine (QVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pyQuil :py:class:`~pyquil.api.QVM` object, which fills the :py:class:`~pyquil.api.QuantumComputer` ``qam`` attribute,
is essentially a client around a Rigetti QVM endpoint.

The Rigetti Quantum Virtual Machine is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_  It is implemented in ANSI Common LISP and
executes programs specified in the Quantum Instruction Language (Quil).

Quil is an opinionated quantum instruction language: its basic belief is that in the near term quantum computers will
operate as coprocessors, working in concert with traditional CPUs.  This means that Quil is designed to execute on
a Quantum Abstract Machine that has a shared classical/quantum architecture at its core.

The QVM is a wavefunction simulation of unitary evolution with classical control flow
and shared quantum classical memory.

The QVM is part of the Forest SDK, and it's available for you to use on your local machine.
After :ref:`downloading the SDK <sdkinstall>`, you can initialize a local
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


We also offer a Wavefunction Simulator (formerly a part of the :py:class:`~pyquil.api.QVM` object),
which allows users to contruct and inspect wavefunctions of quantum programs. Learn more
about the Wavefunction Simulator :ref:`here <wavefunction_simulator>`. For information on constructing quantum
programs, please refer back to :ref:`basics`.


.. [1] https://arxiv.org/abs/1608.03355

