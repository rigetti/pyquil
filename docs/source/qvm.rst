.. _qvm:

The Quantum Computer
====================

PyQuil is used to build Quil (Quantum Instruction Language) programs and execute them on simulated or real quantum devices. Quil is an opinionated
quantum instruction language: its basic belief is that in the near term quantum computers will
operate as coprocessors, working in concert with traditional CPUs. This means that Quil is designed to execute on
a Quantum Abstract Machine (QAM) that has a shared classical/quantum architecture at its core.

A QAM must, therefore, implement certain abstract methods to manipulate classical and quantum states, such as loading
programs, writing to shared classical memory, and executing programs.

The program execution itself is sent from pyQuil to quantum computer endpoints, which will be one of two options:

  - A Rigetti Quantum Virtual Machine (QVM)
  - A Rigetti Quantum Processing Unit (QPU)

Within pyQuil, there is a :py:class:`~pyquil.api.QVM` object and a :py:class:`~pyquil.api.QPU` object which use
the exposed APIs of the QVM and QPU servers, respectively.

On this page, we'll learn a bit about the :ref:`QVM <qvm_use>` and :ref:`QPU <qpu>`. Then we will
show you how to use them from pyQuil with a :ref:`QuantumComputer <quantum_computer>` object.

For information on constructing quantum programs, please refer back to :ref:`basics`.

.. _qvm_use:

The Quantum Virtual Machine (QVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rigetti Quantum Virtual Machine is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_  It is implemented in ANSI Common LISP and
executes programs specified in Quil.

The QVM simulates the unitary evolution of a wavefunction with
classical control. The QVM has a plethora of other features,
including:

  - Stochastic pure-state evolution, density matrix evolution, and
    Pauli noise channels;
  - Shared memory access to the quantum state, allowing direct NumPy
    access to the state without copying or transmission delay; and
  - A fast just-in-time compilation mode for rapid simulation of large
    programs with many qubits.

The QVM is part of the Forest SDK, and it's available for you to use on your local machine.
After :ref:`downloading and installing the SDK <sdkinstall>`, you can initialize a local
QVM server by typing ``qvm -S`` into your terminal. You should see the following message.

.. code:: text

    $ qvm -S
    ******************************
    * Welcome to the Rigetti QVM *
    ******************************
    Copyright (c) 2018 Rigetti Computing.

    (Configured with 2048 MiB of workspace and 8 workers.)

    [2018-11-06 18:18:18] Starting server on port 5000.

By default, the server is started on port 5000 on your local machine. Consequently, the endpoint which
the pyQuil :py:class:`~pyquil.api.QVM` will default to for the QVM address is ``http://127.0.0.1:5000``. When you
run your program, a pyQuil client will send a Quil program to the QVM server and wait for a response back.

It's also possible to use the QVM from the command line. You can write a Quil program in its own file:

.. code:: text

    # example.quil

    DECLARE ro BIT[1]
    RX(pi/2) 0
    CZ 0 1

and then execute it with the QVM directly from the command line:

.. code:: text

    $ qvm < example.quil

    [2018-11-30 11:13:58] Reading program.
    [2018-11-30 11:13:58] Allocating memory for QVM of 2 qubits.
    [2018-11-30 11:13:58] Allocation completed in 4 ms.
    [2018-11-30 11:13:58] Loading quantum program.
    [2018-11-30 11:13:58] Executing quantum program.
    [2018-11-30 11:13:58] Execution completed in 6 ms.
    [2018-11-30 11:13:58] Printing 2-qubit state.
    [2018-11-30 11:13:58] Amplitudes:
    [2018-11-30 11:13:58]   |00>: 0.0, P=  0.0%
    [2018-11-30 11:13:58]   |01>: 0.0-1.0i, P=100.0%
    [2018-11-30 11:13:58]   |10>: 0.0, P=  0.0%
    [2018-11-30 11:13:58]   |11>: 0.0, P=  0.0%
    [2018-11-30 11:13:58] Classical memory (low -> high indexes):
    [2018-11-30 11:13:58]     ro:  1 0

The QVM offers a simple benchmarking mode with ``qvm --verbose
--benchmark``. Example output looks like this:

.. code:: text

   $ ./qvm --verbose --benchmark
   ******************************
   * Welcome to the Rigetti QVM *
   ******************************
   Copyright (c) 2016-2019 Rigetti Computing.

   (Configured with 8192 MiB of workspace and 8 workers.)

   <135>1 2019-05-01T18:26:14Z workstation.local qvm 96177 - - Selected simulation method: pure-state
   <135>1 2019-05-01T18:26:15Z workstation.local qvm 96177 - - Computing baseline serial norm timing...
   <135>1 2019-05-01T18:26:15Z workstation.local qvm 96177 - - Baseline serial norm timing: 96 ms
   <135>1 2019-05-01T18:26:15Z workstation.local qvm 96177 - - Starting "bell" benchmark with 26 qubits...

   ; Transition H 0 took 686 ms (gc: 0 ms; alloc: 65536 bytes)
   ; Transition CNOT 0 1 took 651 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition CNOT 1 2 took 658 ms (gc: 0 ms; alloc: 32656 bytes)
   ; Transition CNOT 2 3 took 661 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition CNOT 3 4 took 650 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition CNOT 4 5 took 662 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition CNOT 5 6 took 673 ms (gc: 0 ms; alloc: 0 bytes)
   [...]
   <135>1 2019-05-01T18:30:13Z workstation.local qvm 96288 - - Total time for program run: 24385 ms

The QVM also has mode for faster execution of long quantum programs
operating on a large number of qubits, called **compiled
mode**. Compiled mode can be enabled by adding ``-c`` to the command
line options. Observe the speed-up in the benchmark:

.. code:: text

   $ ./qvm --verbose --benchmark -c
   ******************************
   * Welcome to the Rigetti QVM *
   ******************************
   Copyright (c) 2016-2019 Rigetti Computing.

   (Configured with 8192 MiB of workspace and 8 workers.)

   <135>1 2019-05-01T18:28:07Z workstation.local qvm 96285 - - Selected simulation method: pure-state
   <135>1 2019-05-01T18:28:08Z workstation.local qvm 96285 - - Computing baseline serial norm timing...
   <135>1 2019-05-01T18:28:08Z workstation.local qvm 96285 - - Baseline serial norm timing: 95 ms
   <135>1 2019-05-01T18:28:08Z workstation.local qvm 96285 - - Starting "bell" benchmark with 26 qubits...

   ; Compiling program loaded into QVM...
   ; Compiled in 87 ms.
   ; Optimization eliminated 26 instructions ( 50.0%).
   ; Transition compiled{ FUSED-GATE-0 1 0 } took 138 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition compiled{ CNOT 1 2 } took 144 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition compiled{ CNOT 2 3 } took 137 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition compiled{ CNOT 3 4 } took 143 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition compiled{ CNOT 4 5 } took 95 ms (gc: 0 ms; alloc: 0 bytes)
   ; Transition compiled{ CNOT 5 6 } took 75 ms (gc: 0 ms; alloc: 0 bytes)
   [...]
   <135>1 2019-05-01T18:29:12Z workstation.local qvm 96287 - - Total time for program run: 2416 ms

The runtime reduced to 2.4 seconds from 24 seconds, a 10x speedup.

.. note::
   Compiled mode speeds up the execution of a program at the
   cost of an initial compilation. Note in the above example that
   compilation took 87 ms.  If you are running small programs with low
   qubit counts, this cost may be significant, and it may be worth
   executing in the usual ("interpreted") mode. However, if your
   programs contain a large number of qubits or a large number of
   instructions, the initial cost is far outweighed by the benefits.

For a detailed description of how to use the ``qvm`` from the command line, see the QVM `README
<https://github.com/rigetti/qvm>`_ or type ``man qvm`` in your terminal.

We also offer a Wavefunction Simulator (formerly a part of the :py:class:`~pyquil.api.QVM` object),
which allows users to contruct and inspect wavefunctions of quantum programs. Learn more
about the Wavefunction Simulator :ref:`here <wavefunction_simulator>`.

.. _qpu:

The Quantum Processing Unit (QPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access a QPU endpoint, you will have to `sign up <https://qcs.rigetti.com/>`_ for Quantum Cloud Services (QCS).
Documentation for getting started with your Quantum Machine Image (QMI) is found
`here <https://www.rigetti.com/qcs-docs>`_. Using QCS, you will ``ssh`` into your QMI, and reserve a
QPU lattice for a particular time block.

When your reservation begins, you will be authorized to access the QPU. A configuration file will be
automatically populated for you with the proper QPU endpoint for your reservation. Both your QMI and the QPU
are located on premises, giving you low latency access to the QPU server. That server accepts jobs in the form
of a ``BinaryExecutableRequest`` object, which is precisely what you get back when you compile your program in
pyQuil and target the QPU (more on this soon).  This request contains all the information necessary to run
your program on the control rack which sends and receives waveforms from the QPU, so that you can receive
classical binary readout results.

For information on available lattices, you can check out your dashboard at https://qcs.rigetti.com/dashboard after you've
been invited to QCS.


.. _quantum_computer:

The ``QuantumComputer``
~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`~pyquil.api.QuantumComputer` abstraction offered by pyQuil provides an easy access point to the most
critical objects used in pyQuil for building and executing your quantum programs.
We will cover the main methods and attributes on this page.
The `QuantumComputer API Reference <apidocs/quantum_computer.html>`_ provides a reference for all of its methods and
options.

At a high level, the :py:class:`~pyquil.api.QuantumComputer` wraps around our favorite quantum computing tools:

  - **A quantum abstract machine** ``.qam`` : this is our general purpose quantum computing device,
    which implements the required abstract methods described :ref:`above <qvm>`. It is implemented as a
    :py:class:`~pyquil.api.QVM` or :py:class:`~pyquil.api.QPU` object in pyQuil.
  - **A compiler** ``.compiler`` : this determines how we manipulate the Quil input to something more efficient when possible,
    and then into a form which our QAM can accept as input.
  - **A device** ``.device`` : this specifies the topology and Instruction Set Architecture (ISA) of
    the targeted device by listing the supported 1Q and 2Q gates.

When you instantiate a :py:class:`~pyquil.api.QuantumComputer` instance, these subcomponents will be compatible with
each other. So, if you get a ``QPU`` implementation for the ``.qam``, you will have a ``QPUCompiler`` for the
``.compiler``, and your ``.device`` will match the device used by the ``.compiler.``

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
much of which is already in your :ref:`config files <advanced_usage>` (or provided reasonable defaults when running locally).
Typically, you will want a :py:class:`~pyquil.api.QuantumComputer` which either:

  - pertains to a real, available QPU device
  - is a QVM but mimics the topology of a QPU
  - is some generic QVM

All of this can be accomplished with :py:func:`~pyquil.api.get_qc`.

.. code:: python

    def get_qc(name: str, *, as_qvm: bool = None, noisy: bool = None,
               connection: ForestConnection = None) -> QuantumComputer:

.. code:: python

    from pyquil import get_qc

    # Get a QPU
    qc = get_qc(QPU_LATTICE_NAME)  # QPU_LATTICE_NAME is just a string naming the device

    # Get a QVM with the same topology as the QPU lattice
    qc = get_qc(QPU_LATTICE_NAME, as_qvm=True)
    # or, equivalently
    qc = get_qc(f"{QPU_LATTICE_NAME}-qvm")

    # A fully connected QVM
    number_of_qubits = 10
    qc = get_qc(f"{number_of_qubits}q-qvm")

For now, you will have to join QCS to get ``QPU_LATTICE_NAME`` by running the
``qcs lattices`` command from your QMI. Access to the QPU is only possible from a QMI, during a booked reservation.
If this sounds unfamiliar, check out our `documentation for QCS <https://www.rigetti.com/qcs-docs>`_
and `join the waitlist <https://qcs.rigetti.com/>`_.

For more information about creating and adding your own noise models, check out :ref:`noise`.

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

``trials`` specifies how many times to run this program. Let's see our results:

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
    from pyquil.gates import X, MEASURE

    qc = get_qc("8q-qvm")

    p = Program()
    ro = p.declare('ro', 'BIT', 2)
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
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),  # first octagon
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
you can swap out the type of the QAM (QVM <=> QPU) and you will still
get reasonable results back. As long as the topologies of the devices are the same, programs compiled and run on the QVM
will be able to run on the QPU and vice versa. Since :py:class:`~pyquil.api.QuantumComputer` is built on the ``QAM``
abstract class, its methods will also work for both QAM implementations.

This makes the QVM a powerful tool for testing quantum programs before executing them on the QPU.

.. code:: python

    qpu = get_qc(QPU_LATTICE_NAME)
    qvm = get_qc(QPU_LATTICE_NAME, as_qvm=True)

By simply providing ``as_qvm=True``, we get a QVM which will have the same topology as
the named QPU. It's a good idea to run your programs against the QVM before booking QPU time to iron out
bugs. To learn more about how to add noise models to your virtual ``QuantumComputer`` instance, check out
:ref:`noise`.

In the next section, we will see how to use the Wavefunction Simulator aspect of the Rigetti QVM to inspect the full
wavefunction set up by a Quil program.

.. [1] https://arxiv.org/abs/1608.03355

