.. _the_quantum_computer:

====================
The quantum computer
====================

pyQuil is used to build Quil (Quantum Instruction Language) programs, simulate them, and execute them on real quantum
processors. Quil is an opinionated quantum instruction language: its basic belief is that in the near term quantum
computers will operate as coprocessors, working in concert with traditional CPUs. This means that Quil is designed to
execute on a Quantum Abstract Machine (QAM) that has a shared classical/quantum architecture at its core.

A QAM must, therefore, implement certain abstract methods to manipulate classical and quantum states, such as loading
programs, writing to shared classical memory, and executing programs. Within pyQuil, the :py:class:`~pyquil.api.QPU`
object implements the QAM by using the APIs of a Quantum Processing Unit (QPU) through Rigetti's Quantum Cloud Services.

On this page, we'll learn a bit about the :ref:`QPU <qpu>`, and then show you how to use it from pyQuil with a
:ref:`QuantumComputer <quantum_computer>` object. To simulate programs on your own machine instead, see
:ref:`local_simulation`.

For information on constructing quantum programs, please refer back to :ref:`basics`.

.. _local_simulation:

****************
Local simulation
****************

pyQuil simulates programs in-process, with no server, using the JAX-based simulators described in :ref:`simulation`.
A simulator is constructed from a program, not from a :py:class:`~pyquil.api.QuantumComputer`, and returns either the
final quantum state or sampled measurement outcomes. Noise is added with a noise model (see :ref:`noise`).

To check that a program will compile for a particular quantum processor, compile it with that processor's
:py:class:`~pyquil.api.QuantumComputer` (see :ref:`compiler`) and simulate the result; a compiled program is an
ordinary :py:class:`~pyquil.Program`.

.. _qpu:

*********************************
The Quantum Processing Unit (QPU)
*********************************

To access a QPU endpoint, you will have to `sign up <https://www.rigetti.com/>`_ for Quantum Cloud Services (QCS).
Documentation for getting started can be found `here <https://docs.rigetti.com>`_. Once you've been authorized to
access a QPU you can submit requests to it using pyQuil.

For information on available QPUs, you can check out `your dashboard <https://qcs.rigetti.com/dashboard>`_ after you've
been invited to QCS.

.. _quantum_computer:

***********************
The ``QuantumComputer``
***********************

The :py:class:`~pyquil.api.QuantumComputer` abstraction offered by pyQuil provides an easy access point to the most
critical objects used in pyQuil for compiling and executing your quantum programs. We will cover the main methods and attributes
on this page. The `QuantumComputer API Reference <apidocs/pyquil.api.html#pyquil.api.QuantumComputer>`_ provides a reference for all of its methods
and options.

At a high level, the :py:class:`~pyquil.api.QuantumComputer` wraps around our favorite quantum computing tools:

  - **A quantum abstract machine** ``.qam`` : this is our general purpose quantum computing device,
    which implements the required abstract methods described :ref:`above <the_quantum_computer>`. For a real quantum
    processor it is a :py:class:`~pyquil.api.QPU` object.
  - **A compiler** ``.compiler`` : this determines how we manipulate the Quil input to something more efficient when possible,
    and then into a form which our QAM can accept as input.
  - **A quantum processor** ``.quantum_processor`` : this specifies the topology and Instruction Set Architecture (ISA) of
    the targeted processor by listing the supported 1Q and 2Q gates.

When you instantiate a :py:class:`~pyquil.api.QuantumComputer` instance, these subcomponents will be compatible with
each other. So, if you get a ``QPU`` implementation for the ``.qam``, you will have a ``QPUCompiler`` for the
``.compiler``, and your ``.quantum_processor`` will match the processor used by the ``.compiler``.

The :py:class:`~pyquil.api.QuantumComputer` instance makes methods available which are built on the above objects. If
you need more fine grained controls for your work, you might try exploring what is offered by these objects.

For more information on each of the above, check out the following pages:

 - :ref:`Quil Compiler docs <compiler>`
 - :ref:`new_topology`
 - `Quantum abstract machine (QAM) API Reference <apidocs/pyquil.api.html#pyquil.api.QAM>`_
 - `The Quil Whitepaper <https://arxiv.org/abs/1608.03355>`_ which describes the QAM

Instantiation
=============

A decent amount of information needs to be provided to initialize the ``compiler``, ``quantum_processor``, and ``qam`` attributes,
much of which is already in your :ref:`config files <advanced_usage>` (or provided reasonable defaults when running locally).
Typically, you will want a :py:class:`~pyquil.api.QuantumComputer` which either:

  - pertains to a real, available QPU, to compile and run programs on it, or
  - is a generic lattice, such as a fully connected or square lattice of qubits, to compile programs against locally.

Both can be accomplished with :py:func:`~pyquil.api.get_qc`.

.. testcode:: instantiation

    from pyquil import get_qc

    QPU_NAME = "Ankaa-3"

    # Get a QPU
    # qc = get_qc(QPU_NAME)  # QPU_NAME is just a string naming the quantum_processor

    # A fully connected lattice of 10 qubits, for compiling only
    number_of_qubits = 10
    qc = get_qc(f"{number_of_qubits}q")

A generic lattice such as ``"10q"`` or ``"9q-square"`` has a compiler and a quantum processor, but no QAM: it compiles
programs, which you can then simulate (see :ref:`simulation`), but it can't run them.

As a reminder, you will have to join QCS to get access to a specific quantum processor.
Check out our `documentation for QCS <https://docs.rigetti.com>`_ and `join the waitlist <https://www.rigetti.com/>`_ if you don't have access already.

.. note::

    This page just covers the essentials, but you can customize the behavior of compilation, execution and more using the
    various parameters on :py:func:`~pyquil.api.get_qc`, see the API documentation to see everything that is available.

.. note::

    Compiling, with a QPU or a generic lattice, needs a running ``quilc``; see :ref:`server mode <server>`.

Methods
=======

Now that you have your ``qc``, there's a lot you can do with it. Most users will want to use ``compile``, ``run`` very
regularly. The general flow of use would look like this:

.. code:: python

    from pyquil import get_qc, Program
    from pyquil.gates import *

    qc = get_qc("Ankaa-3")

    qubits = qc.qubits()                    # this information comes from qc.quantum_processor
    p = Program()
    # ... build program, potentially making use of the qubits list

    compiled_program = qc.compile(p)        # this makes multiple calls to qc.compiler

    results = qc.run(compiled_program)      # this makes multiple calls to qc.qam

The ``.run(...)`` method
------------------------

When using the ``.run(...)`` method, **you are responsible for compiling your program before running it.**
For example:

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import X, MEASURE

    qc = get_qc("Ankaa-3")

    p = Program()
    ro = p.declare('ro', 'BIT', 2)
    p += X(0)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    p.wrap_in_numshots_loop(5)

    executable = qc.compile(p)
    result = qc.run(executable)  # .run takes in a compiled program
    bitstrings = result.get_register_map().get("ro")
    print(bitstrings)

The results returned is a *list of lists of integers*. In the above case, on a noiseless device, that would be

.. code:: text

    [[1 0]
     [1 0]
     [1 0]
     [1 0]
     [1 0]]

Let's unpack this. The *outer* list is an enumeration over the trials; the argument given to
``wrap_in_numshots_loop`` will match the length of ``results``.

The *inner* list, on the other hand, is an enumeration over the results stored in the memory region named ``ro``, which
we use as our readout register. We see that the result of this program is that the memory region ``ro[0]`` now stores
the state of qubit 0, which should be ``1`` after an :math:`X`-gate. See :ref:`declaring_memory` and :ref:`measurement`
for more details about declaring and accessing classical memory regions.

.. tip:: Get the results for qubit 0 with ``numpy.array(bitstrings)[:,0]``.

In addition to readout data, the result of ``.run(...)`` includes other information about the job's execution, such
as the run duration. See :py:class:`~pyquil.api.QAMExecutionResult` for details.

``.execute`` and ``.get_result``
--------------------------------

The ``.run(...)`` method is itself a convenience wrapper around two other methods: ``.execute(...)`` and
``.get_result(...)``. ``run`` makes your program appear synchronous (request and then wait for the response),
when in reality on some backends (such as a live QPU), execution is in fact asynchronous (request execution,
then request results at a later time). For finer-grained control over your program execution process,
you can use these two methods in place of ``.run``. This is most useful when you want to execute work
concurrently - for that, please see :ref:`advanced_usage`.

QPU-specific features
=====================

The subcomponents of a :py:class:`~pyquil.api.QuantumComputer` follow common interfaces, namely
:py:class:`~pyquil.api.QAM` and :py:class:`~pyquil.api.AbstractCompiler`, but a QPU-based ``QuantumComputer`` offers
some methods and properties that a generic lattice does not: ``qc.qam`` is a :py:class:`~pyquil.api.QPU` instance and
``qc.compiler`` is a :py:class:`~pyquil.api.QPUCompiler` instance.

You can access these features and keep your code robust by performing type checks on ``qc.qam`` and/or ``qc.compiler``.
For example, if you wanted to refresh the calibration program, which only applies to QPU-based ``QuantumComputers``, but still
want a script that also compiles against a generic lattice, you could do the following:

.. testcode:: differences

    from pyquil import get_qc
    from pyquil.api import QPUCompiler

    qc = get_qc("2q")  # or "Ankaa-3"

    if isinstance(qc.compiler, QPUCompiler):
        # Working with a QPU - refresh calibrations
        qc.compiler.get_calibration_program(force_refresh=True)

Requesting a job cancellation
-----------------------------

Jobs submitted to a QPU are queued for execution before they are run. While a job is pending execution, you can request
that the job be cancelled with :py:meth:`~pyquil.api.QPU.cancel`. Use a similar strategy to the one above to make sure
the Quantum Abstract Machine (QAM) backing the QuantumComputer is indeed a QPU before calling this method:

.. code:: python

    from pyquil.api import QPU

    job_handle = qc.qam.execute(p)

    if isinstance(qc.qam, QPU):
        try:
            qc.qam.cancel(job_handle)
            print("Job was cancelled")
        except QpuApiError:
            # If this error was raised, then the job failed to be cancelled.
            result = qc.qam.get_result(job_handle)

.. _new_topology:

Providing your own quantum processor topology
=============================================

You can provide your own quantum processor topology by specifying qubits (as numeric indices) and edges.
Here is an example that uses a subset of the instruction set architecture of a
Rigetti QPU to specify a 16 qubit topology.

.. code:: python

    import networkx as nx
    from pyquil import get_qc
    from pyquil.api import QuilcCompiler
    from pyquil.quantum_processor import NxQuantumProcessor

    qpu = get_qc("Ankaa-3")
    isa = qpu.to_compiler_isa()
    qubits = sorted(int(k) for k in isa.qubits.keys())[:16]
    edges = [(q1, q2) for q1 in qubits for q2 in qubits if f"{q1}-{q2}" in isa.edges]

    # Build the NX graph
    topo = nx.from_edgelist(edges)
    # You would uncomment the next line if you have disconnected qubits
    # topo.add_nodes_from(qubits)
    quantum_processor = NxQuantumProcessor(topo)

    # A compiler that targets the new topology
    compiler = QuilcCompiler(quantum_processor=quantum_processor)

Programs compiled with ``compiler.quil_to_native_quil(program)`` respect the topology, and can be simulated as in
:ref:`simulation`. To simulate a device's noise as well, build a noise model for it (see :ref:`noise`).
