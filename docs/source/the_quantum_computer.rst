.. _the_quantum_computer:

====================
The quantum computer
====================

pyQuil is used to build Quil (Quantum Instruction Language) programs and execute them on simulated or real quantum processors. Quil is an opinionated
quantum instruction language: its basic belief is that in the near term quantum computers will
operate as coprocessors, working in concert with traditional CPUs. This means that Quil is designed to execute on
a Quantum Abstract Machine (QAM) that has a shared classical/quantum architecture at its core.

A QAM must, therefore, implement certain abstract methods to manipulate classical and quantum states, such as loading
programs, writing to shared classical memory, and executing programs.

The program execution itself is sent from pyQuil to quantum computer endpoints, which will be one of two options:

  - A Quantum Virtual Machine (QVM)
  - A Quantum Processing Unit (QPU)

Within pyQuil, there is a :py:class:`~pyquil.api.QVM` object and a :py:class:`~pyquil.api.QPU` object which use
the exposed APIs of the QVM and QPU servers, respectively.

On this page, we'll learn a bit about the :ref:`QVM <qvm_use>` and :ref:`QPU <qpu>`. Then we will
show you how to use them from pyQuil with a :ref:`QuantumComputer <quantum_computer>` object.

For information on constructing quantum programs, please refer back to :ref:`basics`.

.. _qvm_use:

*********************************
The Quantum Virtual Machine (QVM)
*********************************

The Quantum Virtual Machine is an implementation of the Quantum Abstract Machine from *A Practical Quantum Instruction Set Architecture*. [1]_  It is implemented in ANSI Common LISP and
executes programs specified in Quil.

As we learned in the :ref:`pre-requisites<prerequisites>` the QVM is part of the Quil SDK, and it's available for you
to use on your local machine.

For a detailed description of how to use the ``qvm`` from the command line, see the QVM `README
<https://github.com/rigetti/qvm>`_ or type ``man qvm`` in your terminal.

We also offer a wavefunction simulator, which allows users to contruct and inspect wavefunctions of quantum programs.
You can learn more about the wavefunction simulator :ref:`here <wavefunction_simulator>`.

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
critical objects used in pyQuil for building and executing your quantum programs. We will cover the main methods and attributes
on this page. The `QuantumComputer API Reference <apidocs/quantum_computer.html>`_ provides a reference for all of its methods
and options.

At a high level, the :py:class:`~pyquil.api.QuantumComputer` wraps around our favorite quantum computing tools:

  - **A quantum abstract machine** ``.qam`` : this is our general purpose quantum computing device,
    which implements the required abstract methods described :ref:`above <the_quantum_computer>`. It is implemented as a
    :py:class:`~pyquil.api.QVM` or :py:class:`~pyquil.api.QPU` object in pyQuil.
  - **A compiler** ``.compiler`` : this determines how we manipulate the Quil input to something more efficient when possible,
    and then into a form which our QAM can accept as input.
  - **A quantum processor** ``.quantum_processor`` : this specifies the topology and Instruction Set Architecture (ISA) of
    the targeted processor by listing the supported 1Q and 2Q gates.

When you instantiate a :py:class:`~pyquil.api.QuantumComputer` instance, these subcomponents will be compatible with
each other. So, if you get a ``QPU`` implementation for the ``.qam``, you will have a ``QPUCompiler`` for the
``.compiler``, and your ``.quantum_processor`` will match the processor used by the ``.compiler.``

The :py:class:`~pyquil.api.QuantumComputer` instance makes methods available which are built on the above objects. If
you need more fine grained controls for your work, you might try exploring what is offered by these objects.

For more information on each of the above, check out the following pages:

 - `Compiler API Reference <apidocs/compilers.html>`_
 - :ref:`Quil Compiler docs <compiler>`
 - `Quantum Processor API Reference <apidocs/quantum_processors.html>`_
 - :ref:`new_topology`
 - `Quantum abstract machine (QAM) API Reference <apidocs/qam.html>`_
 - `The Quil Whitepaper <https://arxiv.org/abs/1608.03355>`_ which describes the QAM

Instantiation
=============

A decent amount of information needs to be provided to initialize the ``compiler``, ``quantum_processor``, and ``qam`` attributes,
much of which is already in your :ref:`config files <advanced_usage>` (or provided reasonable defaults when running locally).
Typically, you will want a :py:class:`~pyquil.api.QuantumComputer` which either:

  - pertains to a real, available QPU
  - is a QVM but mimics the topology of a QPU
  - is some generic QVM

All of this can be accomplished with :py:func:`~pyquil.api.get_qc`.

.. code:: python

    def get_qc(name: str, *, as_qvm: bool = None, noisy: bool = None, ...) -> QuantumComputer:

.. testcode:: instantiation

    from pyquil import get_qc

    QPU_NAME="Aspen-M-3"

    # Get a QPU
    # qc = get_qc(QPU_NAME)  # QPU_NAME is just a string naming the quantum_processor

    # Get a QVM with the same topology as the QPU
    # qc = get_qc(QPU_NAME, as_qvm=True)

    # A fully connected QVM
    number_of_qubits = 10
    qc = get_qc(f"{number_of_qubits}q-qvm")

As a reminder, you will have to join QCS to get access to a specific quantum processor.
Check out our `documentation for QCS <https://docs.rigetti.com>`_ and `join the waitlist <https://www.rigetti.com/>`_ if you don't have access already.

For more information about creating and adding your own noise models, check out :ref:`noise`.

.. note::

    This page just covers the essentials, but you can customize the behavior of compilation, execution and more using the
    various parameters on :py:func:`~pyquil.api.get_qc`, see the API documentation to see everything that is available.

.. note::

    When connecting to a QVM locally (such as with ``get_qc(..., as_qvm=True)``) you'll have to set up the QVM
    in :ref:`server mode <server>`.

Methods
=======

Now that you have your ``qc``, there's a lot you can do with it. Most users will want to use ``compile``, ``run`` very
regularly. The general flow of use would look like this:

.. testcode:: methods

    from pyquil import get_qc, Program
    from pyquil.gates import *

    qc = get_qc('9q-square-qvm')            # not general to any number of qubits, 9q-square-qvm is special

    qubits = qc.qubits()                    # this information comes from qc.quantum_processor
    p = Program()
    # ... build program, potentially making use of the qubits list

    compiled_program = qc.compile(p)        # this makes multiple calls to qc.compiler

    results = qc.run(compiled_program)      # this makes multiple calls to qc.qam

.. note::

    In addition to a running QVM server, you will need a running ``quilc`` server to compile your program. Setting
    up both of these is explained :ref:`here <server>`.

The ``.run(...)`` method
------------------------

When using the ``.run(...)`` method, **you are responsible for compiling your program before running it.**
For example:

.. testcode:: methods

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
    result = qc.run(executable)  # .run takes in a compiled program
    bitstrings = result.get_register_map().get("ro")
    print(bitstrings)

The results returned is a *list of lists of integers*. In the above case, that's

.. testoutput:: methods

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

.. _new_topology:

``.execute`` and ``.get_result``
--------------------------------

The ``.run(...)`` method is itself a convenience wrapper around two other methods: ``.execute(...)`` and
``.get_result(...)``. ``run`` makes your program appear synchronous (request and then wait for the response),
when in reality on some backends (such as a live QPU), execution is in fact asynchronous (request execution,
then request results at a later time). For finer-grained control over your program execution process,
you can use these two methods in place of ``.run``. This is most useful when you want to execute work
concurrently - for that, please see "Advanced Usage."

********************************
Simulating the QPU using the QVM
********************************

The :py:class:`~pyquil.api.QAM` methods are intended to be used in the same way, whether a QVM or QPU is being targeted.
For everywhere on this page, you can swap out the type of the QAM (QVM <=> QPU) and you will still
get reasonable results back. As long as the topologies of the quantum processors are the same, programs compiled and run on the QVM
will be able to run on the QPU and vice versa. Since :py:class:`~pyquil.api.QuantumComputer` is built on the ``QAM``
abstract class, its methods will also work for both QAM implementations.

This makes the QVM a powerful tool for testing quantum programs before executing them on the QPU.

.. code:: python

    QPU_NAME="Aspen-M-3"
    qpu = get_qc(QPU_NAME)
    qvm = get_qc(QPU_NAME, as_qvm=True)

By simply providing ``as_qvm=True``, we get a QVM which will have the same topology as
the named QPU. It's a good idea to run your programs against the QVM before booking QPU time to iron out
bugs. To learn more about how to add noise models to your virtual ``QuantumComputer`` instance, check out
:ref:`noise`.

Differences between a QVM and a QPU based ``QuantumComputer``
=============================================================

As mentioned above, pyQuil is designed such that code based on a ``QuantumComputer`` can be used in more or less the same way,
regardless if it is based on a QVM or QPU. However, depending on which you are using, the subcompoments have additional features
worth knowing about.

For instance, if your code targets a QVM, ``qc.qam`` will be a :py:class:`~pyquil.api.QVM``` instance, and ``qc.compiler`` will
be a :py:class:`~pyquil.api.QVMCompiler` instance. However, if your code targets a QPU, ``qc.qam`` will be a :py:class:`~pyquil.api.QPU` instance, and ``qc.compiler`` will be a :py:class:`~pyquil.api.QPUCompiler` instance.

While these subcomponents follow common interfaces, namely :py:class:`~pyquil.api.QAM` and
:py:class:`~pyquil.api.AbstractCompiler`, there may be some methods or properties that are accessible on the QPU-based instances
but not on the QVM-based instances, and vice versa.

You can access these features and keep your code robust by performing type checks on ``qc.qam`` and/or ``qc.compiler``.
For example, if you wanted to refresh the calibration program, which only applies to QPU-based ``QuantumComputers``, but still
want a script that works for both QVM and QPU targets, you could do the following:

.. testcode:: differences

    from pyquil import get_qc
    from pyquil.api import QPUCompiler

    qc = get_qc("2q-qvm")  # or "Aspen-M-3" 

    if isinstance(qc.compiler, QPUCompiler):
        # Working with a QPU - refresh calibrations
        qc.compiler.get_calibration_program(force_refresh=True)

Providing your own quantum processor topology
=============================================

You can provide your own quantum processor topology by specifying qubits as a number, and which edges exist
between those qubits. Here is an example that uses a subset of the instruction set architecture of a
Rigetti QPU to specify a 16 qubit topology.

.. code:: python

    import networkx as nx
    from pyquil import get_qc
    from pyquil.quantum_processor import NxQuantumProcessor
    from pyquil.noise import decoherence_noise_with_asymmetric_ro
    
    qpu = get_qc("Aspen-M-3")
    isa = qpu.to_compiler_isa()
    qubits = sorted(int(k) for k in isa.qubits.keys())[:16]
    edges = [(q1, q2) for q1 in qubits for q2 in qubits if f"{q1}-{q2}" in isa.edges]

    # Build the NX graph
    topo = nx.from_edgelist(edges)
    # You would uncomment the next line if you have disconnected qubits
    # topo.add_nodes_from(qubits)
    quantum_processor = NxQuantumProcessor(topo)
    quantum_processor.noise_model = decoherence_noise_with_asymmetric_ro(quantum_processor.to_compiler_isa())  # Optional

Now that you have your quantum processor, you could set ``qc.compiler.quantum_processor`` to point to your new quantum processor,
or use it to make new objects.

.. [1] https://arxiv.org/abs/1608.03355
