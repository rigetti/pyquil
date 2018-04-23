.. _compiler:

Using the Quil Compiler
=======================


Expectations for Program Contents
---------------------------------

The QPUs have much more limited natural gate sets than the standard gate set offered by pyQuil: the
gate operators are constrained to lie in ``RZ(θ)``, ``RX(kπ/2)``, ``CZ``, and ``CPHASE(θ)``; and the
gates are required to act on physically available hardware (for single-qubit gates, this means
acting only on live qubits, and for qubit-pair gates, this means acting on neighboring qubits).

To ameliorate these limitations, the QPU execution stack contains an optimizing compiler that
translates arbitrary ProtoQuil to QPU-executable Quil.  The compiler is designed to avoid changing
even non-semantic details of input Quil code, except to make it shorter when possible.  For
instance, it will not readdress Quil code that is already appropriately addressed to physically
realizable hardware objects on the QPU.  The following figure illustrates the layout and addressing
of the Rigetti 19Q-Acorn QPU.

.. figure:: images/acorn_connectivity.png
    :width: 540px
    :align: center
    :height: 300px
    :alt: 19Q connectivity
    :figclass: align-center

    Qubit adjacency schematic for the Rigetti 19Q-Acorn QPU.
    In particular, notice that qubit 3 is disabled.


Interacting with the Compiler
-----------------------------

The ``QVMConnection`` and ``QPUConnection`` classes in pyQuil offer indirect support for interacting
with the compiler: they are both capable of submitting jobs to the compiler for preprocessing before
the job is forwarded to the execution target.  This behavior is disabled by default for the QVM and
enabled by default for the QPU.  PyQuil also offers the ``CompilerConnection`` class for direct
access to the compiler, which returns compiled Program jobs to the user without executing them.
``CompilerConnection`` can be used to learn about the properties of the program,
like gate volume, single qubit gate depth, topological swaps, program fidelity and multiqubit gate
depth. In all cases, the user's Forest plan must have compiler access enabled to use these features.

Here’s an example of using ``CompilerConnection`` to compile a program that targets the 19Q-Acorn
QPU, separately from sending a program to the QPU/QVM.

.. code:: python

    from pyquil.api import CompilerConnection, get_devices
    from pyquil.quil import Pragma, Program
    from pyquil.gates import CNOT, H

    devices = get_devices(as_dict=True)
    acorn = devices['19Q-Acorn']
    compiler = CompilerConnection(acorn)

    job_id = compiler.compile_async(Program(H(0), CNOT(0,1), CNOT(1,2)))
    job = compiler.wait_for_job(job_id)

    print('compiled quil', job.compiled_quil())
    print('gate volume', job.gate_volume())
    print('gate depth', job.gate_depth())
    print('topological swaps', job.topological_swaps())
    print('program fidelity', job.program_fidelity())
    print('multiqubit gate depth', job.multiqubit_gate_depth())


Here's what you should see:

.. code:: python


    PRAGMA EXPECTED_REWIRING "#(6 1 7 3 4 5 0 2 8 9 10 11 12 13 14 15 16 17 18 19)"
    RZ(-pi/2) 1
    RX(pi/2) 1
    RZ(-pi/2) 6
    RX(-pi/2) 6
    CZ 1 6
    RX(-pi/2) 1
    RZ(-pi/2) 7
    RX(pi/2) 7
    CZ 7 1
    RZ(-pi/2) 1
    RZ(pi/2) 6
    RX(-pi/2) 7
    RZ(pi/2) 7
    PRAGMA CURRENT_REWIRING "#(6 1 7 3 4 5 0 2 8 9 10 11 12 13 14 15 16 17 18 19)"
    PRAGMA EXPECTED_REWIRING "#(6 1 7 3 4 5 0 2 8 9 10 11 12 13 14 15 16 17 18 19)"
    PRAGMA CURRENT_REWIRING "#(6 1 7 3 4 5 0 2 8 9 10 11 12 13 14 15 16 17 18 19)"

    gate volume 13
    gate depth 7
    topological swaps 0
    program fidelity 0.872503399848938
    multiqubit gate depth 2


The ``QVMConnection`` and ``QPUConnection`` objects have their compiler interactions set up in the
same way: the ``.run`` and ``.run_and_measure`` methods take the optional arguments
``needs_compilation`` and ``isa`` that respectively toggle the compilation preprocessing step and
provide the compiler with a target instruction set architecture, specified as a pyQuil ``ISA``
object. The compiler can be bypassed by passing the method parameter ``needs_compilation=False``.
If the ``isa`` named argument is not set, then the ``default_isa`` property on the
connection object is used instead. The compiled program can be accessed after a job has been
submitted to the QPU by using the ``.compiled_quil()`` accessor method on the resulting ``Job``
object instance.

The Quil compiler can also be communicated with through ``PRAGMA`` commands embedded in the Quil
program.

    + It can be circumvented in user-specified regions. The start of such a region is denoted by
      ``PRAGMA PRESERVE_BLOCK``, and the end is denoted by ``PRAGMA END_PRESERVE_BLOCK``.
      The Quil compiler promises not to modify any instructions contained in such a region.
    + It can sometimes arrange gate sequences more cleverly if the user gives it hints about
      sequences of gates that commute.  A region containing commuting sequences is bookended by
      ``PRAGMA COMMUTING_BLOCKS`` and ``PRAGMA END_COMMUTING_BLOCKS``; within such a region, a
      given commuting sequence is bookended by ``PRAGMA BLOCK`` and ``PRAGMA END_BLOCK``.
      The following snippet demonstrates this hinting syntax:

.. code:: python

    PRAGMA COMMUTING_BLOCKS
    PRAGMA BLOCK
    CZ 0 1
    PRAGMA END_BLOCK
    PRAGMA BLOCK
    CZ 1 2
    PRAGMA END_BLOCK
    PRAGMA BLOCK
    CZ 0 2
    PRAGMA END_BLOCK
    PRAGMA END_COMMUTING_BLOCKS


Common Error Messages
---------------------

The compiler itself is subject to some limitations, and some of the more commonly observed errors
follow:

+ ``! ! ! Error: Failed to select a SWAP instruction. Perhaps the qubit graph is disconnected?``
  This error indicates a readdressing failure: some non-native Quil could not be reassigned to lie
  on native devices.  Two common reasons for this failure are:

    + It is possible for the readdressing problem to be too difficult for the compiler to sort out,
      causing deadlock.
    + If a qubit-qubit gate is requested to act on two qubit resources that lie on disconnected
      regions of the qubit graph, the addresser will fail.

+ ``! ! ! Error: Matrices do not lie in the same projective class.`` The compiler attempted to
  decompose an operator as native Quil instructions, and the resulting instructions do not match the
  original operator.  This can happen when the original operator is not a unitary matrix, and could
  indicate an invalid ``DEFGATE`` block.
+ ``! ! ! Error: Addresser loop only supports pure quantum instructions.`` The compiler inspected an
  instruction that it does not understand.  The most common cause of this error is the inclusion of
  classical control in a program submission, which is legal Quil but falls outside of the
  domain of ProtoQuil.

