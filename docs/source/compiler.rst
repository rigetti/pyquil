.. _compiler:

The Quil Compiler
=================

Expectations for Program Contents
---------------------------------

The QPUs have much more limited natural gate sets than the standard gate set offered by pyQuil: the
gate operators are constrained to lie in ``RZ(θ)``, ``RX(kπ/2)``, and ``CZ``; and the
gates are required to act on physically available hardware (for single-qubit gates, this means
acting only on live qubits, and for qubit-pair gates, this means acting on neighboring qubits).
To ameliorate these limitations, the Rigetti software toolkit contains an optimizing compiler that
translates arbitrary Quil to native Quil and native ProtoQuil to executables suitable for Rigetti
hardware.


Interacting with the Compiler
-----------------------------

A ``QuantumComputer`` object supplied by the function ``pyquil.api.get_qc()`` comes equipped with a
connection to a Rigetti quantum compiler.  If the object is called ``qc``, then this can be accessed
using the instance method ``.compile()``, as in the following:

.. code:: python

    from pyquil.quil import Pragma, Program
    from pyquil.api import get_qc
    from pyquil.gates import CNOT, H

    qc = get_qc("9q-generic-qvm")

    ep = qc.compile(Program(H(0), CNOT(0,1), CNOT(1,2)))

    print(ep.program) # here ep is of type PyquilExecutableResponse, which is not always inspectable

with output

.. code:: python

    PRAGMA EXPECTED_REWIRING "#(7 8 5 0 1 2 3 4 6)"
    RZ(pi/2) 7
    RX(pi/2) 7
    RZ(-pi/2) 8
    RX(pi/2) 8
    CZ 8 7
    RZ(-pi/2) 5
    RX(pi/2) 5
    RX(-pi/2) 8
    CZ 5 8
    RX(-pi/2) 5
    RZ(pi/2) 5
    RZ(-pi/2) 7
    RZ(-pi/2) 8
    PRAGMA CURRENT_REWIRING "#(7 8 5 0 1 2 3 4 6)"
    PRAGMA EXPECTED_REWIRING "#(7 8 5 0 1 2 3 4 6)"
    PRAGMA CURRENT_REWIRING "#(7 8 5 0 1 2 3 4 6)"

The compiler connection is also available directly via the property ``qc.compiler``.  The precise
class of this object changes based on context (e.g., ``QPUCompiler``, ``QVMCompiler``, or
``LocalQVMCompiler``), but it always conforms to the interface laid out by ``pyquil.api._qac``:

* ``compiler.quil_to_native_quil(program)``: This method converts a Quil program into native Quil,
  according to the ISA that the compiler is initialized with.  The input parameter is specified as a
  ``Program`` object, and the output is given as a new ``Program`` object, equipped with a
  ``.metadata`` property that gives extraneous information about the compilation output (e.g., gate
  depth, as well as many others).  This call blocks until Quil compilation finishes.
* ``compiler.native_quil_to_executable(nq_program)``: This method converts a ProtoQuil program, which
  is promised to consist only of native gates for a given ISA, into an executable suitable for
  submission to one of a QVM or a QPU.  This call blocks until the executable is generated.

The instance method ``qc.compile`` described above is a combination of these two methods: first the
incoming Quil is nativized, and then that is immediately turned into an executable.  Accordingly,
the previous example snippet is identical to the following:

.. code:: python

    from pyquil.quil import Pragma, Program
    from pyquil.api import get_qc
    from pyquil.gates import CNOT, H

    qc = get_qc("9q-generic-qvm")

    p = Program(H(0), CNOT(0,1), CNOT(1,2))
    np = qc.compiler.quil_to_native_quil(p)
    ep = qc.compiler.native_quil_to_executable(np)

    print(ep.program) # here ep is of type PyquilExecutableResponse, which is not always inspectable


Legal compiler input
--------------------

The QPU is not able to execute all possible Quil programs, and so Quil bound for execution on a QPU
must conform to the "ProtoQuil" standard.  At present, a Quil program qualifies as ProtoQuil if it
has the following form:

* The program may or may not begin with a ``RESET`` instruction.  (If provided, the QPU will actively
  reset the state of the quantum device to the ground state before program execution.  If omitted,
  the QPU will wait for a relaxation period to pass before program execution instead.)
* This is then followed by a block of native quantum gates.  A gate is native if it is of the form
  ``RZ(θ)`` for any value ``θ``, ``RX(kπ/2)`` for an integer ``k``, or ``CZ q0 q1`` for ``q0``, ``q1``
  a pair of qubits participating in a qubit-qubit interaction.
* This is then followed by a block of ``MEASURE`` instructions.


Region-specific compiler features through PRAGMA
------------------------------------------------

The Quil compiler can also be communicated with through ``PRAGMA`` commands embedded in the Quil
program.

.. note::

    The pyQuil compiler interface is under construction, and some of the ``PRAGMA`` directives will
    soon be replaced by finer-grained method calls.


Preserved regions
~~~~~~~~~~~~~~~~~

The compiler can be circumvented in user-specified regions. The start of such a region is denoted by
``PRAGMA PRESERVE_BLOCK``, and the end is denoted by ``PRAGMA END_PRESERVE_BLOCK``.  The Quil
compiler promises not to modify any instructions contained in such a region.

The following is an example of a program that prepares a Bell state on qubits 0 and 1, then performs
a time delay to invite noisy system interaction before measuring the qubits.  The time delay region
is marked by ``PRAGMA PRESERVE_BLOCK`` and ``PRAGMA END_PRESERVE_BLOCK``; without these delimiters,
the compiler will remove the identity gates that serve to provide the time delay.  However, the
regions outside of the ``PRAGMA`` region will still be compiled, converting the Bell state preparation
to the native gate set.

.. code:: python

    DECLARE ro BIT[2]

    #   prepare a Bell state
    H 0
    CNOT 0 1

    #   wait a while
    PRAGMA PRESERVE_BLOCK
    I 0
    I 1
    I 0
    I 1
    # ...
    I 0
    I 1
    PRAGMA END_PRESERVE_BLOCK

    #   and read out the results
    MEASURE 0 ro[0]
    MEASURE 1 ro[1]

Parallelizable regions
~~~~~~~~~~~~~~~~~~~~~~

The compiler can sometimes arrange gate sequences more cleverly if the user gives it hints about
sequences of gates that commute.  A region containing commuting sequences is bookended by
``PRAGMA COMMUTING_BLOCKS`` and ``PRAGMA END_COMMUTING_BLOCKS``; within such a region, a given
commuting sequence is bookended by ``PRAGMA BLOCK`` and ``PRAGMA END_BLOCK``.

The following snippet demonstrates this hinting syntax in a context typical of VQE-type algorithms:
after a first stage of performing some state preparation on individual qubits, there is a second
stage of "mixing operations" that both re-use qubit resources and mutually commute, followed by a
final rotation and measurement.  The following program is naturally laid out on a ring with vertices
(read either clockwise or counterclockwise) as 0, 1, 2, 3.  After scheduling the first round of
preparation gates, the compiler will use the hinting to schedule the first and third blocks (which
utilize qubit pairs 0-1 and 2-3) before the second and fourth blocks (which utilize qubit pairs 1-2
and 0-3), resulting in a reduction in circuit depth by one half.  Without hinting, the compiler will
instead execute the blocks in their written order.

.. code:: python

    DECLARE ro BIT[4]

    # Stage one
    H 0
    H 1
    H 2
    H 3

    # Stage two
    PRAGMA COMMUTING_BLOCKS
    PRAGMA BLOCK
    CNOT 0 1
    RZ(0.4) 1
    CNOT 0 1
    PRAGMA END_BLOCK
    PRAGMA BLOCK
    CNOT 1 2
    RZ(0.6) 2
    CNOT 1 2
    PRAGMA END_BLOCK
    PRAGMA BLOCK
    CNOT 2 3
    RZ(0.8) 3
    CNOT 2 3
    PRAGMA END_BLOCK
    PRAGMA BLOCK
    CNOT 0 3
    RZ(0.9) 3
    CNOT 0 3
    PRAGMA END_BLOCK
    PRAGMA END_COMMUTING_BLOCKS

    # Stage three
    H 0
    H 1
    H 2
    H 3

    MEASURE 0 ro[0]
    MEASURE 1 ro[1]
    MEASURE 2 ro[2]
    MEASURE 3 ro[3]


Rewirings
~~~~~~~~~

When a Quil program contains multi-qubit instructions that do not name qubit-qubit links present on a
target device, the compiler will rearrange the qubits so that execution becomes possible.  In order to
help the user understand what rearrangement may have been done, the compiler emits two forms of
``PRAGMA``: ``PRAGMA EXPECTED_REWIRING`` and ``PRAGMA CURRENT_REWIRING``.  From the perspective of the
user, both ``PRAGMA`` instructions serve the same purpose: ``PRAGMA ..._REWIRING "#(n0 n1 ... nk)"``
indicates that the logical qubit labeled ``j`` in the program has been assigned to lie on the physical
qubit labeled ``nj`` on the device.  This is strictly for human-readability: user-supplied instructions
of the form ``PRAGMA [EXPECTED|CURRENT]_REWIRING`` are discarded and have no effect.

In addition, you have some control over how the compiler constructs its
rewiring. If you include a
``PRAGMA INITIAL_REWIRING "[NAIVE|RANDOM|PARTIAL|GREEDY]"``
instruction before any non-pragmas, the compiler will alter its rewiring
behavior.

+ `PARTIAL` (default): The compiler will start with nothing assigned to each
  physical qubit. Then, it will fill in the logical-to-physical mapping as it
  encounters new qubits in the program, making its best guess for where they
  should be placed.
+ `NAIVE`: The compiler will start with an identity mapping as the initial
  rewiring.  In particular, qubits will **not** be rewired unless the program
  requests a qubit-qubit interaction not natively available on the QPU.
+ `RANDOM`: the compiler will start with a random permutation
+ `GREEDY`: the compiler will make a guess for the initial rewiring based on a
  quick initial scan of the entire program.

Common Error Messages
---------------------

The compiler itself is subject to some limitations, and some of the more commonly observed errors
follow:

+ ``! ! ! Error: Matrices do not lie in the same projective class.`` The compiler attempted to
  decompose an operator as native Quil instructions, and the resulting instructions do not match the
  original operator.  This can happen when the original operator is not a unitary matrix, and could
  indicate an invalid ``DEFGATE`` block.
