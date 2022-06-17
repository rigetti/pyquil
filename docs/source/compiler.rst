.. _compiler:

The Quil Compiler
=================

Expectations for Program Contents
---------------------------------

The QPUs have much more limited natural gate sets than the standard gate set offered by pyQuil: on
Rigetti QPUs, the gate operators are constrained to lie in ``RZ(θ)``, ``RX(k*π/2)``, ``CZ`` and
``XY``; and the gates are required to act on physically available hardware (for single-qubit gates,
this means acting only on live qubits, and for qubit-pair gates, this means acting on neighboring
qubits). However, as a programmer, it is often (though not always) desirable to to be able to write
programs which don't take these details into account. This generally leads to more portable code if
one isn't tied to a specific set of gates or QPU architecture. To ameliorate these limitations, the
Rigetti software toolkit contains an optimizing compiler that translates arbitrary Quil to native
Quil and native Quil to executables suitable for Rigetti hardware.


Interacting with the Compiler
-----------------------------

After :ref:`installing the SDK <sdkinstall>`, the Quil Compiler, ``quilc`` is available on your
local machine. You can initialize a local ``quilc`` server by typing ``quilc -R`` into your
terminal. You should see the following message.

.. code:: text

    $ quilc -S
    +-----------------+
    |  W E L C O M E  |
    |   T O   T H E   |
    |  R I G E T T I  |
    |     Q U I L     |
    | C O M P I L E R |
    +-----------------+
    Copyright (c) 2018 Rigetti Computing.

    ... - Launching quilc.
    ... - Spawning server at (tcp://*:5555) .

To get a description of ``quilc`` and its options and examples of command line use, see the quilc `README
<https://github.com/rigetti/quilc>`_ or type ``man quilc`` in your terminal.


A ``QuantumComputer`` object supplied by the function ``pyquil.api.get_qc()`` comes equipped with a
connection to your local Rigetti Quil compiler.  This can be accessed using the instance method ``.compile()``,
as in the following:

.. code:: python

    from pyquil.quil import Pragma, Program
    from pyquil.api import get_qc
    from pyquil.gates import CNOT, H

    qc = get_qc("9q-square-qvm")

    ep = qc.compile(Program(H(0), CNOT(0,1), CNOT(1,2)))

    print(ep)

with output

.. code:: python

    RZ(pi/2) 0
    RX(pi/2) 0
    RZ(-pi/2) 1
    RX(pi/2) 1
    CZ 1 0
    RX(-pi/2) 1
    RZ(-pi/2) 2
    RX(pi/2) 2
    CZ 2 1
    RZ(-pi/2) 0
    RZ(-pi/2) 1
    RX(-pi/2) 2
    RZ(pi/2) 2

The compiler connection is also available directly via the property ``qc.compiler``.  The
precise class of this object changes based on context (e.g., ``QPUCompiler``,
``QVMCompiler``), but it always conforms to the interface laid out by ``AbstractCompiler``:

* ``compiler.quil_to_native_quil(program, *, protoquil)``: This method converts a Quil program into
  native Quil, according to the ISA that the compiler is initialized with.  The input parameter is
  specified as a :py:class:`~pyquil.quil.Program` object. The optional ``protoquil`` keyword
  argument instructs the compiler to restrict both its input and output to protoquil (Quil code that
  can be executed on a QPU). If the server is started with the ``-P`` option, or you specify
  ``protoquil=True`` the returned ``Program`` object will be equipped with a ``.metadata`` property
  that gives extraneous information about the compilation output (e.g., gate depth, as well as many
  others).  This call blocks until Quil compilation finishes.
* ``compiler.native_quil_to_executable(nq_program)``: This method converts a native Quil program, which
  is promised to consist only of native gates for a given ISA, into an executable suitable for
  submission to one of a QVM or a QPU.  This call blocks until the executable is generated.

The instance method ``qc.compile`` described above is a combination of these two methods: first the
incoming Quil is nativized, and then that is immediately turned into an executable.  Accordingly,
the previous example snippet is identical to the following:

.. code:: python

    from pyquil.quil import Pragma, Program
    from pyquil.api import get_qc
    from pyquil.gates import CNOT, H

    qc = get_qc("9q-square-qvm")

    p = Program(H(0), CNOT(0,1), CNOT(1,2))

    np = qc.compiler.quil_to_native_quil(p, protoquil=True)
    print(np.metadata)

    ep = qc.compiler.native_quil_to_executable(np)
    print(ep)

Timeouts
--------

If your circuit is sufficiently complex the compiler may require more time than is permitted by
default. To change this timeout, use the `compiler_timeout` option on `get_qc`:

.. code:: python

    qc = get_qc(..., compiler_timeout=100) # 100 seconds

Legal compiler input
--------------------

The QPU is not able to execute all possible Quil programs. At present, a Quil program qualifies for
execution if has the following form:

* The program may begin with a ``RESET`` instruction. (If provided, the QPU will actively reset the
  state of the quantum device to the ground state before program execution. If omitted, the QPU will
  wait for a relaxation period to pass before program execution instead.)
* This is then followed by a block of native quantum gates. A gate is native if it is of the form
  ``RZ(θ)`` for any value ``θ``, ``RX(k*π/2)`` for an integer ``k``, or ``CZ q0 q1`` for ``q0``,
  ``q1`` a pair of qubits participating in a qubit-qubit interaction. Some devices provide the
  ``XY(θ) q0 q1`` two qubit gate.
* This is then followed by a block of ``MEASURE`` instructions.

To instruct the compiler to produce Quil code that can be executed on a QPU, you can use the
``protoquil`` keyword in a call to ``compiler.quil_to_native_quil(program, protoquil=True)`` or
``qc.compile(program, protoquil=True)``.

.. note::

   If your compiler server is started with the protoquil option ``-P`` (as is the case for your
   JupyterLab notebook's compiler) then specifying ``protoquil=False`` will override the server
   and forcefully disable protoquil. Specifying ``protoquil=None`` defers to the server's choice.

Compilation metadata
--------------------

When your compiler is started with the ``-P`` option, the ``compiler.quil_to_native_quil()`` method
will return both the compiled program and a dictionary of statistics for the compiled program. This
dictionary contains the keys

- ``final_rewiring``: see section below on rewirings.
- ``gate_depth``: the longest subsequence of compiled instructions where adjacent instructions
  share resources.
- ``multiqubit_gate_depth``: like ``gate_depth`` but restricted to multi-qubit gates.
- ``gate_volume``: total number of gates in the compiled program.
- ``program_duration``: program duration with parallel executation of gates (using hard-coded values
  of individual gate durations).
- ``qpu_runtime_estimation``: estimated runtime on a Rigetti QPU (in milliseconds). This is
  extrapolated from a single shot of a 16Q program with final measurements on all 16 qubits. If you
  are running a parametric program then you should estimate the total runtime as ``size of parameter
  space * estimated runtime of single shot``. This should be treated only as an approximation.
- ``program_fidelity``: the estimated fidelity of the compiled program.
- ``topological_swaps``: the number of topological swaps incurred during compilation of the program.

For example, to inspect the ``qpu_runtime_estimation`` you might do the following:

.. code:: python

    from pyquil import get_qc, Program

    # If you have a reserved QPU, use it here
    qc = get_qc("Aspen-X")
    # Otherwise use a QVM
    # qc = get_qc("8q-qvm")

    # Likely you will have a more complex program:
    p = Program("RX(pi) 0")

    native_p = qc.compiler.quil_to_native_quil(p)

    # The program will now have only native gates
    print(native_p)
    # And also a dictionary, with the above keys
    print(native_p.native_quil_metadata["qpu_runtime_estimation"])

.. _pragma:

Region-specific compiler features through PRAGMA
------------------------------------------------

The Quil compiler can also be communicated with through ``PRAGMA`` commands embedded in the Quil
program.

.. note::

    The interface to the Quil compiler from pyQuil is under construction, and some of the ``PRAGMA`` directives will soon be replaced by finer-grained method calls.


Preserved regions
~~~~~~~~~~~~~~~~~

The compiler can be circumvented in user-specified regions. The start of such a region is denoted by
``PRAGMA PRESERVE_BLOCK``, and the end is denoted by ``PRAGMA END_PRESERVE_BLOCK``.  The Quil
compiler promises not to modify any instructions contained in such a region.

.. warning::
   If a preserved block is not legal QPU input, then it is not guaranteed to execute or it may produced unexpected results.

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

.. warning::
   Lying to the compiler about what blocks can commute can cause incorrect results.

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

.. _compiler_rewirings:

Rewirings
~~~~~~~~~

When a Quil program contains multi-qubit instructions that do not name qubit-qubit links present on a
target device, the compiler will rearrange the qubits so that execution becomes possible.  In order to
help the user understand what rearrangement may have been done, the compiler emits comments at various
points in the raw Quil code (which is not currently visible from a pyQuil ``Program`` object's ``.out()``
method): ``# Entering rewiring`` and ``# Exiting rewiring``.  From the perspective of the user, both
comments serve the same purpose: ``# Entering rewiring: #(n0 n1 ... nk)`` indicates that the logical
qubit labeled ``j`` in the program has been assigned to lie on the physical qubit labeled ``nj`` on
the device.  This is strictly for human-readability: these comments are discarded and have no effect.

.. _swaps:

SWAPs
*****

When the compiler needs to move an instruction's qubits closer it will insert ``SWAP`` gates which
can be costly. If, however, the swaps are inserted at the very beginning of the program, the
compiler can treat them as `virtual` swaps which do not appear in the resulting program but instead
affect the initial rewiring of the program.

For example, consider running a ``CZ`` on non-neighboring qubits on a linear device:

.. code:: python

   import networkx as nx
   from pyquil import Program, get_qc
   from pyquil.api._quantum_computer import _get_qvm_with_topology
   from pyquil.gates import CZ

   graph = nx.from_edgelist([(0, 1), (1, 2)])
   qc = _get_qvm_with_topology(name="line", topology=graph)

   p = Program(CZ(0, 2))
   print(qc.compile(p))

   CZ 2 1

We see that the resulting program has only a single ``CZ`` even though the original program would
usually require the insertion of a ``SWAP`` gate. The compiler instead opted to just relabel (or
rewire) the qubits, thus not inflating the number of gates in the result.

For larger and more complex programs (with more entanglement) it may not always be possible to avoid
inserting swaps. For example, the following program requires a ``SWAP`` that increases its gate depth:

.. code:: python

   import networkx as nx
   from pyquil import Program, get_qc
   from pyquil.api._quantum_computer import _get_qvm_with_topology
   from pyquil.gates import H, CZ

   graph = nx.from_edgelist([(0, 1), (1, 2)])
   qc = _get_qvm_with_topology(name="line", topology=graph)

   p = Program(CZ(0, 1), H(0), CZ(1, 2), CZ(0, 2))
   print(qc.compile(p))

   CZ 2 1
   RX(-pi/2) 2
   RX(-pi/2) 2
   CZ 2 1
   CZ 1 0
   RZ(pi/2) 1
   RX(-pi/2) 1
   RX(-pi/2) 1
   RX(-pi/2) 2
   RX(pi/2) 2
   XY(pi) 2 1
   RX(-pi/2) 1
   CZ 1 0
   RZ(pi/2) 1
   RX(pi/2) 2
   RX(pi/2) 2

.. note::

   ``SWAP`` gates generally cost three ``CZ`` gates or three ``XY`` gates. However, if your device
   has `both` ``CZ`` and ``XY`` gates available, then the compiler can produce a ``SWAP`` gate that
   uses only `two` two-qubit gates (one ``CZ`` and one ``XY``).

Initial rewiring
****************

In addition, you have some control over how the compiler constructs its
rewiring, which is controlled by ``PRAGMA INITIAL_REWIRING``. The syntax is as follows.

.. code:: python
   
   # <type> can be NAIVE, RANDOM, PARTIAL, or GREEDY
   #
   # The double quotes are required.
   PRAGMA INITIAL_REWIRING "<type>"

Including this `before any non-pragmas` will allow the compiler to alter its rewiring
behavior.

The default initial rewiring strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Each initial rewiring strategy is described in more detail after the discussion about defaults.

When no ``INITIAL_REWIRING`` pragma is provided the compiler will choose one of two options
depending on the program:

+ ``NAIVE``: The qubits used in all instructions in the program satisfy the topological constraints of the device.

+ ``PARTIAL``: Otherwise.

For example, if your program consists of two-qubit instructions where the qubits in each instruction are nearest neighbors on the device, the compiler will employ the native strategy:

.. code:: python

   from pyquil import Program, get_qc
   from pyquil.gates import CZ

   qc = get_qc("Aspen-X", as_qvm=True)
   p = Program(CZ(3, 4))

   print(qc.compile(p))

   CZ 3 4

In the above example, `CZ 3 4` touches qubits that are already nearest neighbors (and support a
`CZ` instruction) and so the compiler employs the naive strategy (and thus does not rewire those
qubits to use better ones).

If however, the program uses qubits that `must` be rewired, then the compiler defaults to the
partial strategy:

.. code:: python

   from pyquil import Program, get_qc
   from pyquil.gates import CZ

   qc = get_qc("Aspen-X", as_qvm=True)
   p = Program(CZ(3, 4))

   print(qc.compile(p))

   RZ(-pi/2) 0
   RX(pi/2) 0
   RZ(-pi/2) 0
   RZ(pi/2) 1
   XY(pi) 1 0
   RZ(pi/2) 1
   RX(pi/2) 1
   RZ(-pi/2) 1
   XY(pi) 1 0
   RZ(-pi/2) 0
   RX(-pi/2) 0

.. _naive_rewiring:

NAIVE
^^^^^

In this mode, the compiler chooses the ``naive`` mapping between logical qubits and physical qubits,
where logical qubit ``i`` is assigned to physical qubit ``i``. With this initial rewiring, the
compiler will generally **not** move an instruction's qubits around even if it results in a poor
execution fidelity. For example assume that ``Aspen-X`` has a low-fidelity ``CZ 0 1``, then
compiling this program with naive rewiring will **not** move the ``CZ`` to a better qubit pair:

.. code:: python

   from pyquil import Program, get_qc
   from pyquil.gates import CZ

   qc = get_qc("Aspen-X", as_qvm=True)
   p = Program('PRAGMA INITIAL_REWIRING "NAIVE"', CZ(0, 1))

   print(qc.compile(p))

   PRAGMA INITIAL_REWIRING "NAIVE"
   CZ 0 1

If, however, your program includes an instruction that does **not** use neighboring qubits the
compiler will be required to insert swaps (virtual or real, see swaps_) that might affect the
logical-physical qubit mapping. For example,

.. code:: python

   from pyquil import Program, get_qc
   from pyquil.gates import CZ

   qc = get_qc("Aspen-X", as_qvm=True)
   p = Program('PRAGMA INITIAL_REWIRING "NAIVE"', CZ(0, 2))

   print(qc.compile(p))

   PRAGMA INITIAL_REWIRING "NAIVE"
   CZ 6 5

In the above program ``CZ 0 2`` is not a native instruction (meaning it cannot be directly executed
on the target device) and so the compiler must insert a swap (virtual, in this case) into the
program. When rewiring must occur in this mode it is **not** guaranteed that the resulting program
will have optimal fidelity.

.. _partial_rewiring:

PARTIAL
^^^^^^^

In this mode, the compiler begins with an empty mapping from logical qubits to physical
qubits. During the progression of compilation this mapping will be filled-in, and thus at any point
the mapping is said to be `partial`. Generally this gives the compiler the opportunity to assign a
logical-to-physical qubit mapping that optimizes the fidelity of the resulting program by
incorporating fidelity information about any qubit in the device ISA.

For example, if the instruction ``CZ 0 1`` has poor fidelity, under the partial rewiring strategy
the compiler can find an alternative that improves the program fidelity:

.. code:: python

   from pyquil import Program, get_qc
   from pyquil.gates import CZ

   qc = get_qc("Aspen-X", as_qvm=True)
   p = Program('PRAGMA INITIAL_REWIRING "PARTIAL"', CZ(0, 1))

   print(qc.compile(p))

   PRAGMA INITIAL_REWIRING "PARTIAL"
   CZ 20 27

Here the compiler sees that the instruction ``CZ 20 27`` will produce a program with better fidelity
and so opts to reassign qubits in the original program.

.. _greedy_rewiring:

GREEDY
^^^^^^

In this mode, the compiler chooses an initial mapping between logical and physical qubits based upon
a greedy optimization of the `distances` between qubits used in the program and those available on
the device. When compared to the ``PARTIAL`` strategy it is generally more efficient because it uses
a simple heuristic; however, it will also produce a program with worse overall fidelity. If
compilation feels too slow and you're willing to trade fidelity for compilation speed, then you may
see success with this strategy.

Which strategy should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally, as quantum software engineers, we want to maximize the execution fidelity of our
programs. In other cases, however, for example in QCVV, we want to have more control about where
instructions are placed.

.. list-table:: Choosing an initial rewiring strategy
   :widths: 70 30
   :header-rows: 1

   * - Desired effect
     - Recommended initial rewiring strategy
   * - Maximize program execution fidelity
     - ``PARTIAL``
   * - Preserve, where possible, the qubits used in the input program
     - ``NAIVE``
   * - Faster qubit allocation at expense of fidelity
     - ``GREEDY``

Note that each of these have drawbacks described in the sections above.

Common Error Messages
---------------------

The compiler itself is subject to some limitations, and some of the more commonly observed errors
follow:

+ ``! ! ! Error: Matrices do not lie in the same projective class.`` The compiler attempted to
  decompose an operator as native Quil instructions, and the resulting instructions do not match the
  original operator. This can happen when the original operator is not a unitary matrix, and could
  indicate an invalid ``DEFGATE`` block. In some rare circumstances, it can also happen due to
  floating point precision issues. In the latter case, the issue is resolved simply by recompiling
  the program. If you issue cannot be solved, please contact support@rigetti.com or post an issue to
  `the github project page. <https://github.com/rigetti/quilc/issues>`_
