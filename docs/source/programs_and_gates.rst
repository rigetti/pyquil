.. _basics:

==================
Programs and gates
==================

.. note::

    If you're running locally, remember set up the QVM and quilc in server mode before trying to use
    them: :ref:`server`.

************
Introduction
************

Quantum programs are written in pyQuil by using the :py:class:`~pyquil.quil.Program` object. This ``Program`` abstraction will help us
compose `Quil programs <https://arxiv.org/abs/1608.03355>`_.

.. testcode:: intro

    from pyquil import Program

Programs are constructed by adding quantum gates to it, which are defined in the ``gates`` module. We can import all
standard gates with the following:

.. testcode:: intro

     from pyquil.gates import *

Let's instantiate a ``Program`` and add an operation to it. We will act an ``X`` gate on qubit 0.

.. testcode:: intro

    p = Program()
    p += X(0)

All qubits begin in the ground state. This means that if we measure a qubit without applying operations on it, we expect to receive
a 0 result. The ``X`` gate will rotate qubit 0 from the ground state to the excited state, so a measurement immediately
after should return a 1 result.

We can print our pyQuil program to see the equivalent Quil representation:

.. testcode:: intro

   print(p)

.. testoutput:: intro

    X 0

This isn't going to be very useful to us without measurements. To declare memory and write measurement readout data into
it, write:

.. testcode:: intro

    from pyquil import Program
    from pyquil.gates import *

    p = Program()
    ro = p.declare('ro', 'BIT', 1)
    p += X(0)
    p += MEASURE(0, ro[0])

    print(p)

.. testoutput:: intro

    DECLARE ro BIT[1]
    X 0
    MEASURE 0 ro[0]

We've instantiated a program, declared a memory space named ``ro`` with one single bit of memory, applied
an ``X`` gate on qubit 0, and finally measured qubit 0 into the zeroth index of the memory space named ``ro``.

Awesome! That's all we need to get results back. Now we can actually see what happens if we run this
program on the Quantum Virtual Machine (QVM). We just have to add a few lines to do this.

.. testcode:: intro

    from pyquil import get_qc

    ...

    qc = get_qc('1q-qvm')  # You can make any 'nq-qvm' this way for any reasonable 'n'
    executable = qc.compile(p)
    result = qc.run(executable)
    bitstrings = result.get_register_map().get('ro')
    print(bitstrings)

Congratulations! You just ran your program on the QVM. The returned value should be:

.. testoutput:: intro

    [[1]]

For more information on what the above result means, and on executing quantum programs on the QVM in
general, see :ref:`the_quantum_computer`. The remainder of this section of the docs will be dedicated to constructing
programs in detail, an essential part of becoming fluent in quantum programming.


.. _standard:

*********************
The standard gate set
*********************

The :py:mod:`pyquil.gates` module defines many of the standard gates you would expect. See the module documentation for
everything available, but here's a quick list to get you started:

-  Pauli gates ``I``, ``X``, ``Y``, ``Z``

-  Hadamard gate: ``H``

-  Phase gates: ``PHASE(theta)``, ``S``, ``T``

-  Controlled phase gates: ``CZ``, ``XY``, ``CPHASE00(alpha)``,
   ``CPHASE01(alpha)``, ``CPHASE10(alpha)``, ``CPHASE(alpha)``

-  Cartesian rotation gates: ``RX(theta)``, ``RY(theta)``, ``RZ(theta)``

-  Controlled :math:`X` gates: ``CNOT``, ``CCNOT``

-  Swap gates: ``SWAP``, ``CSWAP``, ``ISWAP``, ``PSWAP(alpha)``

The parameterized gates take a real or complex floating point
number as an argument.

.. _declaring_memory:

****************
Declaring memory
****************

Classical memory regions must be explicitly requested and named in a Quil program by using the ``DECLARE`` directive.
Details about this directive can be found in :py:func:`pyquil.quil.Program.declare`.

In pyQuil, we declare memory with the ``.declare`` method on a ``Program``. Let's inspect the function signature

.. code:: python

    # pyquil.quil.Program

    def declare(self, name, memory_type='BIT', memory_size=1, shared_region=None, offsets=None):


and break down each argument:

 -  ``name`` is any name you want to give this memory region.
 -  ``memory_type`` is one of ``'REAL'``, ``'BIT'``, ``'OCTET'``, or ``'INTEGER'`` (given as a string). Only ``BIT`` and
    ``OCTET`` always have a determined size, which is 1 bit and 8 bits respectively.
 -  ``memory_size`` is the number of elements of that type to reserve.
 -  ``shared_region`` and ``offsets`` allow you to alias memory regions. For example,
    you might want to name the third bit in your readout array as ``q3_ro``.

Now we can get into an example.

.. testcode:: declaring_memory

    from pyquil import Program

    p = Program()
    ro = p.declare('ro', 'BIT', 16)
    theta = p.declare('theta', 'REAL')

    print(p)

.. warning::
    ``.declare`` can't be chained, since it doesn't return a modified ``Program`` object.

Notice that the ``.declare`` method returns a reference to the memory we've just declared. We will need this reference
to make use of these memory spaces again. Let's see how the Quil is looking so far:

.. testoutput:: declaring_memory

    DECLARE ro BIT[16]
    DECLARE theta REAL[1]


That's all we need to do to declare the memory. Continue to the next section on :ref:`measurement` to learn more about
using ``ro`` to store measured readout results. Check out :ref:`parametric_compilation` to see how you might use
``theta`` to compile gate parameters dynamically.

.. _measurement:

***********
Measurement
***********

We can use ``MEASURE`` instructions to measure particular qubits in a program:

.. testcode:: measurement

    from pyquil import Program
    from pyquil.gates import *

    p = Program()
    ro = p.declare('ro', 'BIT', 2)
    p += H(0)
    p += CNOT(0, 1)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])

In the last two lines, we've added our ``MEASURE`` instructions, saying that we want to store the result of qubit 0
into the 0th bit of ``ro``, and the result of qubit 1 into the 1st bit of ``ro``. The following snippet could be a
useful way to measure many qubits, in particular, on a lattice that doesn't start at qubit 0 (although you can
use the compiler to :ref:`re-index <rewiring>` your qubits):

.. testcode:: measurement

    qubits = [5, 6, 7]
    # ...
    ro = p.declare('ro', 'BIT', len(qubits))
    for i, q in enumerate(qubits):
        p += MEASURE(q, ro[i])

.. _specifying_trials:

Specifying the number of trials
===============================

Quantum computing is inherently probabilistic. We often have to repeat the same experiment many times to get the
results we need. Sometimes we expect the results to all be the same, such as when we apply no gates, or only an ``X``
gate. When we prepare a superposition state, we expect probabilistic outcomes, such as a 50% probability measuring 0 or 1.

The number of shots (also called "trials") is the number of times a program is executed in a single request.
This determines the length of the results that are returned.

If you would like to perform multi-shot execution, you can use ``.wrap_in_numshots_loop``. Below, we specify that our
program should be executed 1000 times:

.. code:: python

    p = Program()
    ...   # build up your program here...
    p.wrap_in_numshots_loop(1000)


.. note::

    Did You Know?

    The word “shot” comes from experimental physics where an experiment is
    performed many times, and each result is called a shot.

.. _build_a_fixed_count_loop:

Build a fixed-count loop with Quil
----------------------------------

Specifying trials with :py:meth:`~pyquil.quil.Program.wrap_in_numshots_loop` doesn't modify the Quil in your program in
any way. Instead, the number of shots you specify is included in your job request and tells the executor how many times
to run your program. However, with Quil's :ref:`classical_control_flow`, instructions it is possible to write a program
that itself defines a loop over a number of shots. The :py:meth:`~pyquil.quil.Program.with_loop` method will help you
do just that. It wraps the body of your program in a loop over a number of iteration you specify and returns the looped
program.

Let's see an example. We'll construct a classic bell state program and measure it 1000 times by applying a numshots
loop.

.. testcode:: with_loop

    from pyquil import Program, get_qc
    from pyquil.quilatom import Label
    from pyquil.gates import H, CNOT

    # Setup the bell state program
    p = Program(
        H(0),
        CNOT(0, 1),
    )
    ro = p.declare("ro", "BIT", 2)
    p.measure(0, ro[0])
    p.measure(1, ro[1])

    # Declare a memory region to hold the number of shots
    shot_count = p.declare("shot_count", "INTEGER")

    # Wrap the program in a loop by specifying the number of iterations, a memory reference to
    # hold the number of iterations, and two labels to mark the beginning and end of the loop.
    looped_program = p.with_loop(1000, shot_count, Label("start-loop"), Label("end-loop"))
    print(looped_program.out())

    qc = get_qc("2q-qvm")
    # Specify your desired shot count in the memory map.
    results = qc.run(looped_program, memory_map={"shot_count": [1000]})

.. testoutput:: with_loop

    DECLARE ro BIT[2]
    DECLARE shot_count INTEGER[1]
    MOVE shot_count[0] 1000
    LABEL @start-loop
    H 0
    CNOT 0 1
    MEASURE 0 ro[0]
    MEASURE 1 ro[1]
    SUB shot_count[0] 1
    JUMP-UNLESS @end-loop shot_count[0]
    JUMP @start-loop
    LABEL @end-loop


.. _parametric_compilation:

**********************
Parametric compilation
**********************

Modern quantum algorithms are often parametric, following a hybrid model. In this hybrid
model, the program ansatz (template of gates) is fixed, and iteratively updated with new
parameters. These new parameters are often determined by an update given by a classical
optimizer. Depending on the complexity of the algorithm, problem of interest, and capabilities
of the classical optimizer, this loop may need to run many times. In order to efficiently operate
within this hybrid model, parametric compilation can be used.

Parametric compilation allows one to compile the program ansatz just once. Making use of declared
memory regions, we can load values to the parametric gates at execution time, after compilation.
Taking the compiler out of the execution loop for programs like this offers a huge performance
improvement compared to compiling the program each time a parameter update is required.

The first step is to build our parametric program, which functions like a template for all the precise programs we will
run. Below we create an example program to illustrate, which puts the qubit onto the equator of the Bloch Sphere and then
rotates it around the Z axis for some variable angle theta before applying another X pulse and measuring.

.. testcode:: parametric

    import numpy as np

    from pyquil import Program
    from pyquil.gates import RX, RZ, MEASURE

    qubit = 0

    p = Program()
    ro = p.declare("ro", "BIT", 1)
    theta_ref = p.declare("theta", "REAL")

    p += RX(np.pi / 2, qubit)
    p += RZ(theta_ref, qubit)
    p += RX(-np.pi / 2, qubit)

    p += MEASURE(qubit, ro[0])

.. note::

    This program is actually more than a toy example. It's similar to an experiment which measures the qubit frequency.

Notice how ``theta`` hasn't been specified yet. The next steps will have to involve a ``QuantumComputer`` or a compiler
implementation. For simplicity, we will demonstrate with a ``QuantumComputer`` instance.

.. testcode:: parametric

    from pyquil import get_qc

    # Get a Quantum Virtual Machine to simulate execution
    qc = get_qc("1q-qvm")
    executable = qc.compile(p)

We are able to compile our program, even with ``theta`` still not specified. Now we want to run our program with ``theta``
filled in for, say, 200 values between :math:`0` and :math:`2\pi`. We demonstrate this below.

.. testcode:: parametric

    # Somewhere to store each list of results
    parametric_measurements = []

    for theta in np.linspace(0, 2 * np.pi, 200):
        # Set the desired parameter value in executable memory
        memory_map = {"theta": [theta]}

        # Get the results of the run with the value we want to execute with
        bitstrings = qc.run(executable, memory_map=memory_map).get_register_map().get("ro")

        # Store our results
        parametric_measurements.append(bitstrings)

In the example here, if you called ``qc.run(executable)`` and didn't specify ``'theta'``, the program would apply
``RZ(0, qubit)`` for every execution.

.. note::

    Classical memory defaults to zero. If you don't specify a value for a declared memory region, it will be zero.

**************
Gate modifiers
**************
Gate applications in Quil can be preceded by a `gate modifier`. There are three supported modifiers:
``DAGGER``, ``CONTROLLED``, and ``FORKED``. The ``DAGGER`` modifier represents the dagger of the gate. For instance,

.. parsed-literal::

    DAGGER RX(pi/3) 0

would have an equivalent effect to ``RX(-pi/3) 0``.

The ``CONTROLLED`` modifier takes a gate and makes it a controlled gate. For instance, one could write the Toffoli gate in any of the three following ways:

.. parsed-literal::

    CCNOT 0 1 2
    CONTROLLED CNOT 0 1 2
    CONTROLLED CONTROLLED X 0 1 2

.. note::
    The letter ``C`` in the gate name has no semantic significance in Quil. To make a controlled ``Y`` gate, one `cannot` write ``CY``, but rather one has to write ``CONTROLLED Y``.

The ``FORKED`` modifier allows for a parametric gate to be applied, with the specific choice of parameters conditional on a qubit value. For a parametric gate ``G`` with k parameters,

.. parsed-literal::

    FORKED G(u1, ..., uk, v1, ..., vk) c q1 ... qn

is equivalent to

.. parsed-literal::

    if c == 0:
        G(u1, ..., uk) q1 ... qn
    else if c == 1:
        G(v1, ..., vk) q1 ... qn

extended by linearity for general ``c``. Note that the total number of parameters in the forked gate has doubled.

All gates (objects deriving from the ``Gate`` class) provide the
methods ``Gate.dagger()``, ``Gate.controlled(control_qubit)``, and ``Gate.forked(fork_qubit, alt_params)``  that
can be used to programmatically apply the ``DAGGER``, ``CONTROLLED``, and ``FORKED`` modifiers.

For example, to produce the controlled-NOT gate (``CNOT``) with
control qubit ``0`` and target qubit ``1``

.. testsetup:: gate-modifiers

   import numpy as np
   from pyquil import Program
   from pyquil.gates import X, RX

.. testcode:: gate-modifiers

   prog = Program(X(1).controlled(0))

To produce the doubly-controlled NOT gate (``CCNOT``) with
control qubits ``0`` and ``1`` and target qubit ``2`` you can stack
the ``controlled`` modifier, or pass a list of control qubits

.. testcode:: gate-modifiers

   prog = Program(X(2).controlled(0).controlled(1))
   prog = Program(X(2).controlled([0, 1]))

You can achieve the oft-used `control-off` gate (flip the target qubit
``1`` if the control qubit ``0`` is zero) with

.. testcode:: gate-modifiers

   prog = Program(X(0), X(1).controlled(0), X(0))

The gate ``FORKED RX(pi/2, pi) 0 1`` may be produced by

.. testcode:: gate-modifiers

   prog = Program(RX(np.pi/2, 1).forked(0, [np.pi]))


******************
Defining new gates
******************

New gates can also be added inline to Quil programs. All you need is a
matrix representation of the gate. For example, below we define a
:math:`\sqrt{X}` gate.

.. testcode:: define-gates

    import numpy as np

    from pyquil import Program
    from pyquil.quil import DefGate

    # First we define the new gate from a matrix
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                       [ 0.5-0.5j,  0.5+0.5j]])

    # Get the Quil definition for the new gate
    sqrt_x_definition = DefGate("SQRT-X", sqrt_x)
    # Get the gate constructor
    SQRT_X = sqrt_x_definition.get_constructor()

    # Then we can use the new gate
    p = Program()
    p += sqrt_x_definition
    p += SQRT_X(0)
    print(p)

.. testoutput:: define-gates

    DEFGATE SQRT-X AS MATRIX:
        0.5+0.5i, 0.5-0.5i
        0.5-0.5i, 0.5+0.5i

    SQRT-X 0

Below we show how we can define :math:`X_0\otimes \sqrt{X_1}` as a single gate.

.. testcode:: define-gates

    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                    [ 0.5-0.5j,  0.5+0.5j]])
    x_sqrt_x = np.kron(x_gate_matrix, sqrt_x)

Now we can use this gate in the same way that we used ``SQRT_X``, but we will pass it two arguments
rather than one, since it operates on two qubits.

.. testcode:: define-gates

    x_sqrt_x_definition = DefGate("X-SQRT-X", x_sqrt_x)
    X_SQRT_X = x_sqrt_x_definition.get_constructor()

    # Then we can use the new gate
    p = Program(x_sqrt_x_definition, X_SQRT_X(0, 1))

.. tip::

    To inspect the wavefunction that will result from applying your new gate, you can use
    the :ref:`Wavefunction Simulator <wavefunction_simulator>`
    (e.g. ``print(WavefunctionSimulator().wavefunction(p))``).


*************************
Defining parametric gates
*************************

Let's say we want to have a controlled RX gate. Since RX is a parametric gate, we need a slightly different way of
defining it than in the previous section.

.. testcode:: parametric

    from pyquil import Program
    from pyquil.api import WavefunctionSimulator
    from pyquil.gates import H
    from pyquil.quilatom import Parameter, quil_sin, quil_cos
    from pyquil.quilbase import DefGate
    import numpy as np

    # Define the new gate from a matrix
    theta = Parameter('theta')
    crx = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
        [0, 0, -1j * quil_sin(theta / 2), quil_cos(theta / 2)]
    ])

    gate_definition = DefGate('CRX', crx, [theta])
    CRX = gate_definition.get_constructor()

    # Create our program and use the new parametric gate
    p = Program()
    p += gate_definition
    p += H(0)
    p += CRX(np.pi/2)(0, 1)


``quil_sin`` and ``quil_cos`` work as the regular sines and cosines, but they support the parametrization. Parametrized
functions you can use with pyQuil are: ``quil_sin``, ``quil_cos``, ``quil_sqrt``, ``quil_exp``, and ``quil_cis``.

.. tip::

    To inspect the wavefunction that will result from applying your new gate, you can use
    the :ref:`Wavefunction Simulator <wavefunction_simulator>`
    (e.g. ``print(WavefunctionSimulator().wavefunction(p))``).


**************************
Defining permutation gates
**************************

Some gates can be compactly represented as a permutation. For example, ``CCNOT`` gate can be represented by the matrix

.. testcode:: permutation

   import numpy as np
   from pyquil.quilbase import DefGate

   ccnot_matrix = np.array([
       [1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 1, 0]
   ])

   ccnot_gate = DefGate("MATRIX_CCNOT", ccnot_matrix)

   # etc

It can equivalently be defined by the permutation

.. testcode:: permutation

   import numpy as np
   from pyquil.quilbase import DefPermutationGate

   ccnot_gate = DefPermutationGate("PERMUTATION_CCNOT", [0, 1, 2, 3, 4, 5, 7, 6])

   # etc

*******
Pragmas
*******

``PRAGMA`` directives give users more control over how Quil programs are processed or simulated but generally do not
change the semantics of the Quil program itself. As a general rule of thumb, deleting all ``PRAGMA`` directives in a Quil
program should leave a valid and semantically equivalent program.

In pyQuil, ``PRAGMA`` directives play many roles, such as controlling the behavior of gates in noisy simulations,
or commanding the Quil compiler to perform actions in a certain way. Here, we will cover the basics of using a ``PRAGMA``
directive to specify a qubit rewiring scheme, a common use case for pragmas. For a more comprehensive
review of what pragmas are and what the compiler supports, check out :ref:`compiler`. For more information about
``PRAGMA`` in Quil, see
`A Practical Quantum ISA <https://arxiv.org/pdf/1608.03355.pdf>`_, and
`Simulating Quantum Processor Errors <https://www.european-lisp-symposium.org/static/proceedings/2018.pdf>`_.

.. _rewiring:

Specifying a qubit rewiring scheme
==================================

Qubit rewiring is one of the most powerful features of the Quil compiler. We're able to write Quil programs which are
agnostic to the topology of the chip, and the compiler will intelligently relabel our qubits to
give better performance.

When we intend to run a program on the QPU, sometimes we write programs which use specific qubits targeting a specific
device topology, perhaps to achieve a high-performance program. Other times, we write programs that are agnostic to the
underlying topology, thereby making the programs more portable. Qubit rewiring accommodates both use cases in an
automatic way.

Consider the following program.

.. testcode:: rewiring

    from pyquil import Program
    from pyquil.gates import *

    p = Program(X(3))

We've tested this on the QVM, and we've targeted a lattice on the QPU which has qubits 4, 5, and 6, but not qubit 3.
Rather than rewrite our program, we modify our program to tell the compiler to do this for us.

.. testcode:: rewiring

    from pyquil.quil import Pragma

    p = Program(Pragma('INITIAL_REWIRING', ['"GREEDY"']))
    p += X(3)

Now, when we pass our program through the compiler (such as with :py:func:`QuantumComputer.compile`) we will get native Quil
with the qubit reindexed to one of 4, 5, or 6. If qubit 3 is available, and we don't want that pulse to be applied to
any other qubit, we would instead use ``Pragma('INITIAL_REWIRING', ['"NAIVE"']]``. Detailed information about the
available options is :ref:`here <compiler_rewirings>`.

.. note::
    In general, we assume that the qubits you're supplying as input are also the ones which you prefer to
    operate on, and so NAIVE rewiring is the default.

******************
Asking for a delay
******************

At times, we may want to add a delay in our program. Usually this is associated with qubit characterization.
As part of the :ref:`quilt` extension to Quil, ``DELAY`` instructions allow you to insert a gap within a list
of pulses or gates with a specified duration in seconds. ``DELAY`` instructions aren't regular gate operations,
and they don't affect they abstract semantics of the Quil program, but you can add one to your program much like
any other instruction:

.. testsetup:: delay

    from pyquil.quil import Program
    from pyquil.gates import DELAY
    p = Program()

.. testcode:: delay

    #  ...
    # qubit indices and time in seconds must be provided
    p += DELAY(0, 200e-9)

.. testoutput:: delay
   :hide:

   ...

.. warning::
   `DELAY` and other Quil-T instructions aren't supported by the QVM.

.. warning::
   In pyQuil v3 and below, it was common to specify a delay using ``PRAGMA DELAY``. This is no longer supported in v4 because it
   conflicts with Quil-T's ``DELAY`` instruction described above. They serve the same function, so we recommend using the ``DELAY``
   instruction instead.

**************************
Ways to construct programs
**************************

pyQuil supports a variety of methods for constructing programs.
Multiple instructions can be added at once, and programs can be concatenated together. pyQuil can also produce a
``Program`` by interpreting raw Quil text. The following are all valid programs:

.. testsetup:: construct-programs

   from pyquil import Program
   from pyquil.gates import X, Y

.. testcode:: construct-programs

    # Preferred method
    p = Program()
    p += X(0)
    p += Y(1)
    print(p)

    # Multiple instructions in declaration
    print(Program(X(0), Y(1)))

    # A composition of two programs
    print(Program(X(0)) + Program(Y(1)))

    # Raw Quil with newlines
    print(Program("X 0\nY 1"))

    # Raw Quil comma separated
    print(Program("X 0", "Y 1"))

    # Chained inst; less preferred
    print(Program().inst(X(0)).inst(Y(1)))


All of the above methods will produce the same output:

.. testoutput:: construct-programs
   :hide:

    X 0
    Y 1
    X 0
    Y 1
    X 0
    Y 1
    X 0
    Y 1
    X 0
    Y 1
    X 0
    Y 1

.. parsed-literal::

   X 0
   Y 1
