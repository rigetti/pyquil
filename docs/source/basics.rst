.. _basics:

Programs and Gates
==================

.. note::

    If you're running locally, remember set up the QVM and quilc in server mode before trying to use
    them: :ref:`server`.

Introduction
~~~~~~~~~~~~

Quantum programs are written in Forest using the :py:class:`~pyquil.quil.Program` object. This ``Program`` abstraction will help us
compose `Quil programs <https://arxiv.org/abs/1608.03355>`_.

.. code:: python

    from pyquil import Program

Programs are constructed by adding quantum gates to it, which are defined in the ``gates`` module. We can import all
standard gates with the following:

.. code:: python

     from pyquil.gates import *

Let's instantiate a ``Program`` and add an operation to it. We will act an ``X`` gate on qubit 0.

.. code:: python

    p = Program()
    p += X(0)

All qubits begin in the ground state. This means that if we measure a qubit without applying operations on it, we expect to receive
a 0 result. The ``X`` gate will rotate qubit 0 from the ground state to the excited state, so a measurement immediately
after should return a 1 result. More details about gate operations are explained in :ref:`intro`.

We can print our pyQuil program (``print(p)``) to see the equivalent Quil representation:

.. parsed-literal::

    X 0


This isn't going to be very useful to us without measurements. In pyQuil 2.0, we have to ``DECLARE`` a memory space
to read measurement results, which we call "readout results" and abbreviate as ``ro``. With measurement, our whole program
looks like this:

.. code:: python

    from pyquil import Program
    from pyquil.gates import *

    p = Program()
    ro = p.declare('ro', 'BIT', 1)
    p += X(0)
    p += MEASURE(0, ro[0])

    print(p)

.. parsed-literal::

    DECLARE ro BIT[1]
    X 0
    MEASURE 0 ro[0]

We've instantiated a program, declared a memory space named ``ro`` with one single bit of memory, applied
an ``X`` gate on qubit 0, and finally measured qubit 0 into the zeroth index of the memory space named ``ro``.

Awesome! That's all we need to get results back. Now we can actually see what happens if we run this
program on the Quantum Virtual Machine (QVM). We just have to add a few lines to do this.

.. code:: python

    from pyquil import get_qc

    ...

    qc = get_qc('1q-qvm')  # You can make any 'nq-qvm' this way for any reasonable 'n'
    executable = qc.compile(p)
    result = qc.run(executable)
    print(result)

Congratulations! You just ran your program on the QVM. The returned value should be:

.. parsed-literal::

    [[1]]

For more information on what the above result means, and on executing quantum programs on the QVM in
general, see :ref:`qvm`. The remainder of this section of the docs will be dedicated to constructing
programs in detail, an essential part of becoming fluent in quantum programming.


.. _standard:

The Standard Gate Set
~~~~~~~~~~~~~~~~~~~~~

The following gates methods come standard with Quil and ``gates.py``:

-  Pauli gates ``I``, ``X``, ``Y``, ``Z``

-  Hadamard gate: ``H``

-  Phase gates: ``PHASE(theta)``, ``S``, ``T``

-  Controlled phase gates: ``CZ``, ``CPHASE00(alpha)``,
   ``CPHASE01(alpha)``, ``CPHASE10(alpha)``, ``CPHASE(alpha)``

-  Cartesian rotation gates: ``RX(theta)``, ``RY(theta)``, ``RZ(theta)``

-  Controlled :math:`X` gates: ``CNOT``, ``CCNOT``

-  Swap gates: ``SWAP``, ``CSWAP``, ``ISWAP``, ``PSWAP(alpha)``

The parameterized gates take a real or complex floating point
number as an argument.

.. _declaring_memory:

Declaring Memory
~~~~~~~~~~~~~~~~

Classical memory regions must be explicitly requested and named by a Quil program using the ``DECLARE`` directive.
Details about can be found in the :ref:`migration guide <quil_2_declare>` or in :py:func:`pyquil.quil.Program.declare`.

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
    you might want to name the third bit in your readout array as ``q3_ro``. ``SHARING`` is currently disallowed for
    our QPUs, so we won't focus on this here.

Now we can get into an example.

.. code:: python

    from pyquil import Program

    p = Program()
    ro = p.declare('ro', 'BIT', 16)
    theta = p.declare('theta', 'REAL')

.. warning::
    ``.declare`` cannot be chained, since it doesn't return a modified ``Program`` object.

Notice that the ``.declare`` method returns a reference to the memory we've just declared. We will need this reference
to make use of these memory spaces again. Let's see how the Quil is looking so far:

.. parsed-literal::

    DECLARE ro BIT[16]
    DECLARE theta REAL[1]


That's all we have to do to declare the memory. Continue to the next section on :ref:`measurement` to learn more about
using ``ro`` to store measured readout results. Check out :ref:`parametric_compilation` to see how you might use
``theta`` to compile gate parameters dynamically.

.. _measurement:

Measurement
~~~~~~~~~~~

There are several ways you can handle measurements in your program. We will start with the simplest method -- letting
the ``QuantumComputer`` abstraction do it for us.

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import H, CNOT

    # Get our QuantumComputer instance, with a Quantum Virutal Machine (QVM) backend
    qc = get_qc("8q-qvm")

    # Construct a simple Bell State
    p = Program(H(0), CNOT(0, 1))

    results = qc.run_and_measure(p, trials=10)
    print(results)

.. parsed-literal::

    {0: array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
     1: array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
     2: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     3: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     4: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     5: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     6: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     7: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}

The method ``.run_and_measure`` will handle declaring memory for readout results, adding ``MEASURE`` instructions for each
qubit in the QVM, telling the QVM how many trials to run, running and returning the measurement results.

You might sometimes want finer grained control. In this case, we're probably only interested in the results on
qubits 0 and 1, but ``.run_and_measure`` returns the results for all eight qubits in the QVM. We can change our program
to be more particular about what we want.

.. code:: python

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

.. code:: python

    qubits = [5, 6, 7]
    # ...
    for i, q in enumerate(qubits):
        p += MEASURE(q, ro[i])

.. note::

    The QPU can only handle ``MEASURE`` final programs. You can't operate gates after measurements.

Specifying the number of trials
-------------------------------

Quantum computing is inherently probabilistic. We often have to repeat the same experiment many times to get the
results we need. Sometimes we expect the results to all be the same, such as when we apply no gates, or only an ``X``
gate. When we prepare a superposition state, we expect probabilistic outcomes, such as a 50% probability measuring 0 or 1.

The number of `shots` (also called `trials`) is the number of times to execute a program at once.
This determines the length of the results that are returned.

As we saw above, the ``.run_and_measure`` method of the ``QuantumComputer`` object can handle multiple executions of a program.
If you would like more explicit control for representing multi-shot execution, another way to do this is
with ``.wrap_in_numshots_loop``. This puts the number of shots to be run in the representation of the program itself,
as opposed to in the arguments list of the execution method itself. Below, we specify that our program should
be executed 1000 times.

.. code:: python

    p = Program()
    ...   # build up your program here...
    p.wrap_in_numshots_loop(1000)


.. note::

    Did You Know?

    The word “shot” comes from experimental physics where an experiment is
    performed many times, and each result is called a shot.


.. _parametric_compilation:

Parametric Compilation
~~~~~~~~~~~~~~~~~~~~~~

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
(Some more details about this and an example are found :doc:`here <migration3-declare>`.)

The first step is to build our parametric program, which functions like a template for all the precise programs we will
run. Below we create a simple example program to illustrate, which puts the qubit onto the equator of the Bloch Sphere and then
rotates it around the Z axis for some variable angle theta before applying another X pulse and measuring.

.. code:: python

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

    The example program, although simple, is actually more than just a toy example. It is similar to an
    experiment which measures the qubit frequency.

Notice how ``theta`` hasn't been specified yet. The next steps will have to involve a ``QuantumComputer`` or a compiler
implementation. For simplicity, we will demonstrate with a ``QuantumComputer`` instance.

.. code:: python

    from pyquil import get_qc

    # Get a Quantum Virtual Machine to simulate execution
    qc = get_qc("1q-qvm")
    executable = qc.compile(p)

We are able to compile our program, even with ``theta`` still not specified. Now we want to run our program with ``theta``
filled in for, say, 200 values between :math:`0` and :math:`2\pi`. We demonstrate this below.

.. code:: python

    # Somewhere to store each list of results
    parametric_measurements = []

    for theta in np.linspace(0, 2 * np.pi, 200):
        # Get the results of the run with the value we want to execute with
        bitstrings = qc.run(executable, {'theta': [theta]})
        # Store our results
        parametric_measurements.append(bitstrings)

In the example here, if you called ``qc.run(executable)`` and didn't specify ``'theta'``, the program would apply
``RZ(0, qubit)`` for every execution.

.. note::
    Classical memory defaults to zero. If you don't specify a value for a declared memory region, it will be zero.

Gate Modifiers
~~~~~~~~~~~~~~
Gate applications in Quil can be preceded by a `gate modifier`. There are two supported modifiers:
``DAGGER`` and ``CONTROLLED``. The ``DAGGER`` modifier represents the dagger of the gate. For instance,

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

All gates (objects deriving from the ``Gate`` class) provide the
methods ``Gate.controlled(control_qubit)`` and ``Gate.dagger()`` that
can be used to programmatically apply the ``CONTROLLED`` and
``DAGGER`` modifiers.

For example, to produce the controlled-NOT gate (``CNOT``) with
control qubit ``0`` and target qubit ``1``

.. code:: python

   prog = Program(X(1).controlled(0))

You can achieve the oft-used `control-off` gate (flip the target qubit
``1`` if the control qubit ``0`` is zero) with

.. code:: python

   prog = Program(X(0), X(1).controlled(0), X(0))

Defining New Gates
~~~~~~~~~~~~~~~~~~

New gates can be easily added inline to Quil programs. All you need is a
matrix representation of the gate. For example, below we define a
:math:`\sqrt{X}` gate.

.. code:: python

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

.. parsed-literal::

    DEFGATE SQRT-X:
        0.5+0.5i, 0.5-0.5i
        0.5-0.5i, 0.5+0.5i

    SQRT-X 0

Below we show how we can define :math:`X_0\otimes \sqrt{X_1}` as a single gate.

.. code:: python

    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                    [ 0.5-0.5j,  0.5+0.5j]])
    x_sqrt_x = np.kron(x_gate_matrix, sqrt_x)

Now we can use this gate in the same way that we used ``SQRT_X``, but we will pass it two arguments
rather than one, since it operates on two qubits.

.. code:: python

    x_sqrt_x_definition = DefGate("X-SQRT-X", x_sqrt_x)
    X_SQRT_X = x_sqrt_x_definition.get_constructor()

    # Then we can use the new gate
    p = Program(x_sqrt_x_definition, X_SQRT_X(0, 1))

.. tip::

    To inspect the wavefunction that will result from applying your new gate, you can use
    the :ref:`Wavefunction Simulator <wavefunction_simulator>`
    (e.g. ``print(WavefunctionSimulator().wavefunction(p))``).


Defining Parametric Gates
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we want to have a controlled RX gate. Since RX is a parametric gate, we need a slightly different way of
defining it than in the previous section.

.. code:: python

    from pyquil import Program, WavefunctionSimulator
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


Defining Permutation Gates
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   ``quilc`` supports permutation gate syntax since version ``1.8.0``. pyQuil introduced support in version ``2.8.0``.

Some gates can be compactly represented as a permutation. For example, ``CCNOT`` gate can be represented by the matrix

.. code:: python

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
       [1, 0, 0, 0, 0, 0, 1, 0]
   ])

   ccnot_gate = DefGate("CCNOT", ccnot_matrix)

   # etc

It can equivalently be defined by the permutation

.. code:: python

   import numpy as np
   from pyquil.quilbase import DefPermutationGate

   ccnot_gate = DefPermutationGate("CCNOT", [0, 1, 2, 3, 4, 5, 7, 6])

   # etc

Pragmas
~~~~~~~

``PRAGMA`` directives give users more control over how Quil programs are processed or simulated but generally do not
change the semantics of the Quil program itself. As a general rule of thumb, deleting all ``PRAGMA`` directives in a Quil
program should leave a valid and semantically equivalent program.

In pyQuil, ``PRAGMA`` directives play many roles, such as controlling the behavior of gates in noisy simulations,
or commanding the Quil compiler to perform actions in a certain way. Here, we will cover the basics of two very
common use cases for including a ``PRAGMA`` in your program: qubit rewiring and delays. For a more comprehensive
review of what pragmas are and what the compiler supports, check out :ref:`compiler`. For more information about
``PRAGMA`` in Quil, see
`A Practical Quantum ISA <https://arxiv.org/pdf/1608.03355.pdf>`_, and
`Simulating Quantum Processor Errors <https://www.european-lisp-symposium.org/static/proceedings/2018.pdf>`_.

.. _rewiring:

Specifying A Qubit Rewiring Scheme
----------------------------------

Qubit rewiring is one of the most powerful features of the Quil compiler. We are able to write Quil programs which are
agnostic to the topology of the chip, and the compiler will intelligently relabel our qubits to
give better performance.

When we intend to run a program on the QPU, sometimes we write programs which use specific qubits targeting a specific
device topology, perhaps to achieve a high-performance program. Other times, we write programs that are agnostic to the
underlying topology, thereby making the programs more portable. Qubit rewiring accommodates both use cases in an
automatic way.

Consider the following program.

.. code:: python

    from pyquil import Program
    from pyquil.gates import *

    p = Program(X(3))

We've tested this on the QVM, and we've reserved a lattice on the QPU which has qubits 4, 5, and 6, but not qubit 3.
Rather than rewrite our program for each reservation, we modify our program to tell the compiler to do this for us.

.. code:: python

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

Asking for a Delay
------------------

At times, we may want to add a delay in our program. Usually this is associated with qubit characterization. Delays
are not regular gate operations, and they do not affect the abstract semantics of the Quil program, so they're implemented with a ``PRAGMA`` directive.

.. code:: python

    #  ...
    # qubit index and time in seconds must be defined and provided
    # the time argument accepts exponential notation e.g. 200e-9
    p += Pragma('DELAY', [qubit], str(time))

.. warning::
    These delays currently have effects on the real QPU. They have no effect on QVM's even when those QVM's have noise
    models applied.

.. warning::
    Keep in mind, the program duration is currently capped at 15 seconds, and the length of the program is multiplied
    by the number of shots. If you have a 1000 shot program, where each shot contains a 100ms delay, you won't be able to execute it.

Ways to Construct Programs
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyQuil supports a variety of methods for constructing programs however you prefer.
Multiple instructions can be applied at once, and programs can be added together. PyQuil can also produce a
``Program`` by interpreting raw Quil text. You can still use the more pyQuil 1.X style of using
the ``.inst`` method to add instruction gates. Thus, the following are all valid programs:

.. code:: python

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

.. parsed-literal::

    X 0
    Y 1

The ``pyquil.parser`` submodule provides a front-end to other similar parser
functionality.


Fixing a Mistaken Instruction
-----------------------------

If an instruction was appended to a program incorrectly, you can pop it off.

.. code:: python

    p = Program(X(0), Y(1))
    print(p)

    print("We can fix by popping:")
    p.pop()
    print(p)

.. parsed-literal::

    X 0
    Y 1

    We can fix by popping:
    X 0

QPU-allowable Quil
~~~~~~~~~~~~~~~~~~

Apart from ``DECLARE`` and ``PRAGMA`` directives, a program must break into the following three regions, each optional:

1. A ``RESET`` command.
2. A sequence of quantum gate applications.
3. A sequence of ``MEASURE`` commands.

The only memory that is writeable is the region named ``ro``, and only through ``MEASURE`` instructions. All other
memory is read-only.

The keyword ``SHARING`` is disallowed.

Compilation is unavailable for invocations of ``DEFGATE``\ s with parameters read from classical memory.
