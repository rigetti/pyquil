.. _basics:

Programs and Gates
==================

.. note::

    If you're running locally, remember set up the QVM and quilc in server mode before trying to use
    them: :ref:`server`.

Introduction
~~~~~~~~~~~~

Quantum programs are written in Forest using the ``Program`` object. This ``Program`` abstraction will help us
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
    compiled_program = qc.compile(p)
    result = qc.run(compiled_program)
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


Declaring Memory
~~~~~~~~~~~~~~~~
*Coming soon*


Measurement
~~~~~~~~~~~

*Coming soon*


Specifying the number of trials
-------------------------------

*Coming soon*


Parametric Compilation
~~~~~~~~~~~~~~~~~~~~~~
*Coming soon*



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
    from pyquil.parameters import Parameter, quil_sin, quil_cos
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


Pragmas
~~~~~~~

Specifying A Qubit Rewiring Scheme
----------------------------------

*Coming soon*

Asking for a Delay
------------------
*Coming soon*
(Note: time limit)


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
