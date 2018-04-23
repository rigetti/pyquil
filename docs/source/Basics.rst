The Fundamentals of Quantum Programming
=======================================


The Standard Gate Set
~~~~~~~~~~~~~~~~~~~~~

The following gates methods come standard with Quil and ``gates.py``:

-  Pauli gates ``I``, ``X``, ``Y``, ``Z``

-  Hadamard gate: ``H``

-  Phase gates: ``PHASE(``\ :math:`\theta`\ ``)``, ``S``, ``T``

-  Controlled phase gates: ``CZ``, ``CPHASE00(`` :math:`\alpha` ``)``,
   ``CPHASE01(`` :math:`\alpha` ``)``, ``CPHASE10(`` :math:`\alpha`
   ``)``, ``CPHASE(`` :math:`\alpha` ``)``

-  Cartesian rotation gates: ``RX(`` :math:`\theta` ``)``, ``RY(``
   :math:`\theta` ``)``, ``RZ(`` :math:`\theta` ``)``

-  Controlled :math:`X` gates: ``CNOT``, ``CCNOT``

-  Swap gates: ``SWAP``, ``CSWAP``, ``ISWAP``, ``PSWAP(`` :math:`\alpha`
   ``)``

The parameterized gates take a real or complex floating point
number as an argument.

Defining New Gates
~~~~~~~~~~~~~~~~~~

New gates can be easily added inline to Quil programs. All you need is a
matrix representation of the gate. For example, below we define a
:math:`\sqrt{X}` gate.

.. code:: python

    import numpy as np

    # First we define the new gate from a matrix
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                       [ 0.5-0.5j,  0.5+0.5j]])
    p = Program().defgate("SQRT-X", sqrt_x)

    # Then we can use the new gate,
    p.inst(("SQRT-X", 0))
    print(p)

.. parsed-literal::

    DEFGATE SQRT-X:
        0.5+0.5i, 0.5-0.5i
        0.5-0.5i, 0.5+0.5i

    SQRT-X 0

.. code:: python

    print(qvm.wavefunction(p))

.. parsed-literal::

    (0.5+0.5j)|0> + (0.5-0.5j)|1>

Below we show how we can define :math:`X_0\otimes \sqrt{X_1}` as a single gate.

.. code:: python

    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                    [ 0.5-0.5j,  0.5+0.5j]])
    x_sqrt_x = np.kron(x_gate_matrix, sqrt_x)
    p = Program().defgate("X-SQRT-X", x_sqrt_x)

    # Then we can use the new gate
    p.inst(("X-SQRT-X", 0, 1))
    wavefunction = qvm.wavefunction(p)
    print(wavefunction)

.. parsed-literal::

    (0.5+0.5j)|01> + (0.5-0.5j)|11>

Defining Parametric Gates
~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to define parametric gates using pyQuil.
Let's say we want to have a controlled RX gate. Since RX is a parametric gate, we need a slightly different way of defining it than in the previous section.

.. code:: python

    from pyquil.parameters import Parameter, quil_sin, quil_cos
    from pyquil.quilbase import DefGate
    import numpy as np

    theta = Parameter('theta')
    crx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, quil_cos(theta / 2), -1j * quil_sin(theta / 2)], [0, 0, -1j * quil_sin(theta / 2), quil_cos(theta / 2)]])

    dg = DefGate('CRX', crx, [theta])
    CRX = dg.get_constructor()

    p = Program()
    p.inst(dg)
    p.inst(H(0))
    p.inst(CRX(np.pi/2)(0, 1))

    wavefunction = qvm.wavefunction(p)
    print(wavefunction)

.. parsed-literal::

    (0.7071067812+0j)|00> + (0.5+0j)|01> + -0.5j|11>

``quil_sin`` and ``quil_cos`` work as the regular sinus and cosinus, but they support the parametrization. Parametrized functions you can use with pyQuil are: ``quil_sin``, ``quil_cos``, ``quil_sqrt``, ``quil_exp``, and ``quil_cis``.

Qubit Operations
----------------

In the previous section we introduced our first two **operations**: the ``I``
(or identity) operation and the ``X`` operation. In this section we will get into some
more details on what these operations are.

Quantum states are complex vectors on the Bloch sphere, and quantum operations are matrices with two properties:

1. They are reversible.
2. When applied to a state vector on the Bloch sphere, the resulting vector
   is also on the Bloch sphere.

Matrices that satisfy these two properties are called unitary matrices. Applying an operation to a quantum state is the same as multiplying a vector by one of these matrices. Such an operation is called a **gate**.

Since individual qubits are two-dimensional vectors, operations on
individual qubits are 2x2 matrices. The identity matrix leaves the state vector unchanged:

.. math::

   I = \left(\begin{matrix}
   1 & 0\\
   0 & 1
   \end{matrix}\right)

so the program that applies this operation to the zero state is just

.. math::

    I\,|\,0\rangle = \left(\begin{matrix}
   1 & 0\\
   0 & 1
   \end{matrix}\right)\left(\begin{matrix}
   1 \\
   0
   \end{matrix}\right) = \left(\begin{matrix}
   1 \\
   0
   \end{matrix}\right) = |\,0\rangle

.. code:: python

    p = Program(I(0))
    print(quantum_simulator.wavefunction(p))

.. parsed-literal::

    (1+0j)|0>

Pauli Operators
^^^^^^^^^^^^^^^

Let's revisit the ``X`` gate introduced above. It is one of three important single-qubit gates,
called the Pauli operators:

.. math::


   X = \left(\begin{matrix}
   0 & 1\\
   1 & 0
   \end{matrix}\right)
   \qquad
   Y = \left(\begin{matrix}
   0 & -i\\
   i & 0
   \end{matrix}\right)
   \qquad
   Z = \left(\begin{matrix}
   1 & 0\\
   0 & -1
   \end{matrix}\right)

.. code:: python

    from pyquil.gates import X, Y, Z

    p = Program(X(0))
    wavefunction = quantum_simulator.wavefunction(p)
    print("X|0> = ", wavefunction)
    print("The outcome probabilities are", wavefunction.get_outcome_probs())
    print("This looks like a bit flip.\n")

    p = Program(Y(0))
    wavefunction = quantum_simulator.wavefunction(p)
    print("Y|0> = ", wavefunction)
    print("The outcome probabilities are", wavefunction.get_outcome_probs())
    print("This also looks like a bit flip.\n")

    p = Program(Z(0))
    wavefunction = quantum_simulator.wavefunction(p)
    print("Z|0> = ", wavefunction)
    print("The outcome probabilities are", wavefunction.get_outcome_probs())
    print("This state looks unchanged.")


.. parsed-literal::

    X|0> =  (1+0j)|1>
    The outcome probabilities are {'1': 1.0, '0': 0.0}
    This looks like a bit flip.

    Y|0> =  1j|1>
    The outcome probabilities are {'1': 1.0, '0': 0.0}
    This also looks like a bit flip.

    Z|0> =  (1+0j)|0>
    The outcome probabilities are {'1': 0.0, '0': 1.0}
    This state looks unchanged.

The Pauli matrices have a visual interpretation: they perform 180-degree rotations of
qubit state vectors on the Bloch sphere. They operate about their respective axes
as shown in the Bloch sphere depicted above. For example, the ``X`` gate performs a 180-degree
rotation **about** the \\(x\\) axis. This explains the results of our code above: for a state vector
initially in the +\\(z\\) direction, both ``X`` and ``Y`` gates will rotate it to -\\(z\\),
and the ``Z`` gate will leave it unchanged.

However, notice that while the ``X`` and ``Y`` gates produce the same outcome probabilities, they
actually produce different states. These states are not distinguished if they are measured
immediately, but they produce different results in larger programs.

Quantum programs are built by applying successive gate operations:

.. code:: python

    # Composing qubit operations is the same as multiplying matrices sequentially
    p = Program(X(0), Y(0), Z(0))
    wavefunction = quantum_simulator.wavefunction(p)

    print("ZYX|0> = ", wavefunction)
    print("With outcome probabilities\n", wavefunction.get_outcome_probs())


.. parsed-literal::

    ZYX|0> =  [ 0.-1.j  0.+0.j]
    With outcome probabilities
    {'0': 1.0, '1': 0.0}


Multi-Qubit Operations
^^^^^^^^^^^^^^^^^^^^^^

Operations can also be applied to composite states of multiple qubits.
One common example is the controlled-NOT or ``CNOT`` gate that works on two
qubits. Its matrix form is:

.. math::


   CNOT = \left(\begin{matrix}
   1 & 0 & 0 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 1 \\
   0 & 0 & 1 & 0 \\
   \end{matrix}\right)

Let's take a look at how we could use a ``CNOT`` gate in pyQuil.

.. code:: python

    from pyquil.gates import CNOT

    p = Program(CNOT(0, 1))
    wavefunction = quantum_simulator.wavefunction(p)
    print("CNOT|00> = ", wavefunction)
    print("With outcome probabilities\n", wavefunction.get_outcome_probs())

    p = Program(X(0), CNOT(0, 1))
    wavefunction = quantum_simulator.wavefunction(p)
    print("CNOT|01> = ", wavefunction)
    print("With outcome probabilities\n", wavefunction.get_outcome_probs())

    p = Program(X(1), CNOT(0, 1))
    wavefunction = quantum_simulator.wavefunction(p)
    print("CNOT|10> = ", wavefunction)
    print("With outcome probabilities\n", wavefunction.get_outcome_probs())

    p = Program(X(0), X(1), CNOT(0, 1))
    wavefunction = quantum_simulator.wavefunction(p)
    print("CNOT|11> = ", wavefunction)
    print("With outcome probabilities\n", wavefunction.get_outcome_probs())


.. parsed-literal::

    CNOT|00> =  (1+0j)|00>
    With outcome probabilities
     {'00': 1.0, '01': 0.0, '10': 0.0, '11': 0.0}

    CNOT|01> =  (1+0j)|11>
    With outcome probabilities
     {'00': 0.0, '01': 0.0, '10': 0.0, '11': 1.0}

    CNOT|10> =  (1+0j)|10>
    With outcome probabilities
     {'00': 0.0, '01': 0.0, '10': 1.0, '11': 0.0}

    CNOT|11> =  (1+0j)|01>
    With outcome probabilities
     {'00': 0.0, '01': 1.0, '10': 0.0, '11': 0.0}


The ``CNOT`` gate does what its name implies: the state of the second qubit is flipped
(negated) if and only if the state of the first qubit is 1 (true).

Another two-qubit gate example is the ``SWAP`` gate, which swaps the \\( \|01\\rangle \\)
and \\(\|10\\rangle \\) states:

.. math::


   SWAP = \left(\begin{matrix}
   1 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 1 \\
   \end{matrix}\right)

.. code:: python

    from pyquil.gates import SWAP
    p = Program(X(0), SWAP(0,1))
    wavefunction = quantum_simulator.wavefunction(p)

    print("SWAP|01> = ", wavefunction)
    print("With outcome probabilities\n", wavefunction.get_outcome_probs())


.. parsed-literal::

    SWAP|01> =  (1+0j)|10>
    With outcome probabilities
     {'00': 0.0, '01': 0.0, '10': 1.0, '11': 0.0}

In summary, quantum computing operations are composed of a series of
complex matrices applied to complex vectors. These matrices must be unitary (meaning that
their complex conjugate transpose is equal to their inverse) because the overall probability of
all outcomes must always sum to one.


Qubit Measurements
^^^^^^^^^^^^^^^^^^

Measurements have two effects:

#. They project the state vector onto one of the basic outcomes
#. (*optional*) They store the outcome
   of the measurement in a classical bit.

Here's a simple example:

.. code:: python

    # Create a program that stores the outcome of measuring qubit #0 into classical register [0]
    classical_register_index = 0
    p = Program(I(0)).measure(0, classical_register_index)

Up until this point we have used the quantum simulator to cheat a little bit --- we have
actually looked at the wavefunction that comes back. However, on real
quantum hardware, we are unable to directly look at the wavefunction.
Instead we only have access to the classical bits that are affected by
measurements. This functionality is emulated by the ``run`` command.

.. code:: python

    # Choose which classical registers to look in at the end of the computation
    classical_regs = [0, 1]
    print(quantum_simulator.run(p, classical_regs))


.. parsed-literal::

    [[0, 0]]


We see that both registers are zero. However, if we had flipped the
qubit before measurement then we obtain:

.. code:: python

    classical_register_index = 0
    p = Program(X(0)) # Flip the qubit
    p.measure(0, classical_register_index) # Measure the qubit

    classical_regs = [0, 1]
    print(quantum_simulator.run(p, classical_regs))


.. parsed-literal::

    [[1, 0]]


These measurements are deterministic, e.g. if we make them multiple
times then we always get the same outcome:

.. code:: python

    classical_register_index = 0
    p = Program(X(0)) # Flip the qubit
    p.measure(0, classical_register_index) # Measure the qubit

    classical_regs = [0]
    trials = 10
    print(quantum_simulator.run(p, classical_regs, trials))


.. parsed-literal::

    [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]


Some Program Construction Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple instructions can be applied at once or chained together. The
following are all valid programs:

.. code:: python

    print("Multiple inst arguments with final measurement:")
    print(Program().inst(X(0), Y(1), Z(0)).measure(0, 1))

    print("Chained inst with explicit MEASURE instruction:")
    print(Program().inst(X(0)).inst(Y(1)).measure(0, 1).inst(MEASURE(1, 2)))

    print("A mix of chained inst and measures:")
    print(Program().inst(X(0)).measure(0, 1).inst(Y(1), X(0)).measure(0, 0))

    print("A composition of two programs:")
    print(Program(X(0)) + Program(Y(0)))

.. parsed-literal::

    Multiple inst arguments with final measurement:
    X 0
    Y 1
    Z 0
    MEASURE 0 [1]

    Chained inst with explicit MEASURE instruction:
    X 0
    Y 1
    MEASURE 0 [1]
    MEASURE 1 [2]

    A mix of chained inst and measures:
    X 0
    MEASURE 0 [1]
    Y 1
    X 0
    MEASURE 0 [0]

    A composition of two programs:
    X 0
    Y 0

Fixing a Mistaken Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an instruction was appended to a program incorrectly, one can pop it
off.

.. code:: python

    p = Program().inst(X(0))
    p.inst(Y(1))
    print("Oops! We have added Y 1 by accident:")
    print(p)

    print("We can fix by popping:")
    p.pop()
    print(p)

    print("And then add it back:")
    p += Program(Y(1))
    print(p)

.. parsed-literal::

    Oops! We have added Y 1 by accident:
    X 0
    Y 1

    We can fix by popping:
    X 0

    And then add it back:
    X 0
    Y 1


Connections
~~~~~~~~~~~

Larger pyQuil programs can involve more qubits and take a longer time to run. Instead of running the
program immediately, you can insert your programs into a queue. This is done with the ``use_queue``
parameter to QVMConnection.  By default, this parameter is set to False which means it skips
the queue and runs it immediately. However, the QVM will reject programs that are more than
19 qubits or take longer than 10 seconds to run. Therefore, to run programs of a larger size you must
set the ``use_queue`` parameter to True which has more overhead.

.. code:: python

  from pyquil.quil import Program
  from pyquil.api import QVMConnection

  qvm = QVMConnection(use_queue=True)
  qvm.run(Program(X(0).measure(0, 0), [0])

The Forest queue also allows an asynchronous mode of interaction with methods postfixed with `_async`.
This means that there is a seperate query to post a job and to get the result.

::

  from pyquil.quil import Program
  from pyquil.gates import X, H, I
  from pyquil.api import QVMConnection

  qvm = QVMConnection()
  job_id = qvm.run_async(Program(X(0)).measure(0, 0), [0])

The `job_id` is a string that uniquely identifies the job in Forest. You can use the
`.get_job` method on QVMConnection to get the current status.

::

  job = qvm.get_job(job_id)
  if not job.is_done():
    time.sleep(1)
    job = qvm.get_job(job_id)
  print(job.result())

.. parsed-literal::

  [[1]]

The `wait_for_job` method periodically checks for updates and prints the job's position
in the queue, similar to the above code.

::

  job = qvm.wait_for_job(job_id)
  print(job.result())

.. parsed-literal::

  [[1]]


Optimized Calls
~~~~~~~~~~~~~~~

This same pattern as above applies to the :meth:`~pyquil.api.QVMConnection.wavefunction`,
:meth:`~pyquil.api.QVMConnection.expectation` and :meth:`~pyquil.api.QVMConnection.run_and_measure`.
These are very useful if used appropriately: They all execute a given program *once and only once*
and then either return the final wavefunction or use it to generate expectation values or a
specified number of random bitstring samples.

.. warning::

    This behavior can have unexpected consequences if the program that prepares the final state
    is non-deterministic, e.g., if it contains measurements and/or noisy gate applications.
    In this case, the final state after the program execution is itself a random variable
    and a single call to these functions therefore **cannot** sample the full space of outcomes.
    Therefore, if the program is non-deterministic and sampling the full program output distribution
    is important for the application at hand, we recommend using the basic
    :meth:`~pyquil.api.QVMConnection.run` API function as this re-runs the full program for every
    requested trial.


Basic Exercises
---------------

Basic Exercise 1: Quantum Dice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a quantum program to simulate throwing an 8-sided die. The Python
function you should produce is:

::

    def throw_octahedral_die():
        # return the result of throwing an 8 sided die, an int between 1 and 8, by running a quantum program

Next, extend the program to work for any kind of fair die:

::

    def throw_polyhedral_die(num_sides):
        # return the result of throwing a num_sides sided die by running a quantum program

Basic Exercise 2: Controlled Gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the full generality of NumPy to construct new gate matrices.

1. Write a function ``controlled`` which takes a :math:`2\times 2`
   matrix :math:`U` representing a single qubit operator, and makes a
   :math:`4\times 4` matrix which is a controlled variant of :math:`U`,
   with the first argument being the *control qubit*.

2. Write a Quil program to define a controlled-\ :math:`Y` gate in this
   manner. Find the wavefunction when applying this gate to qubit 1
   controlled by qubit 0.