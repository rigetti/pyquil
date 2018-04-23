.. _basics:

The Basics: Programs and Gates
==============================

To ensure that your installation is working correctly, try running the
following Python commands interactively. First, import the ``quil``
module (which constructs quantum programs) and the ``api`` module (which
allows connections to the Rigetti QVM). We will also import some basic
gates for pyQuil as well as numpy.

.. code:: python

    from pyquil.quil import Program
    from pyquil.api import QVMConnection
    from pyquil.gates import *
    import numpy as np

Next, we want to open a connection to the QVM.

.. code:: python

    qvm = QVMConnection()

Now we can make a program by adding some Quil instruction using the
``inst`` method on a ``Program`` object.

.. code:: python

    p = Program()
    p.inst(X(0)).measure(0, 0)

.. parsed-literal::

    <pyquil.quil.Program at 0x101d45a90>


This program simply applies the :math:`X`-gate to the zeroth qubit,
measures that qubit, and stores the measurement result in the zeroth
classical register. We can look at the Quil code that makes up this
program simply by printing it.

.. code:: python

    print(p)

.. parsed-literal::

    X 0
    MEASURE 0 [0]

Most importantly, of course, we can see what happens if we run this
program on the QVM:

.. code:: python

    classical_regs = [0] # A list of which classical registers to return the values of.

    qvm.run(p, classical_regs)

.. parsed-literal::

    [[1]]

We see that the result of this program is that the classical register
``[0]`` now stores the state of qubit 0, which should be
:math:`\left\vert 1\right\rangle` after an :math:`X`-gate. We can of
course ask for more classical registers:

.. code:: python

    qvm.run(p, [0, 1, 2])

.. parsed-literal::

    [[1, 0, 0]]

The classical registers are initialized to zero, so registers ``[1]``
and ``[2]`` come out as zero. If we stored the measurement in a
different classical register we would obtain:

.. code:: python

    p = Program()   # clear the old program
    p.inst(X(0)).measure(0, 1)
    qvm.run(p, [0, 1, 2])

.. parsed-literal::

    [[0, 1, 0]]

We can also run programs multiple times and accumulate all the results
in a single list.

.. code:: python

    coin_flip = Program().inst(H(0)).measure(0, 0)
    num_flips = 5
    qvm.run(coin_flip, [0], num_flips)

.. parsed-literal::

    [[0], [1], [0], [1], [0]]

Try running the above code several times. You will see that you will,
with very high probability, get different results each time.

As the QVM is a virtual machine, we can also inspect the wavefunction of
a program directly, even without measurements:

.. code:: python

    coin_flip = Program().inst(H(0))
    qvm.wavefunction(coin_flip)

.. parsed-literal::

    <pyquil.wavefunction.Wavefunction at 0x1088a2c10>

The return value is a Wavefunction object that stores the amplitudes of the
quantum state at the conclusion of the program. We can print this object

.. code:: python

    coin_flip = Program().inst(H(0))
    wavefunction = qvm.wavefunction(coin_flip)
    print(wavefunction)

.. parsed-literal::

  (0.7071067812+0j)|0> + (0.7071067812+0j)|1>

To see the amplitudes listed as a sum of computational basis states. We can index into those
amplitudes directly or look at a dictionary of associated outcome probabilities.

.. code:: python

  assert wavefunction[0] == 1 / np.sqrt(2)
  # The amplitudes are stored as a numpy array on the Wavefunction object
  print(wavefunction.amplitudes)
  prob_dict = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes as a dict
  print(prob_dict)
  prob_dict.keys() # these stores the bitstring outcomes
  assert len(wavefunction) == 1 # gives the number of qubits

.. parsed-literal::

  [ 0.70710678+0.j  0.70710678+0.j]
  {'1': 0.49999999999999989, '0': 0.49999999999999989}

The result from a wavefunction call also contains an optional amount of classical memory to
check:

.. code:: python

    coin_flip = Program().inst(H(0)).measure(0,0)
    wavefunction = qvm.wavefunction(coin_flip, classical_addresses=range(9))
    classical_mem = wavefunction.classical_memory

Additionally, we can pass a random seed to the Connection object. This allows us to reliably
reproduce measurement results for the purpose of testing:

.. code:: python

    seeded_cxn = api.QVMConnection(random_seed=17)
    print(seeded_cxn.run(Program(H(0)).measure(0, 0), [0], 20))

    seeded_cxn = api.QVMConnection(random_seed=17)
    # This will give identical output to the above
    print(seeded_cxn.run(Program(H(0)).measure(0, 0), [0], 20))

It is important to remember that this ``wavefunction`` method is just a useful debugging tool
for small quantum systems, and it cannot be feasibly obtained on a
quantum processor.

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
-----------------------------

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

The Standard Gate Set
---------------------

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
------------------

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
-------------------------

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
