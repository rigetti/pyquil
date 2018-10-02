.. _basics:

The Basics: Programs and Gates
==============================

First, initialize a localQVM instance on your laptop. You should have two consoles open in your terminal to run in the
background.

.. code:: python

    ### CONSOLE 1
    $ quilc -S
    port triggered: 6000.
    [2018-09-19 11:22:37] Starting server: 0.0.0.0 : 6000.

    ### CONSOLE 2
    $ qvm -S
    Welcome to the Rigetti QVM
    (Configured with 2048 MiB of workspace and 8 workers.)
    [2018-09-20 15:39:50] Starting server on port 5000.

Quantum programs are written in Forest using the ``Program`` object from the ``quil`` module.

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import *

Programs are then constructed from quantum gates, which can be found in the ``gates`` module. We can
add quantum gates to programs in numerous ways, including using the ``.inst(...)`` method. We use
the ``.measure(...)`` method to measure qubits into classical registers:

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
program on the Quantum Virtual Machine, or QVM:

.. code:: python

    qvm = get_qc('9q-generic-qvm')
    result = qvm.run_and_measure(p, trials=10)
    print(result)

Congratulations! You just ran a program on the QVM. The returned value should be:

.. parsed-literal::

    [[1]]

For more information on what the above result means, and on executing quantum programs on the QVM in
general, see :ref:`qvm`. Feel free to skip ahead and read about executing programs on the QVM
(and the QPU for that matter), but don't forget to come back. The remainder of this section of the
docs will be dedicated to constructing programs in detail, an essential part of becoming fluent in
quantum programming.

Some Program Construction Features
----------------------------------

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

PyQuil can also produce a Program object by interpreting raw Quil text, as in
the following example:

.. code:: python

    print(Program("X 0\nH 1\nCNOT 0 1"))

.. parsed-literal::

    X 0
    H 1
    CNOT 0 1

The ``pyquil.parser`` submodule provides a front-end to other similar parser
functionality.


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
