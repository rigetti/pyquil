.. _advanced_usage:

Advanced Usage
==============

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

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import *
    qvm = get_qc('9q-square-qvm')


Now that our local endpoints are up and running, we can start running pyQuil programs! Open a jupyter notebook (type
..code::`jupyter notebook` in your terminal), or launch python in your terminal (type ..code::`python3`).

Quantum Fourier Transform (QFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us do an example that includes multi-qubit parameterized gates.

Here we wish to compute the discrete Fourier transform of
``[0, 1, 0, 0, 0, 0, 0, 0]``. We do this in three steps:

1. Write a function called ``qft3`` to make a 3-qubit QFT quantum
   program.
2. Write a state preparation quantum program.
3. Execute state preparation followed by the QFT on the QVM.

First we define a function to make a 3-qubit QFT quantum program. This
is a mix of Hadamard and CPHASE gates, with a final bit reversal
correction at the end consisting of a single SWAP gate.

.. code:: python

    from math import pi

    def qft3(q0, q1, q2):
        p = Program()
        p.inst( H(q2),
                CPHASE(pi/2.0, q1, q2),
                H(q1),
                CPHASE(pi/4.0, q0, q2),
                CPHASE(pi/2.0, q0, q1),
                H(q0),
                SWAP(q0, q2) )
        return p

There is a very important detail to recognize here: The function
``qft3`` doesn't *compute* the QFT, but rather it *makes a quantum
program* to compute the QFT on qubits ``q0``, ``q1``, and ``q2``.

We can see what this program looks like in Quil notation by doing
the following:

.. code:: python

    print(qft3(0, 1, 2))

.. parsed-literal::

    H 2
    CPHASE(1.5707963267948966) 1 2
    H 1
    CPHASE(0.7853981633974483) 0 2
    CPHASE(1.5707963267948966) 0 1
    H 0
    SWAP 0 2

Next, we want to prepare a state that corresponds to the sequence we
want to compute the discrete Fourier transform of. Fortunately, this is
easy, we just apply an :math:`X`-gate to the zeroth qubit.

.. code:: python

    state_prep = Program().inst(X(0))

We can verify that this works by computing its wavefunction. However, we
need to add some "dummy" qubits, because otherwise ``wavefunction``
would return a two-element vector.

.. code:: python

    add_dummy_qubits = Program().inst(I(1), I(2))
    wavefunction = qvm.wavefunction(state_prep + add_dummy_qubits)
    print(wavefunction)

.. parsed-literal::

    (1+0j)|001>

If we have two quantum programs ``a`` and ``b``, we can concatenate them
by doing ``a + b``. Using this, all we need to do is compute the QFT
after state preparation to get our final result.

.. code:: python

    wavefunction = qvm.wavefunction(state_prep + qft3(0, 1, 2))
    print(wavefunction.amplitudes)

.. parsed-literal::

    array([  3.53553391e-01+0.j        ,   2.50000000e-01+0.25j      ,
             2.16489014e-17+0.35355339j,  -2.50000000e-01+0.25j      ,
            -3.53553391e-01+0.j        ,  -2.50000000e-01-0.25j      ,
            -2.16489014e-17-0.35355339j,   2.50000000e-01-0.25j      ])

We can verify this works by computing the (inverse) FFT from NumPy.

.. code:: python

    from numpy.fft import ifft
    ifft([0,1,0,0,0,0,0,0], norm="ortho")

.. parsed-literal::

    array([ 0.35355339+0.j        ,  0.25000000+0.25j      ,
            0.00000000+0.35355339j, -0.25000000+0.25j      ,
           -0.35355339+0.j        , -0.25000000-0.25j      ,
            0.00000000-0.35355339j,  0.25000000-0.25j      ])

Classical Control Flow
~~~~~~~~~~~~~~~~~~~~~~

Here are a couple quick examples that show how much richer the classical
control of a Quil program can be. In this first example, we have a
register called ``classical_flag_register`` which we use for looping.
Then we construct the loop in the following steps:

1. We first initialize this register to ``1`` with the ``init_register``
   program so our while loop will execute. This is often called the
   *loop preamble* or *loop initialization*.

2. Next, we write body of the loop in a program itself. This will be a
   program that computes an :math:`X` followed by an :math:`H` on our
   qubit.

3. Lastly, we put it all together using the ``while_do`` method.

.. code:: python

    # Name our classical registers:
    classical_flag_register = 2

    # Write out the loop initialization and body programs:
    init_register = Program(TRUE([classical_flag_register]))
    loop_body = Program(X(0), H(0)).measure(0, classical_flag_register)

    # Put it all together in a loop program:
    loop_prog = init_register.while_do(classical_flag_register, loop_body)

    print(loop_prog)

.. parsed-literal::

    TRUE [2]
    LABEL @START1
    JUMP-UNLESS @END2 [2]
    X 0
    H 0
    MEASURE 0 [2]
    JUMP @START1
    LABEL @END2

Notice that the ``init_register`` program applied a Quil instruction directly to a
classical register.  There are several classical commands that can be used in this fashion:

- ``TRUE`` which sets a single classical bit to be 1
- ``FALSE`` which sets a single classical bit to be 0
- ``NOT`` which flips a classical bit
- ``AND`` which operates on two classical bits
- ``OR`` which operates on two classical bits
- ``MOVE`` which moves the value of a classical bit at one classical address into another
- ``EXCHANGE`` which swaps the value of two classical bits

In this next example, we show how to do conditional branching in the
form of the traditional ``if`` construct as in many programming
languages. Much like the last example, we construct programs for each
branch of the ``if``, and put it all together by using the ``if_then``
method.

.. code:: python

    # Name our classical registers:
    test_register = 1
    answer_register = 0

    # Construct each branch of our if-statement. We can have empty branches
    # simply by having empty programs.
    then_branch = Program(X(0))
    else_branch = Program()

    # Make a program that will put a 0 or 1 in test_register with 50% probability:
    branching_prog = Program(H(1)).measure(1, test_register)

    # Add the conditional branching:
    branching_prog.if_then(test_register, then_branch, else_branch)

    # Measure qubit 0 into our answer register:
    branching_prog.measure(0, answer_register)

    print(branching_prog)

.. parsed-literal::

    H 1
    MEASURE 1 [1]
    JUMP-WHEN @THEN3 [1]
    JUMP @END4
    LABEL @THEN3
    X 0
    LABEL @END4
    MEASURE 0 [0]

We can run this program a few times to see what we get in the
``answer_register``.

.. code:: python

    qvm.run(branching_prog, [answer_register], 10)

.. parsed-literal::

    [[1], [1], [1], [0], [1], [0], [0], [1], [1], [0]]

Parametric Depolarizing Noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rigetti QVM has support for emulating certain types of noise models.
One such model is *parametric Pauli noise*, which is defined by a
set of 6 probabilities:

-  The probabilities :math:`P_X`, :math:`P_Y`, and :math:`P_Z` which
   define respectively the probability of a Pauli :math:`X`, :math:`Y`,
   or :math:`Z` gate getting applied to *each* qubit after *every* gate
   application. These probabilities are called the *gate noise
   probabilities*.

-  The probabilities :math:`P_X'`, :math:`P_Y'`, and :math:`P_Z'` which
   define respectively the probability of a Pauli :math:`X`, :math:`Y`,
   or :math:`Z` gate getting applied to the qubit being measured
   *before* it is measured. These probabilities are called the
   *measurement noise probabilities*.

We can instantiate a noisy QVM by creating a new connection with these
probabilities specified.

.. code:: python

    # 20% chance of a X gate being applied after gate applications and before measurements.
    gate_noise_probs = [0.2, 0.0, 0.0]
    meas_noise_probs = [0.2, 0.0, 0.0]
    noisy_qvm = qvm(gate_noise=gate_noise_probs, measurement_noise=meas_noise_probs)

We can test this by applying an :math:`X`-gate and measuring. Nominally,
we should always measure ``1``.

.. code:: python

    p = Program().inst(X(0)).measure(0, 0)
    print("Without Noise: {}".format(qvm.run(p, [0], 10)))
    print("With Noise   : {}".format(noisy_qvm.run(p, [0], 10)))

.. parsed-literal::

    Without Noise: [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    With Noise   : [[0], [0], [0], [0], [0], [1], [1], [1], [1], [0]]

Parametric Programs
~~~~~~~~~~~~~~~~~~~

In PyQuil 1.x, there was an object named ``ParametricProgram``::

    # This function returns a quantum circuit with different rotation angles on a gate on qubit 0
    def rotator(angle):
        return Program(RX(angle, 0))

    from pyquil.parametric import ParametricProgram
    par_p = ParametricProgram(rotator) # This produces a new type of parameterized program object

Please consider simply using a Python function for the above functionality::

    par_p = rotator

Or using declared classical memory::

    p = Program()
    angle = p.declare('angle', 'REAL')
    p += RX(angle, 0)

Pauli Operator Algebra
~~~~~~~~~~~~~~~~~~~~~~

Many algorithms require manipulating sums of Pauli combinations, such as
:math:`\sigma = \frac{1}{2}I - \frac{3}{4}X_0Y_1Z_3 + (5-2i)Z_1X_2,` where
:math:`G_n` indicates the gate :math:`G` acting on qubit :math:`n`. We
can represent such sums by constructing ``PauliTerm`` and ``PauliSum``.
The above sum can be constructed as follows:

.. code:: python

    from pyquil.paulis import ID, sX, sY, sZ

    # Pauli term takes an operator "X", "Y", "Z", or "I"; a qubit to act on, and
    # an optional coefficient.
    a = 0.5 * ID
    b = -0.75 * sX(0) * sY(1) * sZ(3)
    c = (5-2j) * sZ(1) * sX(2)

    # Construct a sum of Pauli terms.
    sigma = a + b + c
    print("sigma = {}".format(sigma))

.. parsed-literal::

    sigma = 0.5*I + -0.75*X0*Y1*Z3 + (5-2j)*Z1*X2

Right now, the primary thing one can do with Pauli terms and sums is to construct the
exponential of the Pauli term, i.e., :math:`\exp[-i\beta\sigma]`.  This is
accomplished by constructing a parameterized Quil program that is evaluated
when passed values for the coefficients of the angle :math:`\beta`.

Related to exponentiating Pauli sums we provide utility functions for finding
the commuting subgroups of a Pauli sum and approximating the exponential with the
Suzuki-Trotter approximation through fourth order.

When arithmetic is done with Pauli sums, simplification is automatically
done.

The following shows an instructive example of all three.

.. code:: python

    import pyquil.paulis as pl

    # Simplification
    sigma_cubed = sigma * sigma * sigma
    print("Simplified  : {}".format(sigma_cubed))
    print()

    #Produce Quil code to compute exp[iX]
    H = -1.0 * sX(0)
    print("Quil to compute exp[iX] on qubit 0:")
    print(pl.exponential_map(H)(1.0))

.. parsed-literal::

    Simplified  : (32.46875-30j)*I + (-16.734375+15j)*X0*Y1*Z3 + (71.5625-144.625j)*Z1*X2

    Quil to compute exp[iX] on qubit 0:
    H 0
    RZ(-2.0) 0
    H 0

``exponential_map`` returns a function allowing you to fill in a multiplicative
constant later. This commonly occurs in variational algorithms. The function
``exponential_map`` is used to compute exp[-i * alpha * H] without explicitly filling in a
value for alpha.

.. code:: python

    expH = pl.exponential_map(H)
    print(expH(0.0))
    print(expH(1.0))
    print(expH(2.0))

