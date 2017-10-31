
Installation and Getting Started
================================

This toolkit provides some simple libraries for writing quantum
programs.

.. code:: python

    import pyquil.quil as pq
    import pyquil.api as api
    from pyquil.gates import *
    qvm = api.SyncConnection()
    p = pq.Program()
    p.inst(H(0), CNOT(0, 1))
        <pyquil.pyquil.Program object at 0x101ebfb50>
    wvf, _ = qvm.wavefunction(p)
    print(wvf)
        (0.7071067812+0j)|00> + (0.7071067812+0j)|11>

It comes with a few parts:

1. **Quil**: The Quantum Instruction Language standard. Instructions
   written in Quil can be executed on any implementation of a quantum
   abstract machine, such as the quantum virtual machine (QVM), or on a
   real quantum processing unit (QPU). More details regarding Quil can be
   found in the `whitepaper <https://arxiv.org/abs/1608.03355>`__.
2. **QVM**: A `Quantum Virtual Machine <qvm_overview.html>`_, which is an implementation of the
   quantum abstract machine on classical hardware. The QVM lets you use a
   regular computer to simulate a small quantum computer. You can access
   the Rigetti QVM running in the cloud with your API key.
   `Sign up here <http://forest.rigetti.com>`_ to get your key.
3. **pyQuil**: A Python library to help write and run Quil code and
   quantum programs.
4. **QPUConnection**: pyQuil also includes some a special connection which lets you run experiments
   on Rigetti's prototype superconducting quantum processors over the cloud.  These experiments are
   described in more detail `here <qpu.html>`_.

Environment Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

Before you can start writing quantum programs, you will need Python 2.7
(version 2.7.10 or greater) or Python 3.6 and the
Python package manager pip.

.. note::
PyQuil works on both Python 2 and 3. However, Rigetti **strongly** recommends
using Python 3 if possible. Future feature developments in PyQuil may support
Python 3 only.


Installation
~~~~~~~~~~~~

You can install pyQuil directly from the Python package manager pip using:

::

    pip install pyquil

To instead install the bleeding-edge version from source, clone the
`pyquil GitHub repository <https://github.com/rigetticomputing/pyquil>`_,
navigate into its directory in a terminal, and run:

::

    pip install -e .

On Mac/Linux, if this command does not succeed because of permissions
errors, then instead run:

::

    sudo pip install -e .

This will also install pyQuil's dependencies (numpy, requests, etc.) if you do not already
have them.

The library will now be available globally.

Connecting to the Rigetti Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pyQuil can be used to build and manipulate Quil programs without restriction. However, to run
programs (e.g., to get wavefunctions, get multishot experiment data), you will need an API key
for Rigetti Forest. This will allow you to run your programs on the Rigetti QVM or QPU.

`Sign up here <http://forest.rigetti.com>`_ to get a Forest API key, it's free and only takes a few seconds.

Run the following command to automatically set up the config. This will prompt you for the required information
(URL, key, and user id). It will then create a file in the proper location (the user's root directory):

::

    pyquil-config-setup

If the setup completed successfully then you can skip to the next section.

You can also create the configuration file manually if you'd like and place it at ``~/.pyquil_config``.
The configuration file is in INI format and should contain all the information required to connect to Forest:

::

    [Rigetti Forest]
    key: <Rigetti Forest API key>
    user_id: <Rigetti User ID>

Alternatively, you can place the file at your own chosen location and then set the ``PYQUIL_CONFIG`` environment
variable to the path of the file.

.. note::
  You may specify an absolute path or use the ~ to indicate your home directory.
  On Linux, this points to ``/users/username``.
  On Mac, this points to ``/Users/Username``.
  On Windows, this points to ``C:\Users\Username``

.. note::
  Windows users may find it easier to name the file ``pyquil.ini`` and open it using notepad. Then, set the
  ``PYQUIL_CONFIG`` environment variable by opening up a command prompt and running:
  ``setenv PYQUIL_CONFIG=C:\Users\Username\pyquil.ini``

As a last resort, connection information can be provided via environment variables.

::

    export QVM_API_KEY=<Rigetti Forest API key>
    export QVM_USER_ID=<Rigetti User ID>

If you are still seeing errors or warnings then file a bug using
`Github Issues <https://github.com/rigetticomputing/pyquil/issues>`_.

Endpoints
+++++++++
There are two important endpoints to keep in mind.  You will use different ones for different types
of jobs.

``https://api.rigetti.com/qvm`` is used for making synchronous calls to the QVM.  You should use
this for most of the getting started materials unless otherwise instructed.

``https://job.rigetti.com/beta`` is used for large async `QVM jobs <getting_started.html#jobconnections>`_
or for running `jobs on a QPU <qpu.html>`_.


Running your first quantum program
----------------------------------
pyQuil is a Python library that helps you write programs in the Quantum Instruction Language (Quil).
It also ships with a simple script ``examples/run_quil.py`` that runs Quil code directly. You can
test your connection to Forest using this script by executing the following on your command line

::

    cd examples/
    python run_quil.py hello_world.quil

You should see the following output array ``[[1, 0, 0, 0, 0, 0, 0, 0]]``. This indicates that you have
a good connection to our API.

You can continue to write more Quil code in files and run them using the ``run_quil.py`` script. The
following sections describe how to use the pyQuil library directly to build quantum programs in
Python.

Basic pyQuil Usage
------------------

To ensure that your installation is working correctly, try running the
following Python commands interactively. First, import the ``quil``
module (which constructs quantum programs) and the ``api`` module (which
allows connections to the Rigetti QVM). We will also import some basic
gates for pyQuil as well as numpy.

.. code:: python

    import pyquil.quil as pq
    import pyquil.api as api
    from pyquil.gates import *
    import numpy as np

Next, we want to open a connection to the QVM. Forest supports two types of connections through
pyQuil.  The first is a synchronous connection that immediately runs requested jobs against the QVM.
This will time out on longer jobs that run for more than 30 seconds. Synchronous connections are good
for experimenting interactively as they give quick feedback.

.. code:: python

    # open a synchronous connection
    qvm = api.SyncConnection()

Now we can make a program by adding some Quil instruction using the
``inst`` method on a ``Program`` object.

.. code:: python

    p = pq.Program()
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

    p = pq.Program()   # clear the old program
    p.inst(X(0)).measure(0, 1)
    qvm.run(p, [0, 1, 2])




.. parsed-literal::

    [[0, 1, 0]]



We can also run programs multiple times and accumulate all the results
in a single list.

.. code:: python

    coin_flip = pq.Program().inst(H(0)).measure(0, 0)
    num_flips = 5
    qvm.run(coin_flip, [0], num_flips)




.. parsed-literal::

    [[0], [1], [0], [1], [0]]



Try running the above code several times. You will see that you will,
with very high probability, get different results each time.

As the QVM is a virtual machine, we can also inspect the wavefunction of
a program directly, even without measurements:

.. code:: python

    coin_flip = pq.Program().inst(H(0))
    qvm.wavefunction(coin_flip)




.. parsed-literal::

    (<pyquil.wavefunction.Wavefunction at 0x1088a2c10>, [])


The first element in the returned tuple is a Wavefunction object that stores the amplitudes of the
quantum state at the conclusion of the program. We can print this object

.. code:: python

    coin_flip = pq.Program().inst(H(0))
    wvf, _ = qvm.wavefunction(coin_flip)
    print(wvf)

.. parsed-literal::

  (0.7071067812+0j)|0> + (0.7071067812+0j)|1>

To see the amplitudes listed as a sum of computational basis states. We can index into those
amplitudes directly or look at a dictionary of associated outcome probabilities.

.. code:: python

  assert wvf[0] == 1 / np.sqrt(2)
  # The amplitudes are stored as a numpy array on the Wavefunction object
  print(wvf.amplitudes)
  prob_dict = wvf.get_outcome_probs() # extracts the probabilities of outcomes as a dict
  print(prob_dict)
  prob_dict.keys() # these stores the bitstring outcomes
  assert len(wvf) == 1 # gives the number of qubits

.. parsed-literal::

  [ 0.70710678+0.j  0.70710678+0.j]
  {'1': 0.49999999999999989, '0': 0.49999999999999989}

The second element returned from a wavefunction call is an optional amount of classical memory to
check:

.. code:: python

    coin_flip = pq.Program().inst(H(0)).measure(0,0)
    wavf, classical_mem = qvm.wavefunction(coin_flip, classical_addresses=range(9))


Additionally, we can pass a random seed to the Connection object. This allows us to reliably
reproduce measurement results for the purpose of testing:

.. code:: python

    seeded_cxn = api.SyncConnection(random_seed=17)
    print(seeded_cxn.run(pq.Program(H(0)).measure(0, 0), [0], 20))

    seeded_cxn = api.SyncConnection(random_seed=17)
    # This will give identical output to the above
    print(seeded_cxn.run(pq.Program(H(0)).measure(0, 0), [0], 20))


It is important to remember that this ``wavefunction`` method is just a useful debugging tool
for small quantum systems, and it cannot be feasibly obtained on a
quantum processor.

Some Program Construction Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple instructions can be applied at once or chained together. The
following are all valid programs:

.. code:: python

    print("Multiple inst arguments with final measurement:")
    print(pq.Program().inst(X(0), Y(1), Z(0)).measure(0, 1))
    
    print("Chained inst with explicit MEASURE instruction:")
    print(pq.Program().inst(X(0)).inst(Y(1)).measure(0, 1).inst(MEASURE(1, 2)))
    
    print("A mix of chained inst and measures:")
    print(pq.Program().inst(X(0)).measure(0, 1).inst(Y(1), X(0)).measure(0, 0))
    
    print("A composition of two programs:")
    print(pq.Program(X(0)) + pq.Program(Y(0)))


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

    p = pq.Program().inst(X(0))
    p.inst(Y(1))
    print("Oops! We have added Y 1 by accident:")
    print(p)
    
    print("We can fix by popping:")
    p.pop()
    print(p)
    
    print("And then add it back:")
    p += pq.Program(Y(1))
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
    p = pq.Program().defgate("SQRT-X", sqrt_x)
    
    # Then we can use the new gate,
    p.inst(("SQRT-X", 0))
    print(p)


.. parsed-literal::

    DEFGATE SQRT-X:
        0.5+0.5i, 0.5-0.5i
        0.5-0.5i, 0.5+0.5i
    
    SQRT-X 0
    



.. code:: python

    print(qvm.wavefunction(p)[0])




.. parsed-literal::

    (0.5+0.5j)|0> + (0.5-0.5j)|1>



Quil in general supports defining parametric gates, though right now
only static gates are supported by pyQuil. Below we show how we can
define :math:`X_0\otimes \sqrt{X_1}` as a single
gate.

.. code:: python

    # A multi-qubit defgate example
    x_gate_matrix = np.array(([0.0, 1.0], [1.0, 0.0]))
    sqrt_x = np.array([[ 0.5+0.5j,  0.5-0.5j],
                    [ 0.5-0.5j,  0.5+0.5j]])
    x_sqrt_x = np.kron(x_gate_matrix, sqrt_x)
    p = pq.Program().defgate("X-SQRT-X", x_sqrt_x)
    
    # Then we can use the new gate
    p.inst(("X-SQRT-X", 0, 1))
    wavf, _ = qvm.wavefunction(p)
    print(wavf)




.. parsed-literal::

    (0.5+0.5j)|01> + (0.5-0.5j)|11>


Advanced Usage
--------------

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
        p = pq.Program()
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

    state_prep = pq.Program().inst(X(0))

We can verify that this works by computing its wavefunction. However, we
need to add some "dummy" qubits, because otherwise ``wavefunction``
would return a two-element vector.

.. code:: python

    add_dummy_qubits = pq.Program().inst(I(1), I(2))
    wavf, _ = qvm.wavefunction(state_prep + add_dummy_qubits)
    print(wavf)



.. parsed-literal::

    (1+0j)|001>



If we have two quantum programs ``a`` and ``b``, we can concatenate them
by doing ``a + b``. Using this, all we need to do is compute the QFT
after state preparation to get our final result.

.. code:: python

    wavf, _ = qvm.wavefunction(state_prep + qft3(0, 1, 2))
    print(wavf.amplitudes)



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
    init_register = pq.Program(TRUE([classical_flag_register]))
    loop_body = pq.Program(X(0), H(0)).measure(0, classical_flag_register)
    
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
    then_branch = pq.Program(X(0))
    else_branch = pq.Program()
    
    # Make a program that will put a 0 or 1 in test_register with 50% probability:
    branching_prog = pq.Program(H(1)).measure(1, test_register)
    
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
    noisy_qvm = api.SyncConnection(gate_noise=gate_noise_probs, measurement_noise=meas_noise_probs)

We can test this by applying an :math:`X`-gate and measuring. Nominally,
we should always measure ``1``.

.. code:: python

    p = pq.Program().inst(X(0)).measure(0, 0)
    print("Without Noise: {}".format(qvm.run(p, [0], 10)))
    print("With Noise   : {}".format(noisy_qvm.run(p, [0], 10)))


.. parsed-literal::

    Without Noise: [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    With Noise   : [[0], [0], [0], [0], [0], [1], [1], [1], [1], [0]]


Parametric Programs
~~~~~~~~~~~~~~~~~~~

A big advantage of working in pyQuil is that you are able to leverage all the functionality of
Python to generate Quil programs.  In quantum/classical hybrid algorithms this often leads to
situations where complex classical functions are used to generate Quil programs. pyQuil provides
a convenient construction to allow you to use Python functions to generate templates of Quil
programs, called ``ParametricPrograms``:

.. code:: python

    # This function returns a quantum circuit with different rotation angles on a gate on qubit 0
    def rotator(angle):
        return pq.Program(RX(angle, 0))
    
    from pyquil.parametric import ParametricProgram
    par_p = ParametricProgram(rotator) # This produces a new type of parameterized program object

The parametric program ``par_p`` now takes the same arguments as ``rotator``:

.. code:: python

    print(par_p(0.5))

.. parsed-literal::

    RX(0.5) 0

We can think of ``ParametricPrograms`` as a sort of template for Quil programs.  They cache computations
that happen in Python functions so that templates in Quil can be efficiently substituted.


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
    
A more sophisticated feature of pyQuil is that it can create templates of Quil programs in
ParametricProgram objects.  An example use of these templates is in exponentiating a Hamiltonian
that is parametrized by a constant.  This commonly occurs in variational algorithms. The function
``exponential_map`` is used to compute exp[i * alpha * H] without explicitly filling in a value for
alpha.

.. code:: python

    parametric_prog = pl.exponential_map(H)
    print(parametric_prog(0.0))
    print(parametric_prog(1.0))
    print(parametric_prog(2.0))

This ParametricProgram now acts as a template, caching the result of the ``exponential_map``
calculation so that it can be used later with new values.

JobConnections
--------------
Larger pyQuil programs can take longer than 30 seconds to run.  These jobs can be posted into the
cloud job queue using a different connection object.  The mode of interaction with the API is
asynchronous.  This means that there is a seperate query to post a job and to get the result.

::

  from pyquil.quil import Program
  from pyquil.gates import X, H, I
  from pyquil.api import JobConnection

  job_qvm = JobConnection(endpoint="https://job.rigetti.com/beta")
  res = job_qvm.run(Program(X(0)).measure(0, 0), [0])

The `res` is an instance of a ``JobResult`` object.  It has an id and allows you to make queries
to see if the job result is finished.

::

  zz = res.get()
  print(type(zz), zz)

.. parsed-literal::

    <class 'pyquil.job_results.JobResult'> {u'status': u'QUEUED', u'jobId': u'BLSLJCBGNP'}

`is_done` updates the ``JobResult`` object once, and returns `True` if the job has completed. 
Once the job is finished, then the results can be retrieved from the JobResult object:

::

  import time

  while not res.is_done():
      time.sleep(1)
  print(res)
  answer = res.decode()
  print(answer)

.. parsed-literal::

  {u'result': u'[[1]]', u'jobId': u'BLSLJCBGNP'}

  <type 'list'> [[1]]

This same pattern applies to the ``wavefunction``, ``expectation``, and ``run_and_measure`` calls
on the JobConnection object.

Exercises
---------

Exercise 1 - Quantum Dice
~~~~~~~~~~~~~~~~~~~~~~~~~

Write a quantum program to simulate throwing an 8-sided die. The Python
function you should produce is:

::

    def throw_octahedral_die():
        # return the result of throwing an 8 sided die, an int between 1 and 8, by running a quantum program

Next, extend the program to work for any kind of fair die:

::

    def throw_polyhedral_die(num_sides):
        # return the result of throwing a num_sides sided die by running a quantum program

Exercise 2 - Controlled Gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the full generality of NumPy to construct new gate matrices.

1. Write a function ``controlled`` which takes a :math:`2\times 2`
   matrix :math:`U` representing a single qubit operator, and makes a
   :math:`4\times 4` matrix which is a controlled variant of :math:`U`,
   with the first argument being the *control qubit*.

2. Write a Quil program to define a controlled-\ :math:`Y` gate in this
   manner. Find the wavefunction when applying this gate to qubit 1
   controlled by qubit 0.

Exercise 3 - Grover's Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a quantum program for the single-shot Grover's algorithm. The
Python function you should produce is:

::

    # data is an array of 0's and 1's such that there are exactly three times as many
    # 0's as 1's
    def single_shot_grovers(data):
        # return an index that contains the value 1

As an example: ``single_shot_grovers([0,0,1,0])`` should return 2.

**HINT** - Remember that the Grover's diffusion operator is:

.. math::


   \begin{pmatrix}
   2/N - 1 & 2/N & \cdots & 2/N \\
   2/N &  & &\\
   \vdots & & \ddots & \\
   2/N & & & 2/N-1
   \end{pmatrix}
