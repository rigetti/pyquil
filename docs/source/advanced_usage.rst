.. _advanced_usage:

Advanced Usage
==============

.. note::

    If you're running locally, remember set up the QVM and quilc in server mode before trying to use
    them: :ref:`server`.

PyQuil Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

Network endpoints for the Rigetti Forest infrastructure and information pertaining to QPU access are
stored in a pair of configuration files. These files are located by default at ``~/.qcs_config`` and ``~/.forest_config``.
The location can be changed by setting the environment variables ``QCS_CONFIG`` or ``FOREST_CONFIG`` to point to the new
location.

When running on a QMI, the values in these configuration files are automatically managed so as to
point to the correct endpoints. When running locally, configuration files are not necessary. Thus, the average
user will not have to do any work to get their configuration files set up.

If for some reason you want to use an atypical configuration, you may need to modify these files.

.. exec on engage

The default QCS config file on any QMI looks similar to the following:

::

    # .qcs_config
    [Rigetti Forest]
    url = https://forest-server.qcs.rigetti.com
    key = 4fd12391-11eb-52ec-35c2-262765ae4c4f
    user_id = 4fd12391-11eb-52ec-35c2-262765ae4c4f

    [QPU]
    exec_on_engage = bash exec_on_engage.sh

where

 -  ``url`` is the endpoint that pyQuil hits for device information and for the 2.0 endpoints,
 -  ``key`` stores the Forest 1.X API key,
 -  ``user_id`` stores a Forest 2.0 user ID, and
 -  ``exec_on_engage`` specifies the shell command that the QMI will launch when the QMI becomes QPU-engaged. It
    would have no effect if you are running locally, but is important if you are running on the QMI. By default, it runs the
    ``exec_on_engage.sh`` shell script. It's best to leave the configuration as is, and edit that script.
    More documentation about ``exec_on_engage.sh`` can be found in the QCS docs
    `here <https://www.rigetti.com/qcs/docs/guides#queuing-programs-for-auto-execution>`_.

The Forest config file on any QMI has these contents, with specific IP addresses filled in:

::

    # .forest_config
    [Rigetti Forest]
    qpu_endpoint_address = None
    qvm_address = http://10.1.165.XX:5000
    compiler_server_address = tcp://10.1.165.XX:5555

where

 -  ``qpu_endpoint_address`` is the endpoint where pyQuil will try to communicate with the QPU orchestrating service
    during QPU-engagement. It may not appear until your QMI engages, and furthermore will have no effect if you are
    running locally. It's best to leave this alone. If you obtain access to one of our QPUs, we will fill it in for you.
 -  ``qvm_address`` is the endpoint where pyQuil will try to communicate with the Rigetti Quantum Virtual Machine.
    On a QMI, this points to the provided QVM instance. On a local installation, this should be set to the server endpoint
    for a locally running QVM instance. However, pyQuil will use the default value ``http://localhost:5000`` if this file
    isn't found, which is the correct endpoint when you run the QVM locally with ``qvm -S``.
 -  ``compiler_server_address``: This is the endpoint where pyQuil will try to communicate with the compiler server. On a
    QMI, this points to a provided compiler server instance. On a local installation, this should be set to the server
    endpoint for a locally running ``quilc`` instance. However, pyQuil will use the default value ``tcp://localhost:5555``
    if this isn't set, which is the correct endpoint when you run ``quilc`` locally with ``quilc -R``.

.. note::

     PyQuil itself reads these values out using the helper class ``pyquil._config.PyquilConfig``. PyQuil users should not
     ever need to touch this class directly.

Using Qubit Placeholders
~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    The functionality provided inline by ``QubitPlaceholders`` is similar to writing a function which returns a
    ``Program``, with qubit indices taken as arguments to the function.

In pyQuil, we typically use integers to identify qubits

.. code:: python

    from pyquil import Program
    from pyquil.gates import CNOT, H
    print(Program(H(0), CNOT(0, 1)))

.. parsed-literal::

    H 0
    CNOT 0 1

However, when running on real, near-term QPUs we care about what
particular physical qubits our program will run on. In fact, we may want
to run the same program on an assortment of different qubits. This is
where using ``QubitPlaceholder``\ s comes in.

.. code:: python

    from pyquil.quilatom import QubitPlaceholder
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    p = Program(H(q0), CNOT(q0, q1))
    print(p)

.. parsed-literal::

    H {q4402789176}
    CNOT {q4402789176} {q4402789120}

If you try to use this program directly, it will not work

.. code:: python

    print(p.out())

::

    RuntimeError: Qubit q4402789176 has not been assigned an index


Instead, you must explicitly map the placeholders to physical qubits. By
default, the function ``address_qubits`` will address qubits from 0 to
N.

.. code:: python

    from pyquil.quil import address_qubits
    print(address_qubits(p))

.. parsed-literal::

    H 0
    CNOT 0 1

The real power comes into play when you provide an explicit mapping:

.. code:: python

    print(address_qubits(prog, qubit_mapping={
        q0: 14,
        q1: 19,
    }))

.. parsed-literal::

    H 14
    CNOT 14 19


Register
--------

Usually, your algorithm will use an assortment of qubits. You can use
the convenience function ``QubitPlaceholder.register()`` to request a
list of qubits to build your program.

.. code:: python

    qbyte = QubitPlaceholder.register(8)
    p_evens = Program(H(q) for q in qbyte)
    print(address_qubits(p_evens, {q: i*2 for i, q in enumerate(qbyte)}))


.. parsed-literal::

    H 0
    H 2
    H 4
    H 6
    H 8
    H 10
    H 12
    H 14

Classical Control Flow
~~~~~~~~~~~~~~~~~~~~~~

.. note::

    Classical control flow is not yet supported on the QPU.


Here are a couple quick examples that show how much richer a Quil program
can be with classical control flow. In this first example, we create a while
loop by following these steps:

1. Declare a register called ``flag_register`` to use as a boolean test for looping.

2. Initialize this register to ``1`` program so our while loop will execute. This is often called the
   *loop preamble* or *loop initialization*.

3. Write the body of the loop in its own :py:class:`~pyquil.quil.Program`. This will be a
   program that applies an :math:`X` gate followed by a :math:`H` gate on our
   qubit.

4. Using the :py:func:`~pyquil.quil.Program.while_do` method to add control flow.

.. code:: python

    from pyquil import Program
    from pyquil.gates import *

    # Initialize the Program and declare a 1 bit memory space for our boolean flag
    outer_loop = Program()
    flag_register = outer_loop.declare('flag_register', 'BIT')

    # Set the initial flag value to 1
    outer_loop += MOVE(flag_register, 1)

    # Define the body of the loop with a new Program
    inner_loop = Program()
    inner_loop += Program(X(0), H(0))
    inner_loop += MEASURE(0, flag_register)

    # Run inner_loop in a loop until flag_register is 0
    outer_loop.while_do(flag_register, inner_loop)

    print(outer_loop)

.. parsed-literal::

    DECLARE flag_register BIT[1]
    MOVE flag_register 1
    LABEL @START1
    JUMP-UNLESS @END2 flag_register
    X 0
    H 0
    MEASURE 0 flag_register
    JUMP @START1
    LABEL @END2

Notice that the ``outer_loop`` program applied a Quil instruction directly to a
classical register.  There are several classical commands that can be used in this fashion:

- ``NOT`` which flips a classical bit
- ``AND`` which operates on two classical bits
- ``IOR`` which operates on two classical bits
- ``MOVE`` which moves the value of a classical bit at one classical address into another
- ``EXCHANGE`` which swaps the value of two classical bits

In this next example, we show how to do conditional branching in the
form of the traditional ``if`` construct as in many programming
languages. Much like the last example, we construct programs for each
branch of the ``if``, and put it all together by using the :py:func:`~pyquil.quil.Program.if_then`
method.

.. code:: python

    # Declare our memory spaces
    branching_prog = Program()
    test_register = branching_prog.declare('test_register', 'BIT')
    ro = branching_prog.declare('ro', 'BIT')

    # Construct each branch of our if-statement. We can have empty branches
    # simply by having empty programs.
    then_branch = Program(X(0))
    else_branch = Program()

    # Construct our program so that the result in test_register is equally likely to be a 0 or 1
    branching_prog += H(1)
    branching_prog += MEASURE(1, test_register)

    # Add the conditional branching
    branching_prog.if_then(test_register, then_branch, else_branch)

    # Measure qubit 0 into our readout register
    branching_prog += MEASURE(0, ro)

    print(branching_prog)

.. parsed-literal::

    DECLARE test_register BIT[1]
    DECLARE ro BIT[1]
    H 1
    MEASURE 1 test_register
    JUMP-WHEN @THEN1 test_register
    JUMP @END2
    LABEL @THEN1
    X 0
    LABEL @END2
    MEASURE 0 ro

We can run this program a few times to see what we get in the readout register ``ro``.

.. code:: python

    from pyquil import get_qc

    qc = get_qc("2q-qvm")
    branching_prog.wrap_in_numshots_loop(10)
    qc.run(branching_prog)

.. parsed-literal::

    [[1], [1], [1], [0], [1], [0], [0], [1], [1], [0]]


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
    a = 0.5 * ID()
    b = -0.75 * sX(0) * sY(1) * sZ(3)
    c = (5-2j) * sZ(1) * sX(2)

    # Construct a sum of Pauli terms.
    sigma = a + b + c
    print(f"sigma = {sigma}")

.. parsed-literal::

    sigma = (0.5+0j)*I + (-0.75+0j)*X0*Y1*Z3 + (5-2j)*Z1*X2

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

    from pyquil.paulis import exponential_map

    sigma_cubed = sigma * sigma * sigma
    print(f"Simplified: {sigma_cubed}\n")

    # Produce Quil code to compute exp[iX]
    H = -1.0 * sX(0)
    print(f"Quil to compute exp[iX] on qubit 0:\n"
           f"{exponential_map(H)(1.0)}")

.. parsed-literal::

    Simplified: (32.46875-30j)*I + (-16.734375+15j)*X0*Y1*Z3 + (71.5625-144.625j)*Z1*X2

    Quil to compute exp[iX] on qubit 0:
    H 0
    RZ(-2.0) 0
    H 0

``exponential_map`` returns a function allowing you to fill in a multiplicative
constant later. This commonly occurs in variational algorithms. The function
``exponential_map`` is used to compute :math:`\exp[-i \alpha H]` without explicitly filling in a
value for :math:`\alpha`.

.. code:: python

    expH = exponential_map(H)
    print(f"0:\n{expH(0.0)}\n")
    print(f"1:\n{expH(1.0)}\n")
    print(f"2:\n{expH(2.0)}")

.. parsed-literal::
    0:
    H 0
    RZ(0) 0
    H 0

    1:
    H 0
    RZ(-2.0) 0
    H 0

    2:
    H 0
    RZ(-4.0) 0
    H 0

To take it one step further, you can use :ref:`parametric_compilation` with ``exponential_map``. For instance:

.. code:: python

    ham = sZ(0) * sZ(1)
    prog = Program()
    theta = prog.declare('theta', 'REAL')
    prog += exponential_map(ham)(theta)


