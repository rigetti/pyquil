.. _start:

Installation and Getting Started
================================

Make sure you have a current version of Python installed on your computer. We recommend
installing the `Anaconda Python distribution <https://www.anaconda.com/download/>`_.



Then, install pyQuil with::

    conda install -c rigetti pyquil


.. note::

    PyQuil works on both Python 2 and 3. However, Rigetti **strongly** recommends
    using Python 3 if possible. Future feature developments in PyQuil may support
    Python 3 only.

.. note::

    We also support installation via ``pip`` with ``pip install pyquil``

Connecting to Rigetti Forest
----------------------------

pyQuil can be used to build and manipulate Quil programs without restriction.
However, to run programs (e.g., to get wavefunctions, get multishot experiment data),
you will need an API key for Rigetti Forest. This will allow you to run your programs
on the Rigetti QVM or QPU.

`Sign up here <http://forest.rigetti.com>`_ to get a Forest API key, it's free
and only takes a few seconds. We also highly recommend that you join our
`public slack channel <http://slack.rigetti.com>`_ where you can
connect with other users and Rigetti members for support.

Run the following command to automatically set up the config. This will prompt you for
the required information (URL, key, and user id). It will then create a file in the
proper location (the user's root directory):

::

    pyquil-config-setup

If the setup completed successfully then you can skip to the next section.

You can also create the configuration file manually if you'd like and place
it at ``~/.pyquil_config``. The configuration file is in INI format and should
contain all the information required to connect to Forest:

::

    [Rigetti Forest]
    key: <Rigetti Forest API key>
    user_id: <Rigetti User ID>

Alternatively, you can place the file at your own chosen location and then set
the ``PYQUIL_CONFIG`` environment variable to the path of the file.

.. note::

  You may specify an absolute path or use the ~ to indicate your home directory.
  On Linux, this points to ``/users/username``.
  On Mac, this points to ``/Users/Username``.
  On Windows, this points to ``C:\Users\Username``

.. note::

  Windows users may find it easier to name the file ``pyquil.ini`` and open it using notepad.
  Then, set the ``PYQUIL_CONFIG`` environment variable by opening up a command prompt and
  running: ``setenv PYQUIL_CONFIG=C:\Users\Username\pyquil.ini``

As a last resort, connection information can be provided via environment variables.

::

    export QVM_API_KEY=<Rigetti Forest API key>
    export QVM_USER_ID=<Rigetti User ID>

If you are still seeing errors or warnings then file a bug using
`Github Issues <https://github.com/rigetticomputing/pyquil/issues>`_.


Make sure it's all working!
~~~~~~~~~~~~~~~~~~~~~~~~~~~
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