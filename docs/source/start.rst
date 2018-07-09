.. _start:

Installation and Getting Started
================================

Make sure you have a current version of Python installed on your computer. We recommend
installing the `Anaconda Python distribution <https://www.anaconda.com/download/>`_.



Then, install pyQuil with::

    conda install -c rigetti pyquil


.. note::

    PyQuil requires Python 3.

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

Getting Started
---------------

This toolkit provides some simple libraries for writing quantum
programs.

.. code:: python

    from pyquil.quil import Program
    from pyquil.api import QVMConnection
    from pyquil.gates import CNOT, H

    qvm = QVMConnection()
    p = Program(H(0), CNOT(0, 1))

    wf = qvm.wavefunction(p)
    print(wf)

::

    (0.7071067812+0j)|00> + (0.7071067812+0j)|11>

It comes with a few parts:

1. **Quil**: The Quantum Instruction Language standard. Instructions
   written in Quil can be executed on any implementation of a quantum
   abstract machine, such as the quantum virtual machine (QVM), or on a
   real quantum processing unit (QPU). More details regarding Quil can be
   found in the `whitepaper <https://arxiv.org/abs/1608.03355>`__.
2. **pyQuil**: A Python library to help write and run Quil code and
   quantum programs.
3. **QVM**: A `Quantum Virtual Machine <qvm.html>`_, which is an implementation of the
   quantum abstract machine on classical hardware. The QVM lets you use a
   regular computer to simulate a small quantum computer. You can access
   the Rigetti QVM running in the cloud with your API key.
   `Sign up here <http://forest.rigetti.com>`_ to get your key.
4. **QPU**: pyQuil also includes some a special connection which lets you run experiments
   on Rigetti's prototype superconducting quantum processors over the cloud.
5. **Quilc**: In addition to running on the QVM or the QPU, users can directly use
   the Quil compiler, to investigate how arbitrary quantum programs can be compiled
   to target specific physical instruction set architectures (ISAs).


Your First Quantum Program
--------------------------
pyQuil is a Python library that helps you write programs in the Quantum Instruction Language (Quil).
It also ships with a simple script ``run_quil.py`` that runs Quil code directly. The script is located in the `pyQuil repository <https://github.com/rigetticomputing/pyquil>`_. You can
test your connection to Forest using this script by cloning the pyQuil repository and executing the following on your command line

::

    cd pyquil/examples/
    python run_quil.py hello_world.quil

You should see the following output array ``[[1, 0, 0, 0, 0, 0, 0, 0]]``.
This indicates that you have successfully interacted with our API.

.. note::

    If you installed pyQuil using Anaconda or pip (as explained above), you can find the examples and
    the `run_quil.py` script in the `pyQuil GitHub repository
    <https://github.com/rigetticomputing/pyquil/tree/master/examples>`_.

You can continue to write more Quil code in files and run them using the ``run_quil.py`` script.
The following sections describe how to use the pyQuil library directly to build quantum programs in
Python.
