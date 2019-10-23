.. _quickstart:

New in Forest 2 - Other
=======================

There are many other changes to the Forest SDK (comprising pyQuil, Quil, the Quil Compiler, and
the QVM).

.. note::

    For installation & setup, follow the download instructions in the section :ref:`start` at the top of the page.


Updates to the Quil language
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary differences in the programming language Quil 1.0 (as appearing in pyQuil 1.x) and Quil 2 (as appearing in
pyQuil 2) amount to an enhanced memory model. Whereas the classical memory model in Quil 1.0 amounted to an flat bit array of
indefinite size, the memory model in Quil 2 is segmented into typed, sized, named regions.

In terms of compatibility with Quil 1.0, this primarily changes how ``MEASURE`` instructions are formulated, since their
classical address targets must be modified to fit the new framework. In terms of new functionality, this allows angle
values to be read in from classical memory.

``DAGGER`` and ``CONTROLLED`` modifiers
--------------------------------------------

Quil 2 also introduces easier ways to manipulate gates by using gate modifiers. Two gate modifiers are supported currently,
``DAGGER`` and ``CONTROLLED``.

``DAGGER`` can be written before a gate to refer to its inverse. For instance::

    DAGGER RX(pi/3) 0

would have the same effect as::

    RX(-pi/3) 0

``DAGGER`` can be applied to any gate, but also circuits defined with ``DEFCIRCUIT``. This allows for easy reversal of unitary circuits::

    DEFCIRCUIT BELL:
        H 0
        CNOT 0 1

    # construct a Bell state
    BELL
    # disentangle, bringing us back to identity
    DAGGER BELL

.. _quil_2_declare:

``DECLARE``
------------

Classical memory regions must be explicitly requested and named by a Quil program using ``DECLARE`` directive. A generic
``DECLARE`` directive has the following syntax:

``DECLARE region-name type([count])? (SHARING parent-region-name (OFFSET (offset-count offset-type)+))?``

The non-keyword items have the following allowable values:

-  ``region-name``: any non-keyword formal name.

-  ``type``: one of ``REAL``, ``BIT``, ``OCTET``, or ``INTEGER``

-  ``parent-region-name``: any non-keyword formal name previously used as ``region-name`` in a different ``DECLARE`` statement.

-  ``offset-count``: a nonnegative integer.

-  ``offset-type``: the same allowable values as ``type``.

Here are some examples::

    DECLARE beta REAL[32]
    DECLARE ro BIT[128]
    DECLARE beta-bits BIT[1436] SHARING beta
    DECLARE fourth-bit-in-beta1 BIT SHARING beta OFFSET 1 REAL 4 BIT

In order, the intention of these ``DECLARE`` statements is:

-  Allocate an array called ``beta`` of length 32, each entry of which is a ``REAL`` number.

-  Allocate an array called ``ro`` of length 128, each entry of which is a ``BIT``.

-  Name an array called ``beta-bits``, which is an overlay onto the existing array ``beta``, so that the bit representations of elements of ``beta`` can be directly examined and manipulated.

-  Name a single ``BIT`` called ``fourth-bit-in-beta1`` which overlays the fourth bit of the bit representation of the ``REAL`` value ``beta[1]``.


Backwards compatibility
~~~~~~~~~~~~~~~~~~~~~~~

Quil 1.0 is not compatible with Quil 2 in the following ways:

-  The unnamed memory references ``[n]`` and ``[n-m]`` have no direct equivalent in Quil 2 and must be replaced by named
   memory references. (This primarily affects ``MEASURE`` instructions.)

-  The classical memory manipulation instructions have been modified: the operands of ``AND`` have been reversed (so that
   in Quil 2, the left operand is the target address) and ``OR`` has been replaced by ``IOR`` and its operands reversed (so
   that, again, in Quil 2 the left operand is the target address).

In all other instances, Quil 1.0 will operate identically with Quil 2.

When confronted with program text conforming to Quil 1.0, pyQuil 2 will automatically rewrite ``MEASURE q [n]`` to
``MEASURE q ro[n]`` and insert a ``DECLARE`` statement which allocates a ``BIT``-array of the appropriate size named
``ro``.

Updates to Forest
~~~~~~~~~~~~~~~~~

-  In Forest 1.3, job submission to the QPU was done from your workstation and the ability was gated by on user ID. In
   Forest 2, job submission to the QPU must be done from your remote virtual machine, called a QMI (*Quantum Machine Image*).

-  In Forest 1.3, user data persisted indefinitely in cloud storage and could be accessed using the assigned job ID. In
   Forest 2, user data is stored only transiently, and it is the user's responsibility to handle long-term data storage
   on their QMI.

- Forest 1.3 refered to the software developer kit (pyQuil, QVM, Quilc) and the cloud platform for submitting jobs.
  Forest 2 is the SDK which you can install on your own computer or use pre-installed on a QMI. The entire platform
  is called Quantum Cloud Services (QCS).


Example: Computing the bond energy of molecular hydrogen, pyQuil 1.9 vs 2.0
---------------------------------------------------------------------------

By way of example, let's consider the following pyQuil 1.9 program,
which computes the natural bond distance in molecular hydrogen using a
VQE-type algorithm:

.. code:: python

    from pyquil.api import QVMConnection
    from pyquil.quil import Program


    def setup_forest_objects():
        qvm = QVMConnection()
        return qvm


    def build_wf_ansatz_prep(theta):
        program = Program(f"""
    # set up initial state
    X 0
    X 1

    # build the exponentiated operator
    RX(pi/2) 0
    H 1
    H 2
    H 3

    CNOT 0 1
    CNOT 1 2
    CNOT 2 3
    RZ({theta}) 3
    CNOT 2 3
    CNOT 1 2
    CNOT 0 1

    RX(-pi/2) 0
    H 1
    H 2
    H 3

    # measure out the results
    MEASURE 0 [0]
    MEASURE 1 [1]
    MEASURE 2 [2]
    MEASURE 3 [3]""")
        return program


    # some constants
    bond_step, bond_min, bond_max = 0.05, 0, 200
    angle_step, angle_min, angle_max = 0.1, 0, 63
    convolution_coefficients = [0.1698845197777728, 0.16988451977777283, -0.2188630663199042,
                                -0.2188630663199042]
    shots = 1000

    # set up the Forest object
    qvm = setup_forest_objects()

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        # set up the Program object, each time we have a new parameter
        program = build_wf_ansatz_prep(angle_min + offset * angle_step)
        bitstrings = qvm.run(program, indices, trials=shots)

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

    # compute minimum energy as a function of bond length
    min_energies = list(range(bond_min, bond_max))
    for bond_length in min_energies:
        energies = []
        for offset in range(angle_min, angle_max):
            energy = 0
            for j in range(4):
                energy += occupations[offset][j] * convolution_coefficients[j]
            energies.append(energy)

        min_energies[bond_length] = min(energies)

    min_index = min_energies.index(min(min_energies))
    min_energy, relaxed_length = min_energies[min_index], min_index * bond_step

In order to port this code to pyQuil 2.0, we need change only one thing: the part referencing ``QVMConnection`` should be replaced by an equivalent part referencing a ``QuantumComputer`` connected to a ``QVM``. Specifically, the following
snippet

.. code:: python

    from pyquil.api import QVMConnection

    def setup_forest_objects():
        qvm = QVMConnection()
        return qvm

can be changed to

.. code:: python

    from pyquil.api import get_qc

    def setup_forest_objects():
        qc = get_qc("9q-square-qvm")
        return qc

and the references to ``qvm`` in the main body are changed to ``qc`` instead. Since the ``QuantumComputer`` object also
exposes a ``run`` routine and pyQuil itself automatically rewrites 1.9-style ``MEASURE`` instructions into 2.0-style
instructions, this is all we need to do.

If we are willing to be more intrusive, we can also take advantage of pyQuil 2.0's classical memory and parametric
programs. The first piece to change is the Quil program itself: we remove the argument ``theta`` from the Python
function ``build_wf_ansatz_prep``, with the intention of letting the QPU fill it in later. In turn, we modify the Quil
program itself to have a ``REAL`` memory parameter named ``theta``. We also declare a few ``BIT``\ s for our ``MEASURE``
instructions to target.

.. code:: python

    def build_wf_ansatz_prep():
        program = Program("""
    # set up memory
    DECLARE ro BIT[4]
    DECLARE theta REAL

    # set up initial state
    X 0
    X 1

    # build the exponentiated operator
    RX(pi/2) 0
    H 1
    H 2
    H 3

    CNOT 0 1
    CNOT 1 2
    CNOT 2 3
    RZ(theta) 3
    CNOT 2 3
    CNOT 1 2
    CNOT 0 1

    RX(-pi/2) 0
    H 1
    H 2
    H 3

    # measure out the results
    MEASURE 0 ro[0]
    MEASURE 1 ro[1]
    MEASURE 2 ro[2]
    MEASURE 3 ro[3]""")
        return program

Next, we modify the execution loop. Rather than reformulating the :py:class:`~pyquil.quil.Program` object each time, we build and compile it
once, then use the ``.load()`` method to transfer the parametric program to the (simulated) quantum device. We then set
only the angle value within the inner loop, and we change to using ``.run()`` and ``.wait()`` methods to manage control
between us and the quantum device.

More specifically, the old execution loop

.. code:: python

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        # set up the Program object, each time we have a new parameter
        program = build_wf_ansatz_prep(angle_min + offset * angle_step)
        bitstrings = qvm.run(program, indices, trials=shots)

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

becomes

.. code:: python

    # set up the Program object, ONLY ONCE
    program = build_wf_ansatz_prep().wrap_in_numshots_loop(shots=shots)
    binary = qc.compile(program)

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        bitstrings = qc.run(binary, {'theta': [angle_min + offset * angle_step]})

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

Overall, the resulting program looks like this:

.. code:: python

    from pyquil.api import get_qc
    from pyquil.quil import Program


    def setup_forest_objects():
        qc = get_qc("9q-square-qvm")
        return qc


    def build_wf_ansatz_prep():
        program = Program("""
    # set up memory
    DECLARE ro BIT[4]
    DECLARE theta REAL

    # set up initial state
    X 0
    X 1

    # build the exponentiated operator
    RX(pi/2) 0
    H 1
    H 2
    H 3

    CNOT 0 1
    CNOT 1 2
    CNOT 2 3
    RZ(theta) 3
    CNOT 2 3
    CNOT 1 2
    CNOT 0 1

    RX(-pi/2) 0
    H 1
    H 2
    H 3

    # measure out the results
    MEASURE 0 ro[0]
    MEASURE 1 ro[1]
    MEASURE 2 ro[2]
    MEASURE 3 ro[3]""")
        return program


    # some constants
    bond_step, bond_min, bond_max = 0.05, 0, 200
    angle_step, angle_min, angle_max = 0.1, 0, 63
    convolution_coefficients = [0.1698845197777728, 0.16988451977777283, -0.2188630663199042,
                                -0.2188630663199042]
    shots = 1000

    # set up the Forest object
    qc = setup_forest_objects()

    # set up the Program object, ONLY ONCE
    program = build_wf_ansatz_prep().wrap_in_numshots_loop(shots=shots)
    binary = qc.compile(program)

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        bitstrings = qc.run(binary, {'theta': [angle_min + offset * angle_step]})

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

    # compute minimum energy as a function of bond length
    min_energies = list(range(bond_min, bond_max))
    for bond_length in min_energies:
        energies = []
        for offset in range(angle_min, angle_max):
            energy = 0
            for j in range(4):
                energy += occupations[offset][j] * convolution_coefficients[j]
            energies.append(energy)

        min_energies[bond_length] = min(energies)

    min_index = min_energies.index(min(min_energies))
    min_energy, relaxed_length = min_energies[min_index], min_index * bond_step


Miscellanea
~~~~~~~~~~~

Quil promises that a BIT is 1 bit and that an OCTET is 8 bits. Quil does not promise, however, the size or layout of
INTEGER or REAL. These are implementation-dependent.

On the QPU, ``INTEGER`` refers to an unsigned integer stored in a 48-bit wide little-endian word, and ``REAL`` refers to
a 48-bit wide little-endian fixed point number of type <0.48>. In general, these datatypes are implementation-dependent.
``OCTET`` always refers to an 8-bit wide unsigned integer and is independent of implementation.

Memory regions are all "global": ``DECLARE`` directives cannot appear in the body of a ``DEFCIRCUIT``.

On the QVM, INTEGER is a two's complement signed 64-bit integer. REAL is an IEEE-754 double-precision floating-point number.


Error reporting
~~~~~~~~~~~~~~~

Because the Forest 2.0 execution model is no longer asynchronous, our error reporting model has also changed. Rather
than writing to technical support with a job ID, users will need to provide all pertinent details to how they produced an
error.

PyQuil 2 makes this task easy with the function decorator ``@pyquil_protect``, found in the module
``pyquil.api``. By decorating a failing function (or a function that has the potential to fail), any
unhandled exceptions will cause an error log to be written to disk (at a user-specifiable location). For example, the
nonsense code block

::

    from pyquil.api import pyquil_protect

    ...

    @pyquil_protect
    def my_function():
        ...
        qc.qam.load(qc)
        ...

    my_function()

causes the following error to be printed:

::

    >>> PYQUIL_PROTECT <<<
    An uncaught exception was raised in a function wrapped in pyquil_protect.  We are writing out a
    log file to "/Users/your_name/Documents/pyquil/pyquil_error.log".

    Along with a description of what you were doing when the error occurred, send this file to Rigetti Computing
    support by email at support@rigetti.com for assistance.
    >>> PYQUIL_PROTECT <<<

as well as the following log file to be written to disk at the indicated
location:

::

    {
      "stack_trace": [
        {
          "name": "pyquil_protect_wrapper",
          "filename": "/Users/your_name/Documents/pyquil/pyquil/error_reporting.py",
          "line_number": 197,
          "locals": {
            "e": "TypeError('quil_binary argument must be a QVMExecutableResponse. This error is typically triggered by
                forgetting to pass (nativized) Quil to native_quil_to_executable or by using a compiler meant to be used
                for jobs bound for a QPU.',)",
            "old_filename": "'pyquil_error.log'",
            "kwargs": "{}",
            "args": "()",
            "log_filename": "'pyquil_error.log'",
            "func": "<function my_function at 0x106dc4510>"
          }
        },
        {
          "name": "my_function",
          "filename": "<stdin>",
          "line_number": 10,
          "locals": {
            "offset": "0",
            "occupations": "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]"
          }
        },
        {
          "name": "wrapper",
          "filename": "/Users/your_name/Documents/pyquil/pyquil/error_reporting.py",
          "line_number": 228,
          "locals": {
            "pre_entry": "CallLogValue(timestamp_in=datetime.datetime(2018, 9, 11, 18, 40, 19, 65538),
                timestamp_out=None, return_value=None)",
            "key": "run('<pyquil.api._qvm.QVM object at 0x1027e3940>', )",
            "kwargs": "{}",
            "args": "(<pyquil.api._qvm.QVM object at 0x1027e3940>,)",
            "func": "<function QVM.run at 0x106db4e18>"
          }
        },
        {
          "name": "run",
          "filename": "/Users/your_name/Documents/pyquil/pyquil/api/_qvm.py",
          "line_number": 376,
          "locals": {
            "self": "<pyquil.api._qvm.QVM object at 0x1027e3940>",
            "__class__": "<class 'pyquil.api._qvm.QVM'>"
          }
        }
      ],
      "timestamp": "2018-09-11T18:40:19.253286",
      "call_log": {
        "__init__('<pyquil.api._qvm.QVM object at 0x1027e3940>', '<pyquil.api._base_connection.ForestConnection object at
            0x1027e3588>', )": {
          "timestamp_in": "2018-09-11T18:40:18.967750",
          "timestamp_out": "2018-09-11T18:40:18.968170",
          "return_value": "None"
        },
        "run('<pyquil.api._qvm.QVM object at 0x1027e3940>', )": {
          "timestamp_in": "2018-09-11T18:40:19.065538",
          "timestamp_out": null,
          "return_value": null
        }
      },
      "exception": "TypeError('quil_binary argument must be a QVMExecutableResponse. This error is typically triggered
        by forgetting to pass (nativized) Quil to native_quil_to_executable or by using a compiler meant to be used for
        jobs bound for a QPU.',)",
      "system_info": {
        "python_version": "3.6.3 (default, Jan 25 2018, 13:55:02) \n[GCC 4.2.1 Compatible Apple LLVM 9.0.0
            (clang-900.0.39.2)]",
        "pyquil_version": "2.0.0-internal.1"
      }
    }

Please attach such a logfile to any request for support.

Parametric Programs
~~~~~~~~~~~~~~~~~~~

In PyQuil 1.x, there was an object named ``ParametricProgram``::

    # This function returns a quantum circuit with different rotation angles on a gate on qubit 0
    def rotator(angle):
        return Program(RX(angle, 0))

    from pyquil.parametric import ParametricProgram
    par_p = ParametricProgram(rotator) # This produces a new type of parameterized program object

This object has been removed from PyQuil 2. Please consider simply using a Python function for
the above functionality::

    par_p = rotator

Or using declared classical memory::

    p = Program()
    angle = p.declare('angle', 'REAL')
    p += RX(angle, 0)

