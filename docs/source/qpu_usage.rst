..  _qpu_usage:

Using the QPU-based stack
=========================

The broad strokes of working with the QPU-based pyQuil stack are identical to using the QVM-based
stack: the library ``pyquil.api`` supplies an object class ``QPUConnection`` which mediates the
transmission of Quil programs to the QPU, encoded as ``pyquil.quil.Program`` objects, as well as
the receipt of job results, encoded as bitstring lists.

.. note::

    User permissions for QPU access must be enabled by a Forest administrator.  ``QPUConnection``
    calls will automatically fail without these user permissions.  Speak to a Forest administrator
    for information about upgrading your access plan.

Detecting the available QPUs and their structure
------------------------------------------------

The initialization function for a ``QPUConnection`` object takes a QPU name as its sole argument.
Devices are typically named according to the convention ``[n]Q-[name]``, where ``n`` is the number
of active qubits on the device and ``name`` is a human-readable name that designates the device.
The available QPUs can be inspected via a PyQuil interface, as demonstrated in the following
snippet:

.. code:: python

    from pyquil.api import get_devices
    for device in get_devices():
        if device.is_online():
            print('Device {} is online'.format(device.name))

The ``Device`` objects returned by ``get_devices`` will capture other characterizing statistics
about the associated QPU at a later date.

Execution on the QPU
--------------------

The user-facing interface to running Quil programs on the QPU is nearly identical to that of the
QVM.  A ``QPUConnection`` object provides the following methods:

* ``.run(quil_program, classical_addresses, trials=1)``: This method sends the ``Program`` object
  ``quil_program`` to the QPU for execution, which runs the program ``trials`` many times.  After
  each run on the QPU, all the qubits in the QPU are simultaneously measured and their results are
  stored in classical registers according to the MEASURE instructions provided. Then, a list of
  registers listed in ``classical_addresses`` is returned to the user for each trial. This call is
  blocking: it will wait until the QPU returns its results for inspection.
* ``.run_async(quil_program, classical_addresses, trials=1)``: This method has identical behavior
  to ``.run`` except that it is **nonblocking**, and it instead returns a job ID string.
* ``.run_and_measure(quil_program, qubits, trials=1)``: This method sends the ``Program`` object
  ``quil_program`` to the QPU for execution, which runs the program ``trials`` many times.  After
  each run on the QPU, the all the qubits in the QPU are simultaneously measured, and the results
  from those listed in ``qubits`` are returned to the user for each trial. This call is blocking:
  it will wait until the QPU returns its results for inspection.
* ``.run_and_measure_async(quil_program, qubits, trials=1)``: This method has identical behavior
  to ``.run_and_measure`` except that it is **nonblocking**, and it instead returns a job ID string.

.. note::

    The QPU's ``run`` functionality matches that of the QVM's ``run`` functionality, but the
    behavior of ``run_and_measure`` **does not match** its ``QVMConnection`` counterpart (cf.
    `Optimized Calls <getting_started.html#optimized-calls>`_). The ``QVMConnection`` version of
    ``run`` repeats the execution of a program many times, producing a (potentially) different
    outcome each time, whereas ``run_and_measure`` executes a program only once and uses the QVM's
    unique ability to perform wavefunction inspection to multiply sample the same distribution.
    The QPU **does not** have this ability, and thus its ``run_and_measure`` call behaves as the
    QVM's ``run``.

For example, the following Python snippet demonstrates the execution of a small job on the QPU
identified as "19Q-Acorn":

.. code:: python

    from pyquil.quil import Program
    import pyquil.api as api
    from pyquil.gates import *
    qpu = api.QPUConnection('19Q-Acorn')
    p = Program(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
    qpu.run(p, [0, 1], 1000)

When the QPU execution time is expected to be long and there is classical computation that the
program would like to accomplish in the meantime, the ``QPUConnection`` object allows for an
asynchronous ``run_async`` call to be placed instead.  By storing the resulting job ID,
the state of the job and be queried later and its results obtained then.  The mechanism for
querying the state of a job is also through the ``QPUConnection`` object: a job ID string can be
transformed to a ``pyquil.api.Job`` object via the method ``.get_job(job_id)``; the state of a
``Job`` object (taken at its creation time) can then be inspected by the method ``.is_done()``;
and when this returns ``True`` the output of the QPU can be retrieved via the method ``.result()``.

For example, consider the following Python snippet:

.. code:: python

    from pyquil.quil import Program
    import pyquil.api as api
    from pyquil.gates import *
    qpu = api.QPUConnection('19Q-Acorn')
    p = Program(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
    job_id = qpu.run_async(p, [0, 1], 1000)
    while not qpu.get_job(job_id).is_done():
        ## get some other work done while we wait
        ...
        ## and eventually yield to recheck the job result
    ## now the job is guaranteed to be finished, so pull the QPU results
    job_result = qpu.get_job(job_id).result()

    

The Quil compiler and expectations for program contents
-------------------------------------------------------

The QPUs have much more limited natural gate sets than the standard gate set offered by pyQuil: the
gate operators are constrained to lie in ``RZ(θ)``, ``RX(±π/2)``, and ``CZ``; and the gates are
required to act on physically available hardware (for single-qubit gates, this means acting only on
live qubits, and for qubit-pair gates, this means acting on neighboring qubits).

To ameliorate these limitations, the QPU execution stack contains an optimizing compiler that
translates arbitrary ProtoQuil to QPU-executable Quil.  The compiler is designed to avoid changing
even non-semantic details of input Quil code, except to make it shorter when possible.  For
instance, it will not readdress Quil code that is already appropriately addressed to physically
realizable hardware objects on the QPU.  The following figure illustrates the layout and addressing
of the Rigetti 19Q-Acorn QPU.

.. figure:: figures/qubit-connectivity.png
    :width: 540px
    :align: center
    :height: 300px
    :alt: alternate text
    :figclass: align-center

    Qubit adjacency schematic for the Rigetti 19Q-Acorn QPU. In particular, notice that qubit 3 is
    disabled.

.. note::

    The Quil compiler can be circumvented entirely by inserting ``PRAGMA PRESERVE_BLOCK`` at the
    start of the ProtoQuil program, which disables even the optimizing passes of the compiler.
    This can be useful, for instance, when performing hardware-level benchmarking calculations,
    where it can be important to perform long sequences of operations that, ultimately, result in
    the identity gate.

The compiler itself is subject to some limitations, and some of the more commonly observed errors
follow:

+ ``! ! ! Error: Failed to select a SWAP instruction. Perhaps the qubit graph is disconnected?``
  This error indicates a readdressing failure: some non-native Quil could not be reassigned to lie
  on native devices.  Two common reasons for this failure are:

    + It is possible for the readdressing problem to be too difficult for the compiler to sort out,
      causing deadlock.
    + If a qubit-qubit gate is requested to act on two qubit resources that lie on disconnected
      regions of the qubit graph, the addresser will fail.

+ ``! ! ! Error: Matrices do not lie in the same projective class.`` The compiler attempted to
  decompose an operator as native Quil instructions, and the resulting instructions do not match the
  original operator.  This can happen when the original operator is not a unitary matrix, and could
  indicate an invalid ``DEFGATE`` block.
+ ``! ! ! Error: Addresser loop only supports pure quantum instructions.`` The compiler inspected an
  instruction that it does not understand.  The most common cause of this error is the inclusion of
  classical control in a program submission (including the manual inclusion of ``MEASURE``
  instructions), which is legal Quil but falls outside of the domain of ProtoQuil.

After being passed through the compiler, gates are applied to qubits at the earliest available time. As a simple example, considering the following:

.. code:: python

    p = Program()
    p.inst(X(0), H(0), H(1))

In this example, ``X(0)`` and ``H(1)`` will be applied simultaneously, followed by `H(0)`.

Retune interruptions
--------------------

Because the QPU is a physical device, it is occasionally taken offline for recalibration.
This offline period typically lasts 10-40 minutes, depending upon QPU characteristics and other
external factors.  During this period, the QPU will be listed as offline, and it will reject
new jobs (but pending jobs will remain queued).  When the QPU resumes activity, its performance
characteristics may be slightly different (in that different gates may enjoy different process
fidelities).

