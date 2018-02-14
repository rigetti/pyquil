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


Retune interruptions
--------------------

Because the QPU is a physical device, it is occasionally taken offline for recalibration.
This offline period typically lasts 10-40 minutes, depending upon QPU characteristics and other
external factors.  During this period, the QPU will be listed as offline, and it will reject
new jobs (but pending jobs will remain queued).  When the QPU resumes activity, its performance
characteristics may be slightly different (in that different gates may enjoy different process
fidelities).

