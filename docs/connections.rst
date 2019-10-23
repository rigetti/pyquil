.. _connections:

Connections
===========

Larger pyQuil programs can involve more qubits and take a longer time to run. Instead of running the
program immediately, you can insert your programs into a queue. This is done with the ``use_queue``
parameter to QVMConnection.  By default, this parameter is set to False which means it skips
the queue and runs it immediately. However, the QVM will reject programs that are more than
19 qubits or take longer than 10 seconds to run. Therefore, to run programs of a larger size you must
set the ``use_queue`` parameter to True which has more overhead.

.. code:: python

  from pyquil.quil import Program
  from pyquil.api import QVMConnection

  qvm = QVMConnection(use_queue=True)
  qvm.run(Program(X(0).measure(0, 0), [0])

The Forest queue also allows an asynchronous mode of interaction with methods postfixed with ``_async``.
This means that there is a seperate query to post a job and to get the result.

::

  from pyquil.quil import Program
  from pyquil.gates import X, H, I
  from pyquil.api import QVMConnection

  qvm = QVMConnection()
  job_id = qvm.run_async(Program(X(0)).measure(0, 0), [0])

The `job_id` is a string that uniquely identifies the job in Forest. You can use the
`.get_job` method on QVMConnection to get the current status.

::

  job = qvm.get_job(job_id)
  if not job.is_done():
    time.sleep(1)
    job = qvm.get_job(job_id)
  print(job.result())

.. parsed-literal::

  [[1]]

The `wait_for_job` method periodically checks for updates and prints the job's position
in the queue, similar to the above code.

::

  job = qvm.wait_for_job(job_id)
  print(job.result())

.. parsed-literal::

  [[1]]

Optimized Calls
~~~~~~~~~~~~~~~

This same pattern as above applies to the :meth:`~pyquil.api.QVMConnection.wavefunction`,
:meth:`~pyquil.api.QVMConnection.expectation` and :meth:`~pyquil.api.QVMConnection.run_and_measure`.
These are very useful if used appropriately: They all execute a given program *once and only once*
and then either return the final wavefunction or use it to generate expectation values or a
specified number of random bitstring samples.

.. warning::

    This behavior can have unexpected consequences if the program that prepares the final state
    is non-deterministic, e.g., if it contains measurements and/or noisy gate applications.
    In this case, the final state after the program execution is itself a random variable
    and a single call to these functions therefore **cannot** sample the full space of outcomes.
    Therefore, if the program is non-deterministic and sampling the full program output distribution
    is important for the application at hand, we recommend using the basic
    :meth:`~pyquil.api.QVMConnection.run` API function as this re-runs the full program for every
    requested trial.
