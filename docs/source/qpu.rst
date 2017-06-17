
Quantum Processor Unit
======================
pyQuil allows some basic single-qubit experiments to be run on a multi-qubit superconducting quantum
processor.  These three types of experiments are some of the basic building blocks for calibrating
qubits.  This documentation will cover the basics of each experiment, as well as show you how to run
them using pyQuil.

.. note::

    In order to run experiments on the QPU you will need a specially upgraded API key.  If you are
    interested in running these experiments, then email us at support@rigetti.com


Before running any experiments, we need to take a look at what devices are available on the
platform.

.. code-block::

    from pyquil.qpu import QPUConnection, get_info
    print get_info()

.. parsed-literal::

    {u'devices': [{u'name': u'Z12-13-C4a2',
       u'qubits': [{u'num': 5,
         u'rabi_params': {u'start': 0.01,
          u'step': 0.02,
          u'stop': 0.5,
          u'time': 160.0},
         u'ramsey_params': {u'detuning': 0.5,
          u'start': 0.01,
          u'step': 0.2,
          u'stop': 20},
         u'ssr_fidelity': 0.923,
         u't1': 2e-05,
         u't1_params': {u'num_pts': 25, u'start': 0.01, u'stop': 40},
         u't2': 1.5e-05},
        {u'num': 6,
         u'rabi_params': {u'start': 0.01,
          u'step': 0.02,
          u'stop': 0.5,
          u'time': 100.0},
         u'ramsey_params': {u'detuning': 0.5,
          u'start': 0.01,
          u'step': 0.2,
          u'stop': 20},
         u'ssr_fidelity': 0.923,
         u't1': 2.1e-05,
         u't1_params': {u'num_pts': 30, u'start': 0.01, u'stop': 40},
         u't2': 1.5e-05}]}]}

This JSON provides a list of available devices by their name.  In this example we have one device,
called ``Z12-13-C4a2`` which has two qubits (indexed as qubit numbers 5 and 6) configured and
available for experiments.  This configuration information also returns details about the coherence
times ``t1`` and ``t2`` and the measurement fidelity ``ssr_fidelity`` for each qubit.  It also shows
information about the parameters for each of the experiments on each qubit.

Next we open up a connection to the QPU for the available device.

.. code-block::
    qpu = QPUConnection("Z12-13-C4a2")
    qpu.ping() # checks to make sure the connection is good

.. parsed-literal::

    'ok'

Rabi Experiments
----------------
A Rabi experiment runs a series of Quil programs.  Each program is parameterized by a rotation
angle:

.. code-block::

    DEFCIRCUIT RABI(%angle) q:
        RX(%angle) q
        MEASURE q [q]

In our hardware, the angle in the ``RX`` gate is implemented by (roughly) scaling the amplitude of a
micowave pulse.  A larger amplitude corresponds to a larger rotation angle. A Rabi experiment, will
run the RABI program for a series of different amplitudes for the ``RX`` pulse.

Here is how to run a Rabi experiment:

.. code-block::

    my_qubit = 5
    res = qpu.rabi(my_qubit)
    print type(res), res

.. parsed-literal::

    <class 'pyquil.job_results.RabiResult'> {u'status': u'Submitted', u'jobId': u'BLSLJCBGNP'}

Just like in the JobConnection example for working with the QVM, experiments on a QPU work through
the jobqueue pattern.  When a job is completed, we can use a built in method to plot the results

.. code-block::

    from pyquil.job_results import wait_for_job
    from pyquil.plots import analog_plot
    wait_for_job(res) # blocks execution until the job is completed
    analog_plot(res)

TODO include plot
TODO explain plot

Ramsey Experiments
------------------
Ramsey experiments are typically used to measure the T2 coherence time of qubits.  A single run
of the experiment is a ``X-HALF`` pulse, followed by a wait time, followed by another ``X-HALF``
pulse and a measurement.  Sweeping the wait time over many runs gives a Ramsey experiment.

.. code-block::

    my_qubit = 5
    res = qpu.ramsey(my_qubit)
    wait_for_job(res)
    analog_plot(res)

TODO include plot
TODO explain plot


T1 Experiments
--------------
T1 experiments measure the t1 coherence time of qubits. A single run of a T1 experiment is an ``X``
gate followed by a wait time, followed by a measurement.  Sweeping this wait time over many runs
gives a T1 experiment.  Since the ``X`` pulse puts the qubit in the excited state, sweeping over the
wait time gives us a sense of how likely a qubit it to remain in the excited state over time. The
likliehood of the qubit staying in the excited state typically decays exponentially, and the decay
constant of this exponent is called the T1 coherence time.

You can run a T1 experiment on our qubits to check their coherence times.

.. code-block::

    my_qubit = 5
    res = qpu.t1(my_qubit)
    wait_for_job(res)
    analog_plot(res)

TODO include plot
TODO explain plot
