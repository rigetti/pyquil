.. _qpu:

The Quantum Processing Unit (QPU)
=================================

A quantum processing unit (QPU), also referred to as a *quantum chip*, is a physical (fabricated)
chip that contains a number of interconnected qubits. It is the foundational component of a full
quantum computer, which includes the housing environment for the QPU, the control electronics, and
many other components.

This page describes how to use the Forest API for interacting with Rigetti QPUs, and provides
technical details and average performance of **Acorn**, the 19Q QPU currently available, that has
been designed, fabricated and packaged by Rigetti.

.. _qpu_use:

Using the QPU
~~~~~~~~~~~~~

.. note::

    User permissions for QPU access must be enabled by a Forest administrator.  ``QPUConnection``
    calls will automatically fail without these user permissions.  Speak to a Forest administrator
    for information about upgrading your access plan.

One establishes a connection to a Rigetti QPU in the same manner as a QVM:

.. code:: python

    from pyquil.api import QPUConnection
    qpu = QPUConnection() # NOTE: This raises a UserWarning!

There is one caveat, however, as shown in the ``UserWarning`` that is raised by the above
command: You must specify a ``device`` as an argument. This is described in the following section.

Accessing available ``devices`` with ``get_devices()``
------------------------------------------------------

The initialization function for a ``QPUConnection`` object must be provided a speciffic Rigetti QPU
as an argument, so that Forest knows on which quantum computer you want to execute your programs.
The available QPUs, synonymously referred to as ``devices`` in Forest, can be inspected via the
function ``get_devices`` in the ``api`` module:

.. code:: python

    from pyquil.api import get_devices
    for device in get_devices():
        if device.is_online():
            print('Device {} is online'.format(device.name))

.. note::
    The ``Device`` objects returned by ``get_devices`` captures other characteristics about the
    associated QPU, such as its connectivity, coherence times, single- and two-qubit gate
    fidelities. For more information on the ``Device`` class, see :ref:`device_class`.

Devices are typically named according to the convention ``[n]Q-[name]``, where ``n`` is the number
of active qubits on the device and ``name`` is a human-readable name that designates the device.

Execution on the QPU
--------------------

One may execute Quil programs on the QPU (nearly) identically to the QVM, via the ``.run(...)``
method (obviously, since the QPU is a real quantum computer, the ``.wavefunction(...)`` method is
not available). We may fix the above example then by providing a device to the ``QPUConnection``:

.. code:: python

    from pyquil.api import get_devices, QPUConnection

    acorn = get_devices(as_dict=True)['19Q-Acorn']
    qpu = QPUConnection(acorn)
    # The device name as a string is also acceptable
    # qpu = QPUConnection('19Q-Acorn')

You have now established a connection to the ``19Q-Acorn`` QPU. Executing programs is then identical
to the QVM (we may ommit the ``classical_addresses`` and ``trials`` arguments to use their
defaults):

.. code:: python

    from pyquil.quil import Program
    from pyquil.gates import X, MEASURE

    program = Program(X(0), MEASURE(0, 0))
    qpu.run(program)

.. parse-literal:

    [[1]]

In addition to the ``.run(...)`` method, a ``QPUConnection`` object provides the following methods:

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

.. _device_class:

Getting QPU Information from the Device Class
---------------------------------------------

The pyQuil ``Device`` class provides useful information for learning about, and working with,
Rigetti's available QPUs. One may query for available devices using the ``get_devices`` function:

.. code:: python

    from pyquil.api import get_devices

    devices = get_devices(as_dict=True)
    # E.g. {'19Q-Acorn': <Device 19Q-Acorn online>, '8Q-Agave': <Device 8Q-Agave offline>}

    acorn = devices['19Q-Acorn']

The variable ``acorn`` points to a ``Device`` object that holds useful information regarding the
QPU, including:

1. Connectivity via its instruction set architecture (``acorn.isa`` of class ``ISA``).
2. Hardware specifications such as coherence times and fidelities (``acorn.specs`` of class ``Specs``).
3. Noise model information (``acorn.noise_model`` of class ``NoiseModel``).

These 3 attributes are accessed in the following ways (note that the specs shown below may be out of date):

.. code:: python

    print(acorn.specs)
    # Specs(qubits_specs=..., edges_specs=...)

    print(acorn.specs.qubits_specs)
    """
    [_QubitSpecs(id=0, fRO=0.938, f1QRB=0.9815, T1=1.52e-05, T2=7.2e-06),
     _QubitSpecs(id=1, fRO=0.958, f1QRB=0.9907, T1=1.76e-05, T2=7.7e-06),
     _QubitSpecs(id=2, fRO=0.97, f1QRB=0.9813, T1=1.82e-05, T2=1.08e-05),
     _QubitSpecs(id=3, fRO=0.886, f1QRB=0.9908, T1=3.1e-05, T2=1.68e-05),
     _QubitSpecs(id=4, fRO=0.953, f1QRB=0.9887, T1=2.3e-05, T2=5.2e-06),
     _QubitSpecs(id=5, fRO=0.965, f1QRB=0.9645, T1=2.22e-05, T2=1.11e-05),
     _QubitSpecs(id=6, fRO=0.84, f1QRB=0.9905, T1=2.68e-05, T2=2.68e-05),
     _QubitSpecs(id=7, fRO=0.925, f1QRB=0.9916, T1=2.94e-05, T2=1.3e-05),
     _QubitSpecs(id=8, fRO=0.947, f1QRB=0.9869, T1=2.45e-05, T2=1.38e-05),
     _QubitSpecs(id=9, fRO=0.927, f1QRB=0.9934, T1=2.08e-05, T2=1.11e-05),
     _QubitSpecs(id=10, fRO=0.942, f1QRB=0.9916, T1=1.71e-05, T2=1.06e-05),
     _QubitSpecs(id=11, fRO=0.9, f1QRB=0.9901, T1=1.69e-05, T2=4.9e-06),
     _QubitSpecs(id=12, fRO=0.942, f1QRB=0.9902, T1=8.2e-06, T2=1.09e-05),
     _QubitSpecs(id=13, fRO=0.921, f1QRB=0.9933, T1=1.87e-05, T2=1.27e-05),
     _QubitSpecs(id=14, fRO=0.947, f1QRB=0.9916, T1=1.39e-05, T2=9.4e-06),
     _QubitSpecs(id=16, fRO=0.948, f1QRB=0.9906, T1=1.67e-05, T2=7.5e-06),
     _QubitSpecs(id=17, fRO=0.921, f1QRB=0.9895, T1=2.4e-05, T2=8.4e-06),
     _QubitSpecs(id=18, fRO=0.93, f1QRB=0.9496, T1=1.69e-05, T2=1.29e-05),
     _QubitSpecs(id=19, fRO=0.93, f1QRB=0.9942, T1=2.47e-05, T2=9.8e-06)]
    """

    print(acorn.isa)
    # ISA(qubits=..., edges=...)

    print(acorn.isa.edges)
    """
    [Edge(targets=[0, 5], type='CZ', dead=False),
     Edge(targets=[0, 6], type='CZ', dead=False),
     Edge(targets=[1, 6], type='CZ', dead=False),
     Edge(targets=[1, 7], type='CZ', dead=False),
     Edge(targets=[2, 7], type='CZ', dead=False),
     Edge(targets=[2, 8], type='CZ', dead=False),
     Edge(targets=[4, 9], type='CZ', dead=False),
     Edge(targets=[5, 10], type='CZ', dead=False),
     Edge(targets=[6, 11], type='CZ', dead=False),
     Edge(targets=[7, 12], type='CZ', dead=False),
     Edge(targets=[8, 13], type='CZ', dead=False),
     Edge(targets=[9, 14], type='CZ', dead=False),
     Edge(targets=[10, 15], type='CZ', dead=False),
     Edge(targets=[10, 16], type='CZ', dead=False),
     Edge(targets=[11, 16], type='CZ', dead=False),
     Edge(targets=[11, 17], type='CZ', dead=False),
     Edge(targets=[12, 17], type='CZ', dead=False),
     Edge(targets=[12, 18], type='CZ', dead=False),
     Edge(targets=[13, 18], type='CZ', dead=False),
     Edge(targets=[13, 19], type='CZ', dead=False),
     Edge(targets=[14, 19], type='CZ', dead=False)]
    """

    print(acorn.noise_model)
    # NoiseModel(gates=[KrausModel(...) ...] ...)


Additionally, the ``Specs`` class provides methods for access specs info across the chip in a more
succinct manner:

.. code:: python

    acorn.specs.T1s()
    # {0: 1.52e-05, 1: 1.76e-05, 2: 1.82e-05, 3: 3.1e-05, ...}

    acorn.specs.fCZs()
    # {(0, 5): 0.888, (0, 6): 0.8, (1, 6): 0.837, (1, 7): 0.87, ...}

With these tools provided by the ``Device`` class, users may learn more about Rigetti hardware, and
construct programs tailored specifically to that hardware. The ``Device`` class serves two additional
uses:

* The ISA associated to Acorn can be modified by the user to target ``CPHASE`` rather than ``CZ`` on
  any collection of qubit-qubit links not including 1-6.  Passing such a customized ISA to Forest as
  part of a call to ``.run`` or ``.run_and_measure`` will enable compilation utilizing ``CPHASE``
  as a native gate (although the compiler will continue to prefer ``CZ`` to ``CPHASE(π)`` specifically,
  due to its generally higher fidelity on 19Q-Acorn).
* It can be used to seed a QVM with characteristics of the device, supporting noisy simulation. For
  more information on this, see the next section.

Simulating the QPU using the QVM
--------------------------------

The QVM is a powerful tool for testing quantum programs before executing them on the QPU. In
addition to the ``noise.py`` module for generating custom noise models for simulating noise on the
QVM, pyQuil provides a simple interface for loading the QVM with noise models tailored to Rigetti's
available QPUs, in just one modified line of code. This is made possible via the ``Device`` class,
which holds hardware specification information, noise model information, and instruction set
architecture (ISA) information regarding connectivity. This information is held in the ``Specs``,
``ISA`` and ``NoiseModel`` attributes of the ``Device`` class, respectively.

Specifically, to load a QVM with the ``NoiseModel`` information from a ``Device``, all that is
required is to provide a ``Device`` object to the QVM during initialization:

.. code:: python

    from pyquil.api import get_devices, QVMConnection

    acorn = get_devices(as_dict=True)['19Q-Acorn']
    qvm = QVMConnection(acorn)

By simply providing a device during QVM initialization, all programs executed on this QVM will, by
default, have noise applied that is characteristic of the corresponding Rigetti QPU (in the case
above, the ``acorn`` device). One may then efficiently test realistic quantum algorithms on the QVM,
in advance of running those programs on the QPU.

Retune Interruptions
--------------------

Because the QPU is a physical device, it is occasionally taken offline for recalibration.
This offline period typically lasts 10-40 minutes, depending upon QPU characteristics and other
external factors.  During this period, the QPU will be listed as offline, and it will reject
new jobs (but pending jobs will remain queued).  When the QPU resumes activity, its performance
characteristics may be slightly different (in that different gates may enjoy different process
fidelities).

Acorn QPU Properties
~~~~~~~~~~~~~~~~~~~~~~~~

The quantum processor consists of 20 superconducting transmon qubits with fixed capacitive coupling
in the planar lattice design shown in Fig. 1.

.. note::

  While this chip was fabricated with 20 qubits, 16 are currently available for programming.

The resonant frequencies of qubits 0–4 and 10–14 are
tunable while qubits 5–9 and 15–19 are fixed. The former have two Josephson junctions in an
asymmetric SQUID geometry to provide roughly 1 GHz of frequency tunability, and flux-insensitive
“sweet spots” near

:math:`\omega^{\textrm{max}}_{01}/2\pi\approx 4.5 \, \textrm{GHz}`

and

:math:`\omega^{\textrm{min}}_{01}/2\pi\approx 3.0 \, \textrm{GHz}`.

These tunable devices are coupled to bias lines for AC and DC flux delivery. Each qubit is
capacitively coupled to a quasi-lumped element resonator for dispersive readout of the qubit state.
Single-qubit control is effected by applying microwave drives at the resonator ports. Two-qubit
gates are activated via RF drives on the flux bias lines.

Due to a fabrication defect, qubit 3 is not tunable, which prohibits operation of the two-qubit
parametric gate described below between qubit 3 and its neighbors. Additionally, qubits 2, 15, and
18 are not accessible for quantum computation through Forest. Consequently, we will treat this
as a 16-qubit processor.

.. figure:: images/acorn.png
    :width: 540px
    :align: center
    :height: 300px
    :alt: 19Q-Acorn
    :figclass: align-center

    :math:`\textbf{Figure 1 $|$ Connectivity of Rigetti 19Q. a,}` Chip schematic showing tunable
    transmons (green circles) capacitively coupled to fixed-frequency transmons
    (blue circles). :math:`\textbf{b}`, Optical chip image. Note that some couplers have
    been dropped to produce a lattice with three-fold, rather than four-fold
    connectivity.

1-Qubit Gate Performance
------------------------

The device is characterized by several parameters:

- :math:`\omega_\textrm{01}/2\pi` is the qubit transition frequency
- :math:`\omega_\textrm{r}/2\pi` is the resonator frequency
- :math:`\eta/2\pi` is the anharmonicity of the qubit
- :math:`g/2\pi` is the coupling strength between a qubit and a resonator
- :math:`\lambda/2\pi` is the coupling strength between two neighboring qubits

In Rigetti 19Q, each tunable qubit is capacitively coupled to one-to-three fixed-frequency qubits.
We use a parametric flux modulation to activate a controlled Z gate between tunable and fixed
qubits. The typical time-scale of these entangling gates is in the range 100–250 ns.

Table 1 summarizes the main performance parameters of Rigetti 19Q. The resonator and qubit
frequencies are measured with standard spectroscopic techniques. The relaxation time :math:`T_1` is
extracted from repeated inversion recovery experiments. Similarly, the coherence time :math:`T^*_2`
is measured with repeated Ramsey fringe experiments. Single-qubit gate fidelities are estimated
with randomized benchmarking protocols in which a sequence of :math:`m` Clifford gates is applied
to the qubit followed by a measurement on the computational basis. The sequence of Clifford gates
are such that the first :math:`m-1` gates are chosen uniformly at random from the Clifford group,
while the last Clifford gate is chosen to bring the state of the system back to the initial state.
This protocol is repeated for different values of :math:`m\in \{2,4,8,16,32,64,128\}`. The reported
single-qubit gate fidelity is related to the randomized benchmarking decay constant :math:`p` in
the following way: :math:`\mathsf{F}_\textrm{1q} = p +(1-p)/2`. Finally, the readout assignment
fidelities are extracted with dispersive readouts combined with a linear classifier trained on
:math:`|0\rangle` and :math:`|1\rangle` state preparation for each qubit. The reported readout
assignment fidelity is given by expression :math:`\mathsf{F}_\textrm{RO} = [p(0|0)+p(1|1)]/2`, where
:math:`p(b|a)` is the probability of measuring the qubit in state :math:`b` when prepared in state
:math:`a`.

.. csv-table:: :math:`\textbf{Table 1 | Rigetti 19Q performance}`
   :widths: 10, 10, 10, 10, 10, 10, 10
   :stub-columns: 1

   ,:math:`\omega^{\textrm{max}}_{\textrm{r}}/2\pi`,:math:`\omega^{\textrm{max}}_{01}/2\pi`,:math:`T_1`,:math:`T^*_2`,:math:`\mathsf{F}_{\textrm{1q}}`,:math:`\mathsf{F}_{\textrm{RO}}`
   ,:math:`\textrm{MHz}`,:math:`\textrm{MHz}`,:math:`\mu\textrm{s}`,:math:`\mu\textrm{s}`,,
   0 ,5592,4372,17.98,7.47,0.982,0.918
   1 ,5703,4257,24.27,8.17,0.983,0.846
   2 ,5599,3069,8.5,7.47,0.976,0.78
   3 ,5708,3829,31.0,16.8,0.9908,0.886
   4 ,5633,4332,18.01,2.79,0.987,0.962
   5 ,5178,3658,17.76,10.05,0.973,0.932
   6 ,5356,3789,14.15,10.18,0.983,0.92
   7 ,5164,3531,11.94,9.08,0.991,0.803
   8 ,5367,3681,23.7,11.47,0.987,0.948
   9 ,5201,3665,17.68,10.43,0.992,0.918
   10,5801,4564,11.17,4.69,0.983,0.824
   11,5511,4238,20.31,10.3,0.985,0.878
   12,5825,4569,13.0,8.0,0.97,0.963
   13,5523,4384,12.2,8.54,0.980,0.954
   14,5848,4517,18.83,2.98,0.983,0.959
   15,5093,3716,22,3.0,0.808,0.705
   16,5298,3816,21.4,11.22,0.989,0.912
   17,5097,3428,17.34,8.4,0.991,0.87
   18,5301,3864,14.0,0.1,0.976,0.893
   19,5108,3535,22.4,9.13,0.987,0.945



Qubit-Qubit Coupling
--------------------

The coupling strength between two qubits can be extracted from a precise measurement of the shift
in qubit frequency after the neighboring qubit is in the excited state. This protocol consists of
two steps: a :math:`\pi` pulse is applied to the first qubit, followed by a Ramsey fringe
experiment on the second qubit which precisely determines its
transition frequency (see Fig. 2a). The effective shift is denoted by
:math:`\chi_\textrm{qq}` and typical values are in the range
:math:`\approx 100 \, \textrm{kHz}`. The coupling strength :math:`\lambda` between the two qubits
can be calculated in the following way:

  .. math::

     \lambda^{(1,2)} = \sqrt{\left|\frac{\chi^{(1,2)}_\textrm{qq} \left[\,f^\textrm{(1)}_{01}-f^\textrm{(2)}_{12}\right]\left[\,f^\textrm{(1)}_{12}-f^\textrm{(2)}_{01}\right]}{2(\eta_1+\eta_2)}\right|}

Figure 2b shows the coupling strength for our device. This quantity is crucial to predict the gate
time of our parametric entangling gates.

.. figure:: images/acorn_coupling.png
    :width: 500px
    :align: center
    :height: 300px
    :alt: alternate text
    :figclass: align-center

    :math:`\textbf{Figure 2 $|$ Coupling strength. a,}` Quantum circuit
    implemented to measure the qubit-qubit effective frequency shift.
    :math:`\textbf{b,}` Capacitive coupling between neighboring qubits expressed in MHz.


2-Qubit Gate Performance
------------------------

Table 2 shows the two-qubit gate performance of Rigetti 19Q. These parameters refer to parametric
CZ gates performed on one pair at a time. We analyze these CZ gates through quantum process
tomography (QPT). This procedure starts by applying local rotations to the two qubits taken from
the set :math:`\{I,R_x(\pi/2),R_y(\pi/2),R_x(\pi)\}`, followed by a CZ gate and
post-rotations that bring the qubit states back to the computational basis. QPT involves the
analysis of :math:`16\times16 =256` different experiments, each of which we repeat :math:`500`
times. The reported process tomography fidelity :math:`\mathsf{F}^\textrm{cptp}_\textrm{PT}`
is the fidelity of the measured process compared with the ideal process, computed imposing complete positivity (cp) and trace preservation (tp) constraints.

.. csv-table:: :math:`\textbf{Table 2 | Rigetti 19Q two-qubit gate performance}`
   :widths: 10, 10, 10, 10
   :stub-columns: 1

   ,:math:`f_\textrm{m}`,:math:`t_\textrm{CZ}`,:math:`\mathsf{F}^\textrm{cptp}_{\textrm{PT}}`
   ,:math:`\textrm{MHz}`,ns
   0 - 5, 190,277.35,0.83
   1 - 6, 129.998,165.99,0.892
   1 - 7, 92,198.25,0.89
   4 - 9, 191.8, 190.9,0.871
   5 - 10, 285.5,131.4,0.813
   6 - 11, 140,140.18,0.837
   7 - 12, 235.48,264.12,0.818
   8 - 13, 167.67,193.11,0.899
   9 - 14, 221,253.19,0.827
   10 - 16, 342.5,137.82,0.848
   11 - 16, 137.37,181.36,0.898
   11 - 17, 92,200,0.894
   12 - 17, 214.96,221.41,0.851
   13 - 19, 163,201.96,0.827
   14 - 19, 221,253.19,0.8496

Additionally, native ``CPHASE`` gates are available on some qubit-qubit links,
under the proviso that they are still under development, and so their performance is typically
below that of ``CZ``.  Due to the ongoing nature of the work, we decline to quote precise
performance characteristics here.
