..  _qpu_overview:

The Rigetti QPU
===============

A quantum processing unit (QPU) is a physical device that contains a number of interconnected
qubits. This page presents technical details and average performance of **Acorn**, the 19Q Rigetti
QPU device that is made available for quantum computation through the cloud. This device has
been designed, fabricated and packaged at Rigetti Computing.


19Q Acorn QPU Properties
~~~~~~~~~~~~~~~~~~~~~~~~

The quantum processor consists of 20 superconducting transmon qubits with fixed capacitive coupling
in the planar lattice design shown in Fig. 1. The resonant frequencies of qubits 0–4 and 10–14 are
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
parametric gate described below between qubit 3 and its neighbors. Consequently, we will treat this
as a 19-qubit processor. In addition, we recently have disabled qubit 15. This means that
**qubits 3 and 15 are not accessible** for quantum computation through Forest.

.. figure:: images/acorn.png
    :width: 540px
    :align: center
    :height: 300px
    :alt: alternate text
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
   :widths: 10, 10, 10, 10, 10, 10, 10, 10
   :stub-columns: 1

   ,:math:`\omega^{\textrm{max}}_{\textrm{r}}/2\pi`,:math:`\omega^{\textrm{max}}_{01}/2\pi`,:math:`\eta/2\pi`,:math:`T_1`,:math:`T^*_2`,:math:`\mathsf{F}_{\textrm{1q}}`,:math:`\mathsf{F}_{\textrm{RO}}`
   ,:math:`\textrm{MHz}`,:math:`\textrm{MHz}`,:math:`\textrm{MHz}`,:math:`\mu\textrm{s}`,:math:`\mu\textrm{s}`,,
   0 ,5592,4386,-208,**15.2** :math:`\pm` 2.5,**7.2** :math:`\pm` 0.7,0.9815,0.938
   1 ,5703,4292,-210,**17.6** :math:`\pm` 1.7,**7.7** :math:`\pm` 1.4,0.9907,0.958
   2 ,5599,4221,-142,**18.2** :math:`\pm` 1.1,**10.8** :math:`\pm` 0.6,0.9813,0.97
   3 ,5708,3829,-224,**31.0** :math:`\pm` 2.6,**16.8** :math:`\pm` 0.8,0.9908,0.886
   4 ,5633,4372,-220,**23.0** :math:`\pm` 0.5,**5.2** :math:`\pm` 0.2,0.9887,0.953
   5 ,5178,3690,-224,**22.2** :math:`\pm` 2.1,**11.1** :math:`\pm` 1.0,0.9645,0.965
   6 ,5356,3809,-208,**26.8** :math:`\pm` 2.5,**26.8** :math:`\pm` 2.5,0.9905,0.84
   7 ,5164,3531,-216,**29.4** :math:`\pm` 3.8,**13.0** :math:`\pm` 1.2,0.9916,0.925
   8 ,5367,3707,-208,**24.5** :math:`\pm` 2.8,**13.8** :math:`\pm` 0.4,0.9869,0.947
   9 ,5201,3690,-214,**20.8** :math:`\pm` 6.2,**11.1** :math:`\pm` 0.7,0.9934,0.927
   10,5801,4595,-194,**17.1** :math:`\pm` 1.2,**10.6** :math:`\pm` 0.5,0.9916,0.942
   11,5511,4275,-204,**16.9** :math:`\pm` 2.0,**4.9** :math:`\pm` 1.0,0.9901,0.900
   12,5825,4600,-194,**8.2**  :math:`\pm` 0.9,**10.9** :math:`\pm` 1.4,0.9902,0.942
   13,5523,4434,-196,**18.7** :math:`\pm` 2.0,**12.7** :math:`\pm` 0.4,0.9933,0.921
   14,5848,4552,-204,**13.9** :math:`\pm` 2.2,**9.4** :math:`\pm` 0.7,0.9916,0.947
   15,5093,3733,-230,**20.8** :math:`\pm` 3.1,**7.3** :math:`\pm` 0.4,0.9852,0.970
   16,5298,3854,-218,**16.7** :math:`\pm` 1.2,**7.5** :math:`\pm` 0.5,0.9906,0.948
   17,5097,3574,-226,**24.0** :math:`\pm` 4.2,**8.4** :math:`\pm` 0.4,0.9895,0.921
   18,5301,3877,-216,**16.9** :math:`\pm` 2.9,**12.9** :math:`\pm` 1.3,0.9496,0.930
   19,5108,3574,-228,**24.7** :math:`\pm` 2.8,**9.8** :math:`\pm` 0.8,0.9942,0.930



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
   :widths: 10, 10, 10, 10, 10
   :stub-columns: 1


   ,:math:`A_0`,:math:`f_\textrm{m}`,:math:`t_\textrm{CZ}`,:math:`\mathsf{F}^\textrm{cptp}_{\textrm{PT}}`
   ,:math:`\Phi/\Phi_0`,:math:`\textrm{MHz}`,ns
   0 - 5 ,0.27,94.5,168,0.936
   0 - 6 ,0.36,123.9,197,0.889
   1 - 6 ,0.37,137.1,173,0.888
   1 - 7 ,0.59,137.9,179,0.919
   2 - 7 ,0.62,87.4,160,0.817
   2 - 8, 0.23,55.6,189,0.906
   4 - 9, 0.43,183.6,122,0.854
   5 - 10,0.60,152.9,145,0.870
   6 - 11 ,0.38,142.4,180,0.838
   7 - 12 ,0.60,241.9,214,0.87
   8 - 13,0.40,152.0,185,0.881
   9 - 14,0.62,130.8,139,0.872
   10 - 15,0.53,142.1,154,0.854
   10 - 16,0.43,170.3,180,0.838
   11 - 16,0.38,160.6,155,0.891
   11 - 17,0.29,85.7,207,0.844
   12 - 17,0.36,177.1,184,0.876
   12 - 18,0.28,113.9,203,0.886
   13 - 18,0.24,66.2,152,0.936
   13 - 19,0.62,109.6,181,0.921
   14 - 19,0.59,188.1,142,0.797


Using the QPU
~~~~~~~~~~~~~

The broad strokes of working with the QPU-based pyQuil stack are identical to using the QVM-based
stack: the library ``pyquil.api`` supplies an object class ``QPUConnection`` which mediates the
transmission of Quil programs to the QPU, encoded as ``pyquil.quil.Program`` objects, as well as
the receipt of job results, encoded as bitstring lists.

.. note::

    User permissions for QPU access must be enabled by a Forest administrator.  ``QPUConnection``
    calls will automatically fail without these user permissions.  Speak to a Forest administrator
    for information about upgrading your access plan.

Detecting the Available QPUs and Their Structure
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

These 3 attributes are accessed in the following ways:

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
construct programs tailored specifically to that hardware. In addition, the ``Device`` class serves
as a powerful tool for seeding a QVM with characteristics of the device. For more information on
this, see the next section.

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
