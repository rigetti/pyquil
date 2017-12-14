..  _qpu_overview:

The Rigetti QPU
===============

A quantum processor unit (QPU) is a physical device that contains a number of interconnected qubits.
This page presents technical details and average performance of *Acorn*, the 19Q Rigetti QPU device
that is made available for quantum computation through the cloud. This device has been designed,
fabricated and packaged at Rigetti Computing.


Acorn QPU properties
~~~~~~~~~~~~~~~~~~~~

The quantum processor consists of 20 superconducting transmon qubits with fixed capacitive coupling
in the planar lattice design shown in Fig. 1. The resonance frequencies of qubits 0–4 and 10–14 are
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
as a 19-qubit processor. This also means that **qubit 3 is not accessible** for quantum computation
through Forest.

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
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

The coupling strength between two qubits can be extracted from a precise measurement of the shift
in qubit frequency after the neighboring qubit is in the excited state. This protocol consists of
two steps: a :math:`\pi` pulse is applied to the first qubit, followed by a Ramsey fringe
experiment on the second qubit which precisely determines its
transition frequency (see Fig. 3a). The effective shift is denoted by
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
~~~~~~~~~~~~~~~~~~~~~~~~

Table 2 shows the two-qubit gate performance of Rigetti 19Q. These parameters refer to parametric
CZ gates performed on one pair at a time. We analyze these CZ gates through quantum process
tomography (QPT). This procedure starts by applying local rotations to the two qubits taken from
the set :math:`\{I,R_x(\pi/2),R_y(\pi/2),R_x(\pi)\}`, followed by a CZ gate and
post-rotations that bring the qubit states back to the computational basis. QPT involves the
analysis of :math:`16\times16 =256` different experiments, each of which we repeat :math:`500`
times. The reported process tomography fidelity :math:`\mathsf{F}^\textrm{cptp}_\textrm{PT}`
indicates the fidelity between the ideal process and the measured process imposing complete
positivity (cp) and trace preservation (tp) constraints. The quantity
:math:`\mathsf{F}_\textrm{PT}` is instead extracted without cptp constraints on the estimated map.

.. csv-table:: :math:`\textbf{Table 2 | Rigetti 19Q two-qubit gate performance}`
   :widths: 10, 10, 10, 10, 10, 10
   :stub-columns: 1


   ,:math:`A_0`,:math:`f_\textrm{m}`,:math:`t_\textrm{CZ}`,:math:`\mathsf{F}^\textrm{cptp}_{\textrm{PT}}`,:math:`\mathsf{F}_{\textrm{PT}}`
   ,:math:`\Phi/\Phi_0`,:math:`\textrm{MHz}`,ns
   0 - 5 ,0.27,94.5,168,0.936,0.966
   0 - 6 ,0.36,123.9,197,0.889,0.900
   1 - 6 ,0.37,137.1,173,0.888,0.948
   1 - 7 ,0.59,137.9,179,0.919,0.974
   2 - 7 ,0.62,87.4,160,0.817,0.860
   2 - 8, 0.23,55.6,189,0.906,0.918
   4 - 9, 0.43,183.6,122,0.854,0.876
   5 - 10,0.60,152.9,145,0.870,0.902
   6 - 11 ,0.38,142.4,180,0.838,0.927
   7 - 12 ,0.60,241.9,214,0.87,0.890
   8 - 13,0.40,152.0,185,0.881,0.895
   9 - 14,0.62,130.8,139,0.872,0.937
   10 - 15,0.53,142.1,154,0.854,0.875
   10 - 16,0.43,170.3,180,0.838,0.847
   11 - 16,0.38,160.6,155,0.891,0.903
   11 - 17,0.29,85.7,207,0.844,0.875
   12 - 17,0.36,177.1,184,0.876,0.908
   12 - 18,0.28,113.9,203,0.886,0.923
   13 - 18,0.24,66.2,152,0.936,0.975
   13 - 19,0.62,109.6,181,0.921,0.941
   14 - 19,0.59,188.1,142,0.797,0.906


Using the QPU
~~~~~~~~~~~~~

To maintain above performance levels, Rigetti Forest periodically takes the QPU offline to retune
single-qubit and two-qubit gates. To access Acorn for running quantum algorithms, see
:ref:`qpu_usage` for a tutorial.
