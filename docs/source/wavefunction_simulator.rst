.. _wavefunction_simulator:

The Wavefunction Simulator
==========================

Formerly a part of the QVM object in pyQuil, the Wavefunction Simulator allows you to directly inspect the wavefunction
of a quantum state prepared by your program. Because of the probabilistic nature of quantum information, the programs you'll
be running on the QPU can give a distribution of outputs. When running on the QPU or QVM, you would aggregate results
(anywhere from tens of trials to 100k+!) that you can sample to get back a distribution.

With the Wavefunction Simulator, you can look at the distribution without having to collect samples from your program.
This can save a lot of time for small programs. Let's walk through a basic example of using WavefunctionSimulator:

.. code:: python

    from pyquil import Program
    from pyquil.gates import *
    from pyquil.api import WavefunctionSimulator
    wf_sim = WavefunctionSimulator()
    coin_flip = Program(H(0))
    wf_sim.wavefunction(coin_flip)

.. parsed-literal::

    <pyquil.wavefunction.Wavefunction at 0x1088a2c10>

The return value is a Wavefunction object that stores the amplitudes of the quantum state. We can print this object

.. code:: python

    coin_flip = Program(H(0))
    wavefunction = wf_sim.wavefunction(coin_flip)
    print(wavefunction)

.. parsed-literal::

  (0.7071067812+0j)|0> + (0.7071067812+0j)|1>

to see the amplitudes listed as a sum of computational basis states. We can index into those
amplitudes directly or look at a dictionary of associated outcome probabilities.

.. code:: python

  assert wavefunction[0] == 1 / np.sqrt(2)
  # The amplitudes are stored as a numpy array on the Wavefunction object
  print(wavefunction.amplitudes)
  prob_dict = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes as a dict
  print(prob_dict)
  prob_dict.keys() # these store the bitstring outcomes
  assert len(wavefunction) == 1 # gives the number of qubits

.. parsed-literal::

  [ 0.70710678+0.j  0.70710678+0.j]
  {'1': 0.49999999999999989, '0': 0.49999999999999989}


It is important to remember that this ``wavefunction`` method is a useful debugging tool for small quantum systems, and
cannot be feasibly obtained on a quantum processor.

Multi-Qubit Basis Enumeration
-----------------------------

The WavefunctionSimulator enumerates bitstrings such that qubit ``0`` is the least significant bit (LSB)
and therefore on the right end of a bitstring as shown in the table below which contains some
examples.

=============== ============= ===== ========= ========= =========
 bitstring       qubit_(n-1)   ...   qubit_2   qubit_1   qubit_0
=============== ============= ===== ========= ========= =========
  1...101                  1   ...         1         0         1
  0...110                  0   ...         1         1         0
=============== ============= ===== ========= ========= =========



This convention is counter to that often found in the quantum computing literature where
bitstrings are often ordered such that the lowest-index qubit is on the left.
The vector representation of a wavefunction assumes the "canonical" ordering of basis elements.
I.e., for two qubits this order is ``00, 01, 10, 11``.
In the typical Dirac notation for quantum states, the tensor product of two different degrees of
freedom is not always explicitly understood as having a fixed order of those degrees of freedom.
This is in contrast to the kronecker product between matrices which uses the same mathematical
symbol and is clearly not commutative.
This, however, becomes important when writing things down as coefficient vectors or matrices:

.. math::

    \ket{0}_0 \otimes \ket{1}_1 = \ket{1}_1 \otimes \ket{0}_0
    = \ket{10}_{1,0} \equiv \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}

As a consequence there arise some subtle but important differences in the ordering of wavefunction
and multi-qubit gate matrix coefficients.
According to our conventions the matrix

.. math::

    U_{\rm CNOT(1,0)} \equiv
    \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0
    \end{pmatrix}

corresponds to the Quil instruction ``CNOT(1, 0)`` which is counter to how most other people in the
field order their tensor product factors (or more specifically their kronecker products).
In this convention ``CNOT(0, 1)`` is given by

.. math::

    U_{\rm CNOT(0,1)} \equiv
    \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0 \\
        0 & 1 & 0 & 0
    \end{pmatrix}

For additional information why we decided on this basis ordering check out our note
`Someone shouts, "|01000>!" Who is Excited? <https://arxiv.org/abs/1711.02086>`_.
