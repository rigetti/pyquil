
Simulating gate noise on the Rigetti Quantum Virtual Machine
============================================================


Quantum gate errors
-------------------

For a quantum gate given by its unitary operator :math:`U`, a "quantum
gate error" describes the scenario in which the actually induces
transformation deviates from :math:`\ket{\psi} \mapsto U\ket{\psi}`.
There are two basic types of quantum gate errors:

1. **coherent errors** are those that preserve the purity of the input
   state, i.e., instead of the above mapping we carry out a perturbed,
   but unitary operation :math:`\ket{\psi} \mapsto \tilde{U}\ket{\psi}`,
   where :math:`\tilde{U} \neq U`.
2. **incoherent errors** are those that do not preserve the purity of
   the input state, in this case we must actually represent the
   evolution in terms of density matrices. The state
   :math:`\rho := \ket{\psi}\bra{\psi}` is then mapped as

   .. math::


      \rho \mapsto \sum_{j=1}^n K_j\rho K_j^\dagger, 

   where the operators :math:`\{K_1, K_2, \dots, K_m\}` are called
   Kraus operators and must obey
   :math:`\sum_{j=1}^m K_j^\dagger K_j = I` to conserve the trace of
   :math:`\rho`. Maps expressed in the above form are called Kraus maps.
   It can be shown that every physical map on a finite dimensional
   quantum system can be represented as a Kraus map, though this
   representation is not generally unique. `You can find more
   information about quantum operations
   here <https://en.wikipedia.org/wiki/Quantum_operation#Kraus_operators>`__

In a way, coherent errors are *in principle* amendable by more precisely
calibrated control. Incoherent errors are more tricky.

Why do incoherent errors happen?
--------------------------------

When a quantum system (e.g., the qubits on a quantum processor) is not
perfectly isolated from its environment it generally co-evolves with the
degrees of freedom it couples to. The implication is that while the
total time evolution of system and environment can be assumed to be
unitary, restriction to the system state generally is not.

**Let's throw some math at this for clarity:** Let our total Hilbert
space be given by the tensor product of system and environment Hilbert
spaces: :math:`\mathcal{H} = \mathcal{H}_S \otimes \mathcal{H}_E`. Our
system "not being perfectly isolated" must be translated to the
statement that the global Hamiltonian contains a contribution that
couples the system and environment:

.. math::


   H = H_S \otimes I + I \otimes H_E + V

where :math:`V` non-trivally acts on both the system and the
environment. Consequently, even if we started in an initial state that
factorized over system and environment
:math:`\ket{\psi}_{S,0}\otimes \ket{\psi}_{E,0}` if everything evolves
by the Schrödinger equation

.. math::


   \ket{\psi_t} = e^{-i \frac{Ht}{\hbar}} \left(\ket{\psi}_{S,0}\otimes \ket{\psi}_{E,0}\right)

the final state will generally not admit such a factorization.

A toy model
-----------

**In this (somewhat technical) section we show how environment
interaction can corrupt an identity gate and derive its Kraus map.** For
simplicity, let us assume that we are in a reference frame in which both
the system and environment Hamiltonian's vanish :math:`H_S = 0, H_E = 0`
and where the cross-coupling is small even when multiplied by the
duration of the time evolution
:math:`\|\frac{tV}{\hbar}\|^2 \sim \epsilon \ll 1` (any operator norm
:math:`\|\cdot\|` will do here). Let us further assume that
:math:`V = \sqrt{\epsilon} V_S \otimes V_E` (the more general case is
given by a sum of such terms) and that the initial environment state
satisfies :math:`\bra{\psi}_{E,0} V_E\ket{\psi}_{E,0} = 0`. This turns
out to be a very reasonable assumption in practice but a more thorough
discussion exceeds our scope.

Then the joint system + environment state
:math:`\rho = \rho_{S,0} \otimes \rho_{E,0}` (now written as a density
matrix) evolves as

.. math::


   \rho \mapsto \rho' :=  e^{-i \frac{Vt}{\hbar}} \rho e^{+i \frac{Vt}{\hbar}}

Using the Baker-Campbell-Hausdorff theorem we can expand this to second
order in :math:`\epsilon`

.. math::


   \rho' = \rho - \frac{it}{\hbar} [V, \rho] - \frac{t^2}{2\hbar^2} [V, [V, \rho]] + O(\epsilon^{3/2})

We can insert the initially factorizable state
:math:`\rho = \rho_{S,0} \otimes \rho_{E,0}` and trace over the
environmental degrees of freedom to obtain


.. math::


   \begin{align}
   \rho_S' := \tr{\rho'}_E & = \rho_{S,0}  \underbrace{\tr{\rho_{E,0}}_{1}} - \frac{i\sqrt{\epsilon} t}{\hbar} \underbrace{\left[ V_S \rho_{S,0} \underbrace{\tr{V_E\rho_{E,0}}}_{\bra{\psi}_{E,0} V_E\ket{\psi}_{E,0} = 0} - \rho_{S,0}V_S  \underbrace{\tr{\rho_{E,0}V_E}}_{\bra{\psi}_{E,0} V_E\ket{\psi}_{E,0} = 0} \right]}_0 \\
   & - \frac{\epsilon t^2}{2\hbar^2} \left[ V_S^2\rho_{S,0}\tr{V_E^2 \rho_{E,0}} + \rho_{S,0} V_S^2 \tr{\rho_{E,0}V_E^2} - 2 V_S\rho_{S,0}V_S\tr{V_E \rho_{E,0}V_E}\right] \\
   & = \rho_{S,0} - \frac{\gamma}{2} \left[ V_S^2\rho_{S,0} + \rho_{S,0} V_S^2  - 2 V_S\rho_{S,0}V_S\right]
   \end{align}

where the coefficient in front of the second part is by our initial
assumption very small
:math:`\gamma := \frac{\epsilon t^2}{2\hbar^2}\tr{V_E^2 \rho_{E,0}} \ll 1`.
This evolution happens to be approximately equal to a Kraus map with
operators
:math:`K_1 := I - \frac{\gamma}{2} V_S^2, K_2:= \sqrt{\gamma} V_S`:

.. math::

   \begin{align}
   \rho_S \to \rho_S' &= K_1\rho K_1^\dagger + K_2\rho K_2^\dagger
    = \rho - \frac{\gamma}{2}\left[ V_S^2 \rho + \rho V_S^2\right] + \gamma V_S\rho_S V_S + O(\gamma^2)
   \end{align}

This agrees to :math:`O(\epsilon^{3/2})` with the result of our
derivation above. This type of derivation can be extended to many other
cases with little complication and a very similar argument is used to
derive the `Lindblad master
equation <https://en.wikipedia.org/wiki/Lindblad_equation>`__.

Support for noisy gates on the Rigetti QVM
==========================================

As of today, users of our Forest API can annotate their QUIL programs by
certain pragma statements that inform the QVM that a particular gate on
specific target qubits should be replaced by an imperfect realization
given by a Kraus map.

But the QVM propagates *pure states*: How does it simulate noisy gates?
-----------------------------------------------------------------------

It does so by yielding the correct outcomes **in the average over many
executions of the QUIL program**: When the noisy version of a gate
should be applied the QVM makes a random choice which Kraus operator is
applied to the current state with a probability that ensures that the
average over many executions is equivalent to the Kraus map. In
particular, a particular Kraus operator :math:`K_j` is applied to
:math:`\ket{\psi}_S`

.. math::


   \ket{\psi'}_S = \frac{1}{\sqrt{p_j}} K_j \ket{\psi}_S

with probability
:math:`p_j:= \bra{\psi}_S K_j^\dagger K_j \ket{\psi}_S`. In the average
over many execution :math:`N \gg 1` we therefore find that

.. math::

   \begin{align}
   \overline{\rho_S'} & = \frac{1}{N} \sum_{n=1}^N \ket{\psi'_n}_S\bra{\psi'_n}_S \\
   & = \frac{1}{N} \sum_{n=1}^N p_{j_n}^{-1}K_{j_n}\ket{\psi'}_S \bra{\psi'}_SK_{j_n}^\dagger
   \end{align}

where :math:`j_n` is the chosen Kraus operator label in the :math:`n`-th
trial. This is clearly a Kraus map itself! And we can group identical
terms and rewrite it as

.. math::

   \begin{align}
   \overline{\rho_S'} & = 
     \sum_{\ell=1}^n \frac{N_\ell}{N}  p_{\ell}^{-1}K_{\ell}\ket{\psi'}_S \bra{\psi'}_SK_{\ell}^\dagger
   \end{align}

where :math:`N_{\ell}` is the number of times that Kraus operator label
:math:`\ell` was selected. For large enough :math:`N` we know that
:math:`N_{\ell} \approx N p_\ell` and therefore

.. math::

   \begin{align}
   \overline{\rho_S'} \approx \sum_{\ell=1}^n K_{\ell}\ket{\psi'}_S \bra{\psi'}_SK_{\ell}^\dagger
   \end{align}

which proves our claim. **The consequence is that noisy gate simulations
must generally be repeated many times to obtain representative
results**.

How do I get started?
---------------------

1. Come up with a good model for your noise. We will provide some
   examples below and may add more such examples to our public
   repositories over time. Alternatively, you can characterize the gate
   under consideration using `Quantum Process
   Tomography <https://arxiv.org/abs/1202.5344>`__ or `Gate Set
   Tomography <http://www.pygsti.info/>`__ and use the resulting process
   matrices to obtain a very accurate noise model for a particular QPU.
2. Define your Kraus operators as a list of numpy arrays
   ``kraus_ops = [K1, K2, ..., Km]``.
3. For your QUIL program ``p``, call:

   ::

       p.define_noisy_gate("MY_NOISY_GATE", [q1, q2], kraus_ops)

   where you should replace ``MY_NOISY_GATE`` with the gate of interest
   and ``q1, q2`` the indices of the qubits.

**Scroll down for some examples!**

.. code:: ipython3

    from __future__ import print_function
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binom
    import matplotlib.colors as colors
    %matplotlib inline

.. code:: ipython3

    from pyquil.quil import Program, MEASURE
    from pyquil.api.qvm import QVMConnection
    from pyquil.job_results import wait_for_job
    from pyquil.gates import CZ, H, I, X

.. code:: ipython3

    cxn = QVMConnection()

Example 1: Amplitude damping
============================

Amplitude damping channels are imperfect identity maps with Kraus
operators

.. math::


   K_1 = \begin{pmatrix} 
   1 & 0 \\
   0 & \sqrt{1-p}
   \end{pmatrix} \\
   K_2 = \begin{pmatrix} 
   0 & \sqrt{p} \\
   0 & 0
   \end{pmatrix}

where :math:`p` is the probability that a qubit in the :math:`\ket{1}`
state decays to the :math:`\ket{0}` state.

.. code:: ipython3

    def append_damping_to_gate(gate, damp_prob=.1):
        """
        Generate the Kraus operators corresponding to a given unitary 
        single qubit gate followed by an amplitude damping noise channel.
        
        :params np.ndarray|list gate: The 2x2 unitary gate matrix.
        :params float damp_prob: The one-step damping probability.
        :return: A list [k1, k2] of the Kraus operators that parametrize the map.
        :rtype: list
        """
        damping_op = np.sqrt(damp_prob) * np.array([[0, 1],
                                                    [0, 0]])
        
        residual_kraus = np.diag([1, np.sqrt(1-damp_prob)])
        return [residual_kraus.dot(gate), damping_op.dot(gate)]

.. code:: ipython3

    %%time
    
    # single step damping probability
    damping_per_I = 0.02
    
    # number of program executions
    trials = 200
    
    results = []
    outcomes = []
    lengths = np.arange(0, 201, 10, dtype=int)
    for jj, num_I in enumerate(lengths):
        
        print("{}/{}, ".format(jj, len(lengths)), end="")
    
        
        p = Program(X(0))
        # want increasing number of I-gates
        p.inst([I(0) for _ in range(num_I)])
        p.inst(MEASURE(0, [0]))
        
        # overload identity I on qc 0
        p.define_noisy_gate("I", [0], append_damping_to_gate(np.eye(2), damping_per_I))
        cxn.random_seed = int(num_I)
        res = cxn.run(p, [0], trials=trials)
        results.append([np.mean(res), np.std(res) / np.sqrt(trials)])
        
    results = np.array(results)


.. parsed-literal::

    0/21, 1/21, 2/21, 3/21, 4/21, 5/21, 6/21, 7/21, 8/21, 9/21, 10/21, 11/21, 12/21, 13/21, 14/21, 15/21, 16/21, 17/21, 18/21, 19/21, 20/21, CPU times: user 138 ms, sys: 19.2 ms, total: 157 ms
    Wall time: 6.4 s


.. code:: ipython3

    dense_lengths = np.arange(0, lengths.max()+1, .2)
    survival_probs = (1-damping_per_I)**dense_lengths
    logpmf = binom.logpmf(np.arange(trials+1)[np.newaxis, :], trials, survival_probs[:, np.newaxis])/np.log(10)

.. code:: ipython3

    DARK_TEAL = '#48737F'
    FUSCHIA = "#D6619E"
    BEIGE = '#EAE8C6'
    cm = colors.LinearSegmentedColormap.from_list('anglemap', ["white", FUSCHIA, BEIGE], N=256, gamma=1.5)

.. code:: ipython3

    plt.figure(figsize=(14, 6))
    plt.pcolor(dense_lengths, np.arange(trials+1)/trials, logpmf.T, cmap=cm, vmin=-4, vmax=logpmf.max())
    plt.plot(dense_lengths, survival_probs, c=BEIGE, label="Expected mean")
    plt.errorbar(lengths, results[:,0], yerr=2*results[:,1], c=DARK_TEAL, 
                 label=r"noisy qvm, errorbars $ = \pm 2\hat{\sigma}$", marker="o")
    cb = plt.colorbar()
    cb.set_label(r"$\log_{10} \mathrm{Pr}(n_1; n_{\rm trials}, p_{\rm survival}(t))$", size=20)
    
    plt.title("Amplitude damping model of a single qubit", size=20)
    plt.xlabel(r"Time $t$ [arb. units]", size=14)
    plt.ylabel(r"$n_1/n_{\rm trials}$", size=14)
    plt.legend(loc="best", fontsize=18)
    plt.xlim(*lengths[[0, -1]])
    plt.ylim(0, 1)

.. image:: images/GateNoiseModels_14_1.png


Example 2: dephased CZ-gate
===========================

Dephasing is usually characterized through a qubit's :math:`T_2` time.
For a single qubit the dephasing Kraus operators are

.. math::


   K_1(p) = \sqrt{1-p} I_2 \\
   K_2(p) = \sqrt{p} \sigma_Z

where :math:`p = 1 - \exp(-T_2/T_{\rm gate})` is the probability that
the qubit is dephased over the time interval of interest, :math:`I_2` is
the :math:`2\times 2`-identity matrix and :math:`\sigma_Z` is the
Pauli-Z operator.

For two qubits, we must construct a Kraus map that has *four* different
outcomes:

1. No dephasing
2. Qubit 1 dephases
3. Qubit 2 dephases
4. Both dephase

The Kraus operators for this are given by

.. raw:: latex

   \begin{align}
   K'_1(p,q) = K_1(p)\otimes K_1(q) \\
   K'_2(p,q) = K_2(p)\otimes K_1(q) \\
   K'_3(p,q) = K_1(p)\otimes K_2(q) \\
   K'_4(p,q) = K_2(p)\otimes K_2(q) 
   \end{align}

where we assumed a dephasing probability :math:`p` for the first qubit
and :math:`q` for the second.

Dephasing is a *diagonal* error channel and the CZ gate is also
diagonal, therefore we can get the combined map of dephasing and the CZ
gate simply by composing :math:`U_{\rm CZ}` the unitary representation
of CZ with each Kraus operator

.. raw:: latex

   \begin{align}
   K^{\rm CZ}_1(p,q) = K_1(p)\otimes K_1(q)U_{\rm CZ} \\
   K^{\rm CZ}_2(p,q) = K_2(p)\otimes K_1(q)U_{\rm CZ} \\
   K^{\rm CZ}_3(p,q) = K_1(p)\otimes K_2(q)U_{\rm CZ} \\
   K^{\rm CZ}_4(p,q) = K_2(p)\otimes K_2(q)U_{\rm CZ} 
   \end{align}

**Note that this is not always accurate, because a CZ gate is often
achieved through non-diagonal interaction Hamiltonians! However, for
sufficiently small dephasing probabilities it should always provide a
good starting point.**

.. code:: ipython3

    def dephasing_kraus_map(p=.1):
        """
        Generate the Kraus operators corresponding to a dephasing channel.
    
        :params float p: The one-step dephasing probability.
        :return: A list [k1, k2] of the Kraus operators that parametrize the map.
        :rtype: list
        """
        return [np.sqrt(1-p)*np.eye(2), np.sqrt(p)*np.diag([1, -1])]
    
    def tensor_kraus_maps(k1, k2):
        """
        Generate the Kraus map corresponding to the composition
        of two maps on different qubits.
        
        :param list k1: The Kraus operators for the first qubit.
        :param list k2: The Kraus operators for the second qubit.
        :return: A list of tensored Kraus operators.
        """
        return [np.kron(k1j, k2l) for k1j in k1 for k2l in k2]
    
    
    def append_kraus_to_gate(kraus_ops, g):
        """
        Follow a gate `g` by a Kraus map described by `kraus_ops`.
        
        :param list kraus_ops: The Kraus operators.
        :param numpy.ndarray g: The unitary gate.
        :return: A list of transformed Kraus operators.
        """
        return [kj.dot(g) for kj in kraus_ops]


.. code:: ipython3

    %%time
    # single step damping probabilities
    ps = np.linspace(.001, .5, 200)
    
    # number of program executions
    trials = 500
    
    results = []
    
    for jj, p in enumerate(ps):
    
        corrupted_CZ = append_kraus_to_gate(
        tensor_kraus_maps(
            dephasing_kraus_map(p),
            dephasing_kraus_map(p)
        ), 
        np.diag([1, 1, 1, -1]))
    
        
        print("{}/{}, ".format(jj, len(ps)), end="")
        
        # make Bell-state
        p = Program(H(0), H(1), CZ(0,1), H(1))
        
        p.inst(MEASURE(0, [0]))
        p.inst(MEASURE(1, [1]))
        
        # overload identity I on qc 0
        p.define_noisy_gate("CZ", [0, 1], corrupted_CZ)
        cxn.random_seed = jj
        res = cxn.run(p, [0, 1], trials=trials)
        results.append(res)
        
    results = np.array(results)


.. parsed-literal::

    0/200, 1/200, 2/200, 3/200, 4/200, 5/200, 6/200, 7/200, 8/200, 9/200, 10/200, 11/200, 12/200, 13/200, 14/200, 15/200, 16/200, 17/200, 18/200, 19/200, 20/200, 21/200, 22/200, 23/200, 24/200, 25/200, 26/200, 27/200, 28/200, 29/200, 30/200, 31/200, 32/200, 33/200, 34/200, 35/200, 36/200, 37/200, 38/200, 39/200, 40/200, 41/200, 42/200, 43/200, 44/200, 45/200, 46/200, 47/200, 48/200, 49/200, 50/200, 51/200, 52/200, 53/200, 54/200, 55/200, 56/200, 57/200, 58/200, 59/200, 60/200, 61/200, 62/200, 63/200, 64/200, 65/200, 66/200, 67/200, 68/200, 69/200, 70/200, 71/200, 72/200, 73/200, 74/200, 75/200, 76/200, 77/200, 78/200, 79/200, 80/200, 81/200, 82/200, 83/200, 84/200, 85/200, 86/200, 87/200, 88/200, 89/200, 90/200, 91/200, 92/200, 93/200, 94/200, 95/200, 96/200, 97/200, 98/200, 99/200, 100/200, 101/200, 102/200, 103/200, 104/200, 105/200, 106/200, 107/200, 108/200, 109/200, 110/200, 111/200, 112/200, 113/200, 114/200, 115/200, 116/200, 117/200, 118/200, 119/200, 120/200, 121/200, 122/200, 123/200, 124/200, 125/200, 126/200, 127/200, 128/200, 129/200, 130/200, 131/200, 132/200, 133/200, 134/200, 135/200, 136/200, 137/200, 138/200, 139/200, 140/200, 141/200, 142/200, 143/200, 144/200, 145/200, 146/200, 147/200, 148/200, 149/200, 150/200, 151/200, 152/200, 153/200, 154/200, 155/200, 156/200, 157/200, 158/200, 159/200, 160/200, 161/200, 162/200, 163/200, 164/200, 165/200, 166/200, 167/200, 168/200, 169/200, 170/200, 171/200, 172/200, 173/200, 174/200, 175/200, 176/200, 177/200, 178/200, 179/200, 180/200, 181/200, 182/200, 183/200, 184/200, 185/200, 186/200, 187/200, 188/200, 189/200, 190/200, 191/200, 192/200, 193/200, 194/200, 195/200, 196/200, 197/200, 198/200, 199/200, CPU times: user 1.17 s, sys: 166 ms, total: 1.34 s
    Wall time: 1min 49s


.. code:: ipython3

    Z1s = (2*results[:,:,0]-1.)
    Z2s = (2*results[:,:,1]-1.)
    Z1Z2s = Z1s * Z2s
    
    Z1m = np.mean(Z1s, axis=1)
    Z2m = np.mean(Z2s, axis=1)
    Z1Z2m = np.mean(Z1Z2s, axis=1)

.. code:: ipython3

    plt.figure(figsize=(14, 6))
    plt.axhline(y=1.0, color=FUSCHIA, alpha=.5, label="Bell state")
    
    plt.plot(ps, Z1Z2m, "x", c=FUSCHIA, label=r"$\overline{Z_1 Z_2}$")
    plt.plot(ps, 1-2*ps, "--", c=FUSCHIA, label=r"$\langle Z_1 Z_2\rangle_{\rm theory}$")
    
    plt.plot(ps, Z1m, "o", c=DARK_TEAL, label=r"$\overline{Z}_1$")
    plt.plot(ps, 0*ps, "--", c=DARK_TEAL, label=r"$\langle Z_1\rangle_{\rm theory}$")
    
    plt.plot(ps, Z2m, "d", c="k", label=r"$\overline{Z}_2$")
    plt.plot(ps, 0*ps, "--", c="k", label=r"$\langle Z_2\rangle_{\rm theory}$")
    
    plt.xlabel(r"Dephasing probability $p$", size=18)
    plt.ylabel(r"$Z$-moment", size=18)
    plt.title(r"$Z$-moments for a Bell-state prepared with dephased CZ", size=18)
    plt.xlim(0, .5)
    plt.legend(fontsize=18)


.. image:: images/GateNoiseModels_20_1.png

