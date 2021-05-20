Simulators
==========

QVMs promise to behave like a real QPU. However, under-the-hood there is usually a simulation
that has fewer constraints than a physical quantum processor. For example, in a wavefunction (or statevector)
simulation, you can directly inspect amplitudes and probabilities.


.. currentmodule:: pyquil
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    ~pyquil.api.WavefunctionSimulator
    ~pyquil.simulation.ReferenceWavefunctionSimulator
    ~pyquil.simulation.ReferenceDensitySimulator
    ~pyquil.simulation.NumpyWavefunctionSimulator


Reference Utilities
-------------------

.. currentmodule:: pyquil.simulation.tools
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    lifted_pauli
    lifted_gate
    program_unitary
    all_bitstrings

