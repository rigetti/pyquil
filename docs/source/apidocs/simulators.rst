Simulators
==========

QVMs promise to behave like a real QPU. However, under-the-hood there is usually a simulation
that has fewer constraints than a physical device. For example, in a wavefunction (or statevector)
simulation, you can directly inspect amplitudes and probabilities.


.. currentmodule:: pyquil
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    ~pyquil.api.WavefunctionSimulator
    ~pyquil.reference_simulator.ReferenceWavefunctionSimulator
    ~pyquil.reference_simulator.ReferenceDensitySimulator
    ~pyquil.numpy_simulator.NumpyWavefunctionSimulator


Reference Utilities
-------------------

.. currentmodule:: pyquil.unitary_tools
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    lifted_pauli
    lifted_gate
    program_unitary
    all_bitstrings


Numpy Utilities
---------------

.. currentmodule:: pyquil.numpy_simulator
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    targeted_einsum
    targeted_tensordot
