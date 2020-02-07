Noise Models
============

.. currentmodule:: pyquil.noise

Functions
---------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    append_kraus_to_gate
    pauli_kraus_map
    damping_kraus_map
    dephasing_kraus_map
    tensor_kraus_maps
    combine_kraus_maps
    damping_after_dephasing
    get_noisy_gate
    _decoherence_noise_model
    decoherence_noise_with_asymmetric_ro
    estimate_bitstring_probs
    corrupt_bitstring_probs
    bitstring_probs_to_z_moments


Classes
-------

.. autoclass:: pyquil.noise.NoiseModel

   .. rubric:: Methods

   .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~NoiseModel.gates_by_name
        ~NoiseModel.to_dict
        ~NoiseModel.from_dict


.. autoclass:: pyquil.noise.KrausModel

   .. rubric:: Methods

   .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~KrausModel.unpack_kraus_matrix
        ~KrausModel.unpack_kraus_matrix
        ~KrausModel.to_dict
        ~KrausModel.from_dict





