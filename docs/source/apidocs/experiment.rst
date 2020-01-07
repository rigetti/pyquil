Experiment
==========

The ``experiment`` module offers a schema and utilities for succinctly expressing commonly
used applications and algorithms in near-term quantum programming. A ``TomographyExperiment``
is intended to be consumed by the ``QuantumComputer.experiment`` method.

**NOTE**: When working with the `experiment` method, the following declared memory labels are
reserved:

 - "preparation_alpha", "preparation_beta", and "preparation_gamma"
 - "measurement_alpha", "measurement_beta", and "measurement_gamma"
 - "symmetrization"
 - "ro"

.. currentmodule:: pyquil.experiment

Schema
------

.. autoclass:: pyquil.experiment.TomographyExperiment

   .. rubric:: Methods

   .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~TomographyExperiment.get_meas_qubits
        ~TomographyExperiment.get_meas_registers
        ~TomographyExperiment.generate_experiment_program
        ~TomographyExperiment.build_setting_memory_map
        ~TomographyExperiment.build_symmetrization_memory_maps
        ~TomographyExperiment.generate_calibration_experiment

.. autoclass:: pyquil.experiment.ExperimentSetting

.. autoclass:: pyquil.experiment.ExperimentResult

.. autoclass:: pyquil.experiment.SymmetrizationLevel

.. autoclass:: pyquil.experiment.CalibrationMethod

Utilities
---------

.. autofunction:: pyquil.experiment.bitstrings_to_expectations
.. autofunction:: pyquil.experiment.correct_experiment_result
.. autofunction:: pyquil.experiment.merge_memory_map_lists
.. autofunction:: pyquil.experiment.ratio_variance
.. autofunction:: pyquil.experiment.read_json
.. autofunction:: pyquil.experiment.to_json
