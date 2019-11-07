Experiment
==========

The ``experiment`` module offers a schema and utilities for succinctly expressing commonly
used applications and algorithms in near-term quantum programming. A ``TomographyExperiment``
is intended to be consumed by the ``QuantumComputer.experiment`` method.

.. currentmodule:: pyquil.experiment

Schema
------

.. autoclass:: pyquil.experiment.TomographyExperiment

   .. rubric:: Methods

   .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~TomographyExperiment.generate_experiment_program

.. autoclass:: SymmetrizationLevel

.. autoclass:: pyquil.experiment.ExperimentSetting

   .. rubric:: Methods

   .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~ExperimentSetting.build_setting_memory_map

.. autoclass:: pyquil.experiment.ExperimentResult

Utilities
---------

.. autofunction:: pyquil.experiment.bitstrings_to_expectations
.. autofunction:: pyquil.experiment.build_symmetrization_memory_maps
.. autofunction:: pyquil.experiment.merge_memory_map_lists
.. autofunction:: pyquil.experiment.read_json
.. autofunction:: pyquil.experiment.to_json
