QAMs
====

An appropriate QAM is automatically created when using :py:func:`~pyquil.get_qc` and it is
stored on the :py:class:`~pyquil.api.QuantumComputer` object as the ``qam`` attribute.

The Quantum Abstract Machine (QAM) provides an abstract interface for running hybrid
quantum/classical quil programs on either a Quantum Virtual Machine (QVM, a classical simulator)
or a Quantum Processor Unit (QPU, a real quantum device).


.. currentmodule:: pyquil.api
.. autosummary::
    :toctree: autogen

    _qam.QAM
    QPU
    QVM
