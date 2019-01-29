QAMs
====

An appropriate QAM is automatically created when using :py:func:`~pyquil.get_qc` and it is
stored on the :py:class:`~pyquil.api.QuantumComputer` object as the ``qam`` attribute.

The Quantum Abstract Machine (QAM) provides an abstract interface for running hybrid
quantum/classical quil programs on either a Quantum Virtual Machine (QVM, a classical simulator)
or a Quantum Processor Unit (QPU, a real quantum device).


.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    ~pyquil.api._qam.QAM
    ~pyquil.api.QPU
    ~pyquil.api.QVM
    ~pyquil.pyqvm.PyQVM
