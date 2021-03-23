Quantum Processors
==================

An appropriate :py:class:`~pyquil.quantum_processors.AbstractQuantumProcessor` is automatically created when using
:py:func:`~pyquil.get_qc` and it is stored on the :py:class:`~pyquil.api.QuantumComputer` object as the
``quantum_processor`` attribute.

There are properties of real quantum processors that go beyond the quantum abstract machine
(QAM) abstraction. Real processors have performance specs, limited ISAs, and restricted topologies.
:py:class:`~pyquil.quantum_processors.AbstractQuantumProcessor` provides an abstract interface for accessing
properties of a real quantum processor or for mocking out relevant properties for a more realistic
QVM.


.. currentmodule:: pyquil.quantum_processor
.. autosummary::
    :toctree: autogen

    AbstractQuantumProcessor
    CompilerQuantumProcessor
    NxQuantumProcessor
    QCSQuantumProcessor

