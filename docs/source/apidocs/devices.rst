Devices
=======

An appropriate Device is automatically created when using :py:func:`~pyquil.get_qc` and it is
stored on the :py:class:`~pyquil.api.QuantumComputer` object as the ``device`` attribute.

There are properties of real quantum computers that go beyond the quantum abstract machine
(QAM) abstraction. Real devices have performance specs, limited ISAs, and restricted topologies.
:py:class:`~pyquil.device.AbstractDevice` provides an abstract interface for accessing
properties of a real quantum device or for mocking out relevant properties for a more realistic
QVM.


.. currentmodule:: pyquil.device
.. autosummary::
    :toctree: autogen

    AbstractDevice
    Device
    NxDevice

The data structures used are documented here

.. autosummary::
    :toctree: autogen

    ISA
    Specs

Utility functions
~~~~~~~~~~~~~~~~~

.. autofunction:: isa_from_graph
.. autofunction:: specs_from_graph
.. autofunction:: isa_to_graph
.. autofunction:: gates_in_isa
