Compilers
=========

An appropriate compiler is automatically created when using :py:func:`~pyquil.get_qc` and it is
stored on the :py:class:`~pyquil.api.QuantumComputer` object as the ``compiler`` attribute.

The exact process for compilation depends on whether you're targeting a QPU or a QVM, and
you can conceive of other compilation strategies than those included with pyQuil by default.
Therefore, we define an abstract interface that all compilers must follow. See
:py:class:`~pyquil.api._qac.AbstractCompiler` for more, or use one of the listed compilers below.


.. currentmodule:: pyquil.api
.. autosummary::
    :toctree: autogen

    _qac.AbstractCompiler
    QVMCompiler
    QPUCompiler
