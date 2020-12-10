Quil-T Waveforms
================

.. currentmodule:: pyquil.quiltwaveforms

All Quil-T waveforms have corresponding PyQuil syntax objects. For example, a waveform written in Quil-T syntax as ``flat(iq: 1.0, duration: 1e-5)`` corresponds to a :py:class:`~pyquil.quiltwaveforms.FlatWaveform` object.

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    
    FlatWaveform
    GaussianWaveform
    DragGaussianWaveform
    HrmGaussianWaveform
    ErfSquareWaveform
    BoxcarAveragerKernel



Custom Waveforms
----------------

Custom waveform definitions, introduced in Quil-T with ``DEFWAVEFORM``, have a corresponding :py:class:`~pyquil.quilbase.DefWaveform` object. These are referenced in PyQuil with :py:class:`~pyquil.quilatom.WaveformReference`.
