import numpy as np

from pyquil.quiltwaveforms import (
    FlatWaveform,
    GaussianWaveform,
    DragGaussianWaveform,
    ErfSquareWaveform,
    HrmGaussianWaveform,
    BoxcarAveragerKernel,
)


def test_waveform_samples():
    # this is a very naive check: can we sample from the built-in template
    # waveforms?
    duration = 1e-6
    waveforms = [
        FlatWaveform(duration=duration, iq=1.0),
        FlatWaveform(duration=duration, iq=1.0 + 2.0j),
        GaussianWaveform(duration=duration, fwhm=2.0, t0=1.0),
        DragGaussianWaveform(
            duration=duration, fwhm=duration / 4, t0=duration / 2, anh=5.0, alpha=3.0
        ),
        HrmGaussianWaveform(
            duration=duration,
            fwhm=duration / 4,
            t0=duration / 2,
            anh=5.0,
            alpha=3.0,
            second_order_hrm_coeff=0.5,
        ),
        ErfSquareWaveform(duration=duration, risetime=duration / 8, pad_left=0.0, pad_right=0.0),
        BoxcarAveragerKernel(duration=duration),
    ]

    rates = [int(1e9), 1e9, 1e9 + 0.5]

    for rate in rates:
        for wf in waveforms:
            assert wf.samples(rate) is not None


def test_waveform_samples_optional_args():
    def flat(**kwargs):
        return FlatWaveform(duration=1e-8, iq=1.0, **kwargs).samples(1e9)

    assert np.array_equal(2.0 * flat(), flat(scale=2.0))
    assert np.array_equal(np.exp(1j) * flat(), flat(phase=1.0))
