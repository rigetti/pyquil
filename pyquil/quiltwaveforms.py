"""Waveform templates that are commonly useful when working with pulse programs."""

from typing import Optional

import numpy as np
from scipy.special import erf
from typing_extensions import Self

from pyquil.quilatom import (
    TemplateWaveform,
    _template_waveform_property,
    _update_envelope,
)


class FlatWaveform(TemplateWaveform):
    """A flat (constant) waveform."""

    NAME = "flat"

    def __new__(
        cls,
        duration: float,
        iq: complex,
        scale: Optional[float] = None,
        phase: Optional[float] = None,
        detuning: Optional[float] = None,
    ) -> Self:
        """Initialize a new FlatWaveform."""
        return super().__new__(cls, cls.NAME, duration=duration, iq=iq, scale=scale, phase=phase, detuning=detuning)

    iq = _template_waveform_property("iq", doc="A raw IQ value.")
    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)
    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)
    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        iqs = np.full(self.num_samples(rate), self.iq, dtype=np.complex128)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)


class GaussianWaveform(TemplateWaveform):
    """A Gaussian pulse."""

    NAME = "gaussian"

    def __new__(
        cls,
        duration: float,
        fwhm: float,
        t0: float,
        scale: Optional[float] = None,
        phase: Optional[float] = None,
        detuning: Optional[float] = None,
    ) -> Self:
        """Initialize a new GaussianWaveform."""
        return super().__new__(
            cls, cls.NAME, duration=duration, fwhm=fwhm, t0=t0, scale=scale, phase=phase, detuning=detuning
        )

    fwhm = _template_waveform_property("fwhm", doc="The Full-Width-Half-Max of the Gaussian (seconds).", dtype=float)

    t0 = _template_waveform_property("t0", doc="The center time coordinate of the Gaussian (seconds).", dtype=float)

    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)

    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)

    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        iqs = np.exp(-0.5 * (ts - self.t0) ** 2 / sigma**2)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)


class DragGaussianWaveform(TemplateWaveform):
    """A DRAG Gaussian pulse."""

    NAME = "drag_gaussian"

    def __new__(
        cls,
        duration: float,
        fwhm: float,
        t0: float,
        anh: float,
        alpha: float,
        scale: Optional[float] = None,
        phase: Optional[float] = None,
        detuning: Optional[float] = None,
    ) -> Self:
        """Initialize a new DragGaussianWaveform."""
        return super().__new__(
            cls,
            cls.NAME,
            duration=duration,
            fwhm=fwhm,
            t0=t0,
            anh=anh,
            alpha=alpha,
            scale=scale,
            phase=phase,
            detuning=detuning,
        )

    fwhm = _template_waveform_property("fwhm", doc="The Full-Width-Half-Max of the gaussian (seconds).", dtype=float)

    t0 = _template_waveform_property("t0", doc="The center time coordinate of the Gaussian (seconds).", dtype=float)

    anh = _template_waveform_property("anh", doc="The anharmonicity of the qubit, f01-f12 (Hertz).", dtype=float)

    alpha = _template_waveform_property("alpha", doc="Dimensionles DRAG parameter.", dtype=float)

    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)

    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)

    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        env = np.exp(-0.5 * (ts - self.t0) ** 2 / sigma**2)
        env_der = (self.alpha * (1.0 / (2 * np.pi * self.anh * sigma**2))) * (ts - self.t0) * env
        iqs = env + 1.0j * env_der
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)


class HrmGaussianWaveform(TemplateWaveform):
    """A Hermite Gaussian waveform.

    REFERENCE: Effects of arbitrary laser or NMR pulse shapes on population
        inversion and coherence Warren S. Warren. 81, (1984); doi:
        10.1063/1.447644
    """

    NAME = "hrm_gaussian"

    def __new__(
        cls,
        duration: float,
        fwhm: float,
        t0: float,
        anh: float,
        alpha: float,
        second_order_hrm_coeff: float,
        scale: Optional[float] = None,
        phase: Optional[float] = None,
        detuning: Optional[float] = None,
    ) -> Self:
        """Initialize a new HrmGaussianWaveform."""
        return super().__new__(
            cls,
            cls.NAME,
            duration=duration,
            fwhm=fwhm,
            t0=t0,
            anh=anh,
            second_order_hrm_coeff=second_order_hrm_coeff,
            alpha=alpha,
            scale=scale,
            phase=phase,
            detuning=detuning,
        )

    fwhm = _template_waveform_property("fwhm", doc="The Full-Width-Half-Max of the Gaussian (seconds).", dtype=float)

    t0 = _template_waveform_property("t0", doc="The center time coordinate of the Gaussian (seconds).", dtype=float)

    anh = _template_waveform_property("anh", doc="The anharmonicity of the qubit, f01-f12 (Hertz).", dtype=float)

    alpha = _template_waveform_property("alpha", doc="Dimensionles DRAG parameter.", dtype=float)

    second_order_hrm_coeff = _template_waveform_property(
        "second_order_hrm_coeff", doc="Second order coefficient (see Warren 1984).", dtype=float
    )

    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)

    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)

    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        exponent_of_t = 0.5 * (ts - self.t0) ** 2 / sigma**2
        gauss = np.exp(-exponent_of_t)
        env = (1 - self.second_order_hrm_coeff * exponent_of_t) * gauss
        deriv_prefactor = -self.alpha / (2 * np.pi * self.anh)
        env_der = (
            deriv_prefactor
            * (ts - self.t0)
            / (sigma**2)
            * gauss
            * (self.second_order_hrm_coeff * (exponent_of_t - 1) - 1)
        )
        iqs = env + 1.0j * env_der
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)


class ErfSquareWaveform(TemplateWaveform):
    """A pulse with a flat top and edges that are error functions (erf)."""

    NAME = "erf_square"

    def __new__(
        cls,
        duration: float,
        risetime: float,
        pad_left: float,
        pad_right: float,
        scale: Optional[float] = None,
        phase: Optional[float] = None,
        detuning: Optional[float] = None,
    ) -> Self:
        """Initialize a new ErfSquareWaveform."""
        return super().__new__(
            cls,
            cls.NAME,
            duration=duration,
            risetime=risetime,
            pad_left=pad_left,
            pad_right=pad_right,
            scale=scale,
            phase=phase,
            detuning=detuning,
        )

    risetime = _template_waveform_property(
        "risetime", doc="The width of each of the rise and fall sections of the pulse (seconds).", dtype=float
    )

    pad_left = _template_waveform_property(
        "pad_left", doc="Amount of zero-padding to add to the left of the pulse (seconds)", dtype=float
    )

    pad_right = _template_waveform_property(
        "pad_right", doc="Amount of zero-padding to add to the right of the pulse (seconds).", dtype=float
    )

    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)

    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)

    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        fwhm = 0.5 * self.risetime
        t1 = fwhm
        t2 = self.duration - fwhm
        sigma = 0.5 * fwhm / np.sqrt(2.0 * np.log(2.0))
        vals = 0.5 * (erf((ts - t1) / sigma) - erf((ts - t2) / sigma))
        zeros_left = np.zeros(int(np.ceil(self.pad_left * rate)), dtype=np.complex128)
        zeros_right = np.zeros(int(np.ceil(self.pad_right * rate)), dtype=np.complex128)
        iqs = np.concatenate((zeros_left, vals, zeros_right))
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)


class BoxcarAveragerKernel(TemplateWaveform):
    """A boxcar averaging kernel."""

    NAME = "boxcar_kernel"

    def __new__(
        cls,
        duration: float,
        scale: Optional[float] = None,
        phase: Optional[float] = None,
        detuning: Optional[float] = None,
    ) -> Self:
        """Initialize a new BoxcarAveragerKernel."""
        return super().__new__(cls, cls.NAME, duration=duration, scale=scale, phase=phase, detuning=detuning)

    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)

    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)

    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        n = self.num_samples(rate)
        iqs = np.full(n, 1.0 / n, dtype=np.complex128)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)
