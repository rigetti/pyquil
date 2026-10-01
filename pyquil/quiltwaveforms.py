"""Waveform templates that are commonly useful when working with pulse programs."""

import inspect
from typing import ClassVar, Self, TypeAlias

import numpy as np
from scipy.special import erf

from .quilatom import QuilAtom

from quil.expression import ExpressionDesignator, ParameterDesignator
from quil.instructions import WaveformInvocation


def _template_waveform_property(
    name: str, *, dtype: type[int] | type[float] | None = None, doc: str | None = None
) -> property:
    """Initialize a getters, setters, and deleter for a parameter on a ``TemplateWaveform``.

    Should only be used inside of ``TemplateWaveform`` or one its base classes.

    :param name: The name of the property
    :param dtype: An optional parameter that takes the int or float type, and attempts
            to convert the underlying complex value by casting the real part to `dtype`. If set, this function will
            raise an error if the complex number has a non-zero imaginary part.
    :param doc: Docstring for the property.
    """

    def fget(self: "TemplateWaveform") -> ParameterDesignator | None:
        parameter = self.get_parameter(name)
        if parameter is None or dtype is None:
            return parameter

        if dtype is int or dtype is float:
            if isinstance(parameter, dtype):
                return parameter
            try:
                value = complex(parameter)
            except TypeError as e:
                raise TypeError(
                    f"Requested float for parameter {name}, but a non-numeric value of type {type(parameter)} was "
                    "found instead"
                ) from e
            if value.imag != 0.0:
                raise ValueError(
                    f"Requested float for parameter {name}, but a complex number with a non-zero imaginary part was "
                    "found"
                )
            return dtype(value.real)
        raise TypeError(f"TemplateWaveform is not compatible with dtype {dtype}")

    def fset(self: "TemplateWaveform", value: ParameterDesignator) -> None:
        self.set_parameter(name, value)

    def fdel(self: "TemplateWaveform") -> None:
        self.set_parameter(name, None)

    return property(fget, fset, fdel, doc)


class TemplateWaveform(WaveformInvocation, QuilAtom):
    """Base class for creating waveform templates."""

    NAME: ClassVar[str]

    def __new__(
        cls,
        name: str,
        *,
        duration: float,
        **kwargs: ParameterDesignator | None | ExpressionDesignator | None,
    ) -> Self:
        """Initialize a new TemplateWaveform."""
        rs_parameters = {key: value for key, value in kwargs.items() if value is not None}
        rs_parameters["duration"] = duration
        return super().__new__(cls, name, rs_parameters)

    def out(self) -> str:
        """Return the waveform as a valid Quil string."""
        return str(self)

    def get_parameter(self, name: str) -> ParameterDesignator | None:
        """Get a parameter in the waveform by name."""
        return super().parameters.get(name, None)

    def set_parameter(self, name: str, value: ParameterDesignator | None) -> None:
        """Set a parameter with a value."""
        parameters = super().parameters
        if value is None:
            parameters.pop(name, None)
        else:
            parameters[name] = value
        WaveformInvocation.parameters.__set__(self, parameters)  # type: ignore[attr-defined]

    duration = _template_waveform_property("duration", dtype=float)

    def num_samples(self, rate: float) -> int:
        """Return the number of samples in the reference implementation of the waveform.

        Note: this does not include any hardware-enforced alignment (cf. documentation for `samples`).

        :param rate: The sample rate, in Hz.
        :return: The number of samples.
        """
        duration = self.duration.real
        if self.duration.imag != 0.0:
            raise ValueError("Can't calculate number of samples with a complex duration")
        return int(np.ceil(duration * rate))

    def samples(self, rate: float) -> np.ndarray:
        """Generate samples of the waveform.

        Note: this is close but not always exactly equivalent to the actual IQ
        values produced by the waveform generators on Rigetti hardware. The
        actual ADC process imposes some alignment constraints on the waveform
        duration (in particular, it must be compatible with the clock rate).

        :param rate: The sample rate, in Hz.
        :returns: An array of complex samples.

        """
        raise NotImplementedError()

    @classmethod
    def _from_rs_waveform_invocation(cls, waveform: WaveformInvocation) -> "TemplateWaveform":
        """Build a TemplateWaveform from a ``quil`` waveform invocation.

        The ``quil`` package has no equivalent to ``TemplateWaveform``s, this function checks the name and properties of
        a ``quil`` ``WaveformInvocation`` to see if they potentially match a subclass of ``TemplateWaveform``. If a
        match is found and construction succeeds, then that type is returned. Otherwise, a generic
        ``WaveformInvocation`` is returned.
        """
        from pyquil.quiltwaveforms import (
            BoxcarAveragerKernel,
            DragGaussianWaveform,
            ErfSquareWaveform,
            FlatWaveform,
            GaussianWaveform,
            HrmGaussianWaveform,
        )

        template: type[TemplateWaveform]  # mypy needs a type annotation here to understand this.
        for template in [
            FlatWaveform,
            GaussianWaveform,
            DragGaussianWaveform,
            HrmGaussianWaveform,
            ErfSquareWaveform,
            BoxcarAveragerKernel,
        ]:
            if template.NAME != waveform.name:
                continue
            parameter_names = [
                parameter[0] for parameter in inspect.getmembers(template, lambda a: isinstance(a, property))
            ]
            if set(waveform.parameters.keys()).issubset(parameter_names):
                try:
                    return template(**waveform.parameters)  # type: ignore[arg-type]
                except TypeError:
                    break

        return super().__new__(cls, waveform.name, waveform.parameters)

    def __str__(self) -> str:
        return super().to_quil_or_debug()


def _update_envelope(
    iqs: np.ndarray,
    rate: float,
    scale: float | None,
    phase: float | None,
    detuning: float | None,
) -> np.ndarray:
    """Update a pulse envelope by optional shape parameters.

    The optional parameters are: 'scale', 'phase', 'detuning'.

    :param iqs: The basic pulse envelope.
    :param rate: The sample rate (in Hz).
    :return: The updated pulse envelope.
    """

    def default(obj: float | None, val: float) -> float:
        return obj if obj is not None else val

    scale = default(scale, 1.0)
    phase = default(phase, 0.0)
    detuning = default(detuning, 0.0)

    iqs *= scale * np.exp(1j * phase) * np.exp(1j * 2 * np.pi * detuning * np.arange(len(iqs)) / rate)

    return iqs


Waveform: TypeAlias = TemplateWaveform


def _convert_to_py_waveform(waveform: WaveformInvocation) -> Waveform:
    if not isinstance(waveform, WaveformInvocation):
        raise TypeError(f"{type(waveform)} is not a WaveformInvocation")

    return TemplateWaveform._from_rs_waveform_invocation(waveform)


class FlatWaveform(TemplateWaveform):
    """A flat (constant) waveform."""

    NAME = "flat"

    def __new__(
        cls,
        duration: float,
        iq: complex,
        scale: float | None = None,
        phase: float | None = None,
        detuning: float | None = None,
    ) -> Self:
        """Initialize a new FlatWaveform."""
        return super().__new__(cls, cls.NAME, duration=duration, iq=iq, scale=scale, phase=phase, detuning=detuning)

    iq = _template_waveform_property("iq", doc="A raw IQ value.")
    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)
    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)
    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def __getnewargs__(self) -> tuple[float, complex, float | None, float | None, float | None]:
        return self.duration, self.iq, self.scale, self.phase, self.detuning

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
        scale: float | None = None,
        phase: float | None = None,
        detuning: float | None = None,
    ) -> Self:
        """Initialize a new GaussianWaveform."""
        return super().__new__(
            cls, cls.NAME, duration=duration, fwhm=fwhm, t0=t0, scale=scale, phase=phase, detuning=detuning
        )

    def __getnewargs__(self) -> tuple[float, float, float, float | None, float | None, float | None]:
        return self.duration, self.fwhm, self.t0, self.scale, self.phase, self.detuning

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
        scale: float | None = None,
        phase: float | None = None,
        detuning: float | None = None,
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

    def __getnewargs__(self) -> tuple[float, float, float, float, float, float | None, float | None, float | None]:
        return (
            self.duration,
            self.fwhm,
            self.t0,
            self.anh,
            self.alpha,
            self.scale,
            self.phase,
            self.detuning,
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
        scale: float | None = None,
        phase: float | None = None,
        detuning: float | None = None,
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

    def __getnewargs__(
        self,
    ) -> tuple[float, float, float, float, float, float, float | None, float | None, float | None]:
        return (
            self.duration,
            self.fwhm,
            self.t0,
            self.anh,
            self.alpha,
            self.second_order_hrm_coeff,
            self.scale,
            self.phase,
            self.detuning,
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
        scale: float | None = None,
        phase: float | None = None,
        detuning: float | None = None,
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

    def __getnewargs__(self) -> tuple[float, float, float, float, float | None, float | None, float | None]:
        return (
            self.duration,
            self.risetime,
            self.pad_left,
            self.pad_right,
            self.scale,
            self.phase,
            self.detuning,
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
        scale: float | None = None,
        phase: float | None = None,
        detuning: float | None = None,
    ) -> Self:
        """Initialize a new BoxcarAveragerKernel."""
        return super().__new__(cls, cls.NAME, duration=duration, scale=scale, phase=phase, detuning=detuning)

    def __getnewargs__(self) -> tuple[float, float | None, float | None, float | None]:
        return self.duration, self.scale, self.phase, self.detuning

    scale = _template_waveform_property("scale", doc="An optional global scaling factor.", dtype=float)

    phase = _template_waveform_property("phase", doc="An optional phase shift factor.", dtype=float)

    detuning = _template_waveform_property("detuning", doc="An optional frequency detuning factor.", dtype=float)

    def samples(self, rate: float) -> np.ndarray:
        """Get the samples of the waveform at a given sample rate."""
        n = self.num_samples(rate)
        iqs = np.full(n, 1.0 / n, dtype=np.complex128)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)
