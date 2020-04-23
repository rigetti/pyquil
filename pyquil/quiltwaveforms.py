import sys
if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass

from copy import copy
import numpy as np
from scipy.special import erf
from numbers import Complex, Real
from inspect import signature

from typing import List, Optional, Dict

from pyquil.quilatom import TemplateWaveform, _complex_str, Expression, substitute


_waveform_classes = {}
""" A mapping from Quilt wavefom names to their corresponding classes. """


def waveform(name: str):
    """ Define a Quilt wavefom with the given name. """
    def wrap(cls):
        cls = dataclass(cls)
        _waveform_classes[name] = cls
        return cls

    return wrap


def _wf_from_dict(name: str, params: dict) -> TemplateWaveform:
    """Construct a TemplateWaveform from a name and a dictionary of properties.
    :param name: The Quilt name of the template.
    :param params: A mapping from parameter names to their corresponding values.

    :returns: A template waveform.
    """
    params = copy(params)
    if name not in _waveform_classes:
        raise ValueError(f"Unknown template waveform {name}.")
    cls = _waveform_classes[name]
    fields = cls.__dataclass_fields__

    for param, value in params.items():
        if param not in fields:
            raise ValueError(f"Unexpected parameter '{param}' in {name}.")

        if isinstance(value, Expression):
            value = substitute(value, {})

        if isinstance(value, Real):
            # normalize to float
            params[param] = float(value)
        elif isinstance(value, Complex):
            # no normalization needed
            pass
        else:
            raise ValueError(f"Unable to resolve parameter '{param}' in template {name} to a constant value.")

    for field, spec in fields.items():
        if field not in params and spec.default is not None:
            raise ValueError(f"Missing parameter '{field}' in {name}.")

    return cls(**params)


def _optional_field_strs(wf: TemplateWaveform) -> List[str]:
    """Get the printed representations of optional template parameters."""
    result = []
    for field, spec in wf.__dataclass_fields__.items():
        if spec.default is None:
            value = getattr(wf, field, None)
            if value is not None:
                result.append(f"{field}: {value}")
    return result


@waveform("flat")
class FlatWaveform(TemplateWaveform):
    """
    A flat (constant) waveform.
    """

    iq: Complex
    """ A raw IQ value. """

    scale: Optional[float] = None
    """ An optional global scaling factor. """

    phase: Optional[float] = None
    """ An optional phase shift factor. """

    detuning: Optional[float] = None
    """ An optional frequency detuning factor. """

    def out(self) -> str:
        output = "flat("
        output += ", ".join(
            [f'duration: {self.duration}',
             f'iq: {_complex_str(self.iq)}'] +
            _optional_field_strs(self)
        )
        output += ")"
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        iqs = np.full(self.num_samples(rate), self.iq, dtype=np.complex128)
        return self._update_envelope(iqs, rate)


@waveform("gaussian")
class GaussianWaveform(TemplateWaveform):
    """ A Gaussian pulse. """

    fwhm: float
    """ The Full-Width-Half-Max of the Gaussian (seconds). """

    t0: float
    """ The center time coordinate of the Gaussian (seconds). """

    scale: Optional[float] = None
    """ An optional global scaling factor. """

    phase: Optional[float] = None
    """ An optional phase shift factor. """

    detuning: Optional[float] = None
    """ An optional frequency detuning factor. """

    def out(self) -> str:
        output = "gaussian("
        output += ", ".join(
            [f'duration: {self.duration}',
             f'fwhm: {self.fwhm}',
             f't0: {self.t0}'] +
            _optional_field_strs(self)
        )
        output += ")"
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        iqs = np.exp(-0.5*(ts-self.t0) ** 2 / sigma ** 2)
        return self._update_envelope(iqs, rate)


@waveform("drag_gaussian")
class DragGaussianWaveform(TemplateWaveform):
    """ A DRAG Gaussian pulse. """

    fwhm: float
    """ The Full-Width-Half-Max of the gaussian (seconds). """

    t0: float
    """ The center time coordinate of the Gaussian (seconds). """

    anh: float
    """ The anharmonicity of the qubit, f01-f12 (Hertz). """

    alpha: float
    """ Dimensionles DRAG parameter. """

    scale: Optional[float] = None
    """ An optional global scaling factor. """

    phase: Optional[float] = None
    """ An optional phase shift factor. """

    detuning: Optional[float] = None
    """ An optional frequency detuning factor. """

    def out(self) -> str:
        output = "drag_gaussian("
        output += ", ".join(
            [f'duration: {self.duration}',
             f'fwhm: {self.fwhm}',
             f't0: {self.t0}',
             f'anh: {self.anh}',
             f'alpha: {self.alpha}'] +
            _optional_field_strs(self)
        )
        output += ")"
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * fwhm / np.sqrt(2.0 * np.log(2.0))
        env = np.exp(-0.5 * (ts - t0) ** 2 / sigma ** 2)
        env_der = (alpha * (1.0 / (2 * np.pi * anh * sigma ** 2))) * (ts - t0) * env
        iqs = env + 1.0j * env_der
        return self._update_envelope(iqs, rate)


@waveform("erf_square")
class ErfSquareWaveform(TemplateWaveform):
    """ A pulse with a flat top and edges that are error functions (erf). """

    risetime: float
    """ The width of each of the rise and fall sections of the pulse (seconds). """
    pad_left: float
    """ Amount of zero-padding to add to the left of the pulse (seconds)."""
    pad_right: float
    """ Amount of zero-padding to add to the right of the pulse (secodns). """

    scale: Optional[float] = None
    """ An optional global scaling factor. """

    phase: Optional[float] = None
    """ An optional phase shift factor. """

    detuning: Optional[float] = None
    """ An optional frequency detuning factor. """

    def out(self) -> str:
        output = "erf_square("
        output += ", ".join(
            [f'duration: {self.duration}',
             f'risetime: {self.risetime}',
             f'pad_left: {self.pad_left}',
             f'pad_right: {self.pad_right}'] +
            _optional_field_strs(self)
        )
        output += ")"
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / sample_rate
        fwhm = 0.5 * self.risetime
        t1 = self.fwhm
        t2 = self.duration-self.fwhm
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        vals = 0.5 * (erf((ts - t1) / sigma) - erf((ts - t2) / sigma))
        zeros_left = np.zeros(np.ceil(self.pad_left * rate), dtype=np.complex128)
        zeros_right = np.zeros(np.ceil(self.pad_left * rate), dtype=np.complex128)
        iqs = np.concatenate((zeros_left, vals, zeros_right))
        return self._update_envelope(iqs, rate)


@waveform("boxcar_kernel")
class BoxcarAveragerKernel(TemplateWaveform):

    scale: Optional[float] = None
    """ An optional global scaling factor. """

    phase: Optional[float] = None
    """ An optional phase shift factor. """

    detuning: Optional[float] = None
    """ An optional frequency detuning factor. """

    def out(self) -> str:
        output = "boxcar_kernel("
        output += ", ".join(
            [f'duration: {self.duration}'] +
            _optional_field_strs(self)
        )
        output += ")"
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        raise NotImplementedError()
