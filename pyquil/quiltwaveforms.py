import sys
if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass

import numpy as np
from scipy.special import erf
from numbers import Complex, Real
from inspect import signature

from typing import List, Optional

from pyquil.quilatom import TemplateWaveform, _complex_str, Expression, substitute

# NOTE: With dataclasses, one cannot comfortably rely on both default arguments
# and inheritence. We explicitly manage these fields here.
OPTIONAL_FIELDS = ['detuning', 'scale', 'phase']

def _extract_optional_fields(kwargs):
    return {field: kwargs.get(field, None) for field in OPTIONAL_FIELDS}

def _optional_field_strs(wf: TemplateWaveform) -> List[str]:
    result = []
    if getattr(wf, 'detuning', None) is not None:
        result.append(f"detuning: {wf.detuning}")
    if getattr(wf, 'scale', None) is not None:
        result.append(f"scale: {wf.scale}")
    if getattr(wf, 'phase', None) is not None:
        result.append(f"phase: {wf.phase}")
    return result


@dataclass
class FlatWaveform(TemplateWaveform):
    """
    A flat (constant) waveform.
    """

    iq: Complex
    """ A raw IQ value. """

    def __init__(self, **kwargs):
        super().__init__(duration=kwargs['duration'],
                         **_extract_optional_fields(kwargs))
        self.iq = kwargs['iq']

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


@dataclass
class GaussianWaveform(TemplateWaveform):
    """ A Gaussian pulse. """

    fwhm: float
    """ The Full-Width-Half-Max of the Gaussian (seconds). """

    t0: float
    """ The center time coordinate of the Gaussian (seconds). """

    def __init__(self, **kwargs):
        super().__init__(duration=kwargs['duration'],
                         **_extract_optional_fields(kwargs))
        self.fwhm = kwargs['fwhm']
        self.t0 = kwargs['t0']

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

@dataclass
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

    def __init__(self, **kwargs):
        super().__init__(duration=kwargs['duration'],
                         **_extract_optional_fields(kwargs))
        self.fwhm = kwargs['fwhm']
        self.t0 = kwargs['t0']
        self.anh = kwargs['anh']
        self.alpha = kwargs['alpha']

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


@dataclass
class ErfSquareWaveform(TemplateWaveform):
    """ A pulse with a flat top and edges that are error functions (erf). """

    risetime: float
    """ The width of each of the rise and fall sections of the pulse (seconds). """
    pad_left: float
    """ Amount of zero-padding to add to the left of the pulse (seconds)."""
    pad_right: float
    """ Amount of zero-padding to add to the right of the pulse (secodns). """

    def __init__(self, **kwargs):
        super().__init__(duration=kwargs['duration'],
                         **_extract_optional_fields(kwargs))
        self.risetime = kwargs['risetime']
        self.pad_left = kwargs['pad_left']
        self.pad_right = kwargs['pad_right']

    def out(self) -> str:
        output = "erf_square("
        output += ", ".join(
            [f'risetime: {self.risetime}',
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


@dataclass
class BoxcarAverageKernel(TemplateWaveform):
    pass

WAVEFORM_CLASSES = {
    'flat': FlatWaveform,
    'gaussian': GaussianWaveform,
    'drag_gaussian': DragGaussianWaveform,
    'erf_square': ErfSquareWaveform,
}


def _from_dict(name: str, params: dict) -> TemplateWaveform:
    """Construct a TemplateWaveform from a name and a dictionary of properties.
    :param name: The Quilt name of the template.
    :param params: A mapping from parameter names to their corresponding values.

    :returns: A template waveform.
    """
    if name not in WAVEFORM_CLASSES:
        raise ValueError(f"Unknown template waveform {name}.")
    cls = WAVEFORM_CLASSES[name]
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

    for field in fields:
        if field not in params and field not in OPTIONAL_FIELDS:
            raise ValueError(f"Missing parameter '{field}' in {name}.")


    return cls(**params)
