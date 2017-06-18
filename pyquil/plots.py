import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import update_param_vals

from job_results import RabiResult, RamseyResult, T1Result


def analog_plot(job_result):
    """
    Make a plot, using matplotlib, of an experimental result.
    :param JobResult job_result:
    :return: None
    """
    xs, amps = job_result.decode()
    if isinstance(job_result, RabiResult):
        plt.title("Qubit Rabi")
        plt.ylabel("Response")
        plt.xlabel("Pulse Amplitude (mV)")
    elif isinstance(job_result, RamseyResult):
        plt.title("Qubit Ramsey")
        plt.ylabel("Response")
        plt.xlabel("Pulse Delay (us)")
        xs = np.asarray(xs) * 1.e6
    elif isinstance(job_result, T1Result):
        plt.title("Qubit T1")
        plt.ylabel("Response")
        plt.xlabel("Readout Delay (us)")
        xs = np.asarray(xs) * 1.e6
    plt.plot(xs, amps, 'bo')
    plt.show()


##################################################################
# T1 decay
##################################################################
def fn_T1_decay(x, baseline, amplitude, T1):
    return baseline + amplitude * np.exp(-x / T1)

MIN_DATA_POINTS = 5

class T1DecayModel(Model):
    __doc__ = """Class for T1 decay model"""

    def __init__(self, *args, **kwargs):
        super(T1DecayModel, self).__init__(fn_T1_decay, *args, **kwargs)

    __model__ = Model(fn_T1_decay)

    def guess(self, x, y, **kwargs):
        """Takes x and y data and generates initial guesses for
        fit parameters.
        """
        if y[0] > y[-1]:
            b_guess = y.min()
            a_guess = 1.0 * (y.max() - b_guess)
        else:
            b_guess = y.max()
            a_guess = 1.0 * (y.min() - b_guess)
        y2 = y - b_guess
        x2 = x - x[0]
        sy2 = sum(y2)
        if sy2 == 0.:
            # prevent division by 0
            t_guess = 100 * 1.e-6
        else:
            t_guess = 0.5 * np.dot(x2, y2) / sy2  # 1st moment of x: sum(x*f(x)) / sum(f(x))

        pars = self.make_params(baseline=b_guess, amplitude=a_guess, T1=t_guess)
        pars['amplitude'].value = kwargs.pop('amplitude_guess', a_guess)
        pars['baseline'].value = kwargs.pop('baseline_guess', b_guess)
        pars['T1'].value = kwargs.pop('T1_guess', t_guess)
        return update_param_vals(pars, self.prefix, **kwargs)

    def do_fit(self, x, y, errs=None, **kwargs):
        """Performs a fit.
        """
        par = self.guess(x, y, **kwargs)
        if errs is not None:
            fit = self.fit(y, x=x, weights=1.0 / errs, params=par)
        else:
            fit = self.fit(y, x=x, params=par)
        return fit

    def report_fit(self, x, y, **kwargs):
        """Reports the results of a fit. May change depending on what we
        want this to return.
        """

        if not len(x) == len(y):
            raise ValueError("Lengths of x and y arrays must be equal.")
        if not len(x) > MIN_DATA_POINTS:
            raise ValueError("You must provide more than {} data points.".format(MIN_DATA_POINTS))

        fit = self.do_fit(x, y, **kwargs)
        t_fit = fit.params['T1']
        return fit, (t_fit.value, t_fit.stderr)

##################################################################
# T2 Ramsey
##################################################################
def fn_T2_Ramsey(x, baseline, amplitude, T2, detuning, x0=0.0):
    """Ramsey lineshape.
    """
    return baseline + amplitude * np.exp(-x / T2) * np.cos(2 * np.pi * detuning * (x - x0))


class T2RamseyModel(Model):
    __doc__ = """Class for T2 Ramsey model"""

    def __init__(self, *args, **kwargs):
        super(T2RamseyModel, self).__init__(fn_T2_Ramsey, *args, **kwargs)

    __model__ = Model(fn_T2_Ramsey)

    def guess(self, x, y, **kwargs):
        """Takes x and y data and generates initial guesses for
        fit parameters.
        """
        ym = y.min()
        yM = y.max()
        a_guess = 0.5 * (yM - ym) * 1.2
        b_guess = 0.5 * (yM + ym)
        t_guess = 0.25 * x[-1]
        d_guess = get_freq_from_fft(x, y, b_guess)
        x_guess = 0.0

        pars = self.make_params(baseline=b_guess, amplitude=a_guess,
                                T2=t_guess, detuning=d_guess, x0=x_guess)
        pars['amplitude'].value = kwargs.pop('amplitude_guess', a_guess)
        pars['baseline'].value = kwargs.pop('baseline_guess', b_guess)
        pars['T2'].value = kwargs.pop('T2_guess', t_guess)
        pars['detuning'].value = kwargs.pop('detuning_guess', d_guess)
        pars['x0'].value = kwargs.pop('x0_guess', x_guess)
        return update_param_vals(pars, self.prefix, **kwargs)

    def do_fit(self, x, y, errs=None, **kwargs):
        """Performs a fit.
        """
        par = self.guess(x, y, **kwargs)
        if errs is not None:
            fit = self.fit(y, x=x, weights=1.0 / errs, params=par, fit_kws={"nan_policy": "omit"})
        else:
            fit = self.fit(y, x=x, params=par, fit_kws={"nan_policy": "omit"})
        return fit

    def report_fit(self, x, y, **kwargs):
        """Reports the results of a fit. May change depending on what we
        want this to return.
        """

        if not len(x) == len(y):
            raise ValueError("Lengths of x and y arrays must be equal.")
        if not len(x) > MIN_DATA_POINTS:
            raise ValueError("You must provide more than {} data points.".format(MIN_DATA_POINTS))

        fit = self.do_fit(x, y, **kwargs)
        t_fit = fit.params['T2']
        d_fit = fit.params['detuning']
        return fit, (t_fit.value, t_fit.stderr), (d_fit.value, d_fit.stderr)


def get_freq_from_fft(x, y, offset=0.0):
    """Returns the frequency at which an fft is peaked in amplitude.
    Notes:
    - A constant offset is subtracted
    - The two largest frequency components are taken in case f=0 is still the
    largest components
    - A sine wave will have peaks at +/- the true frequency. This function
    returns the absolute value.
    """
    assert len(x) > 1
    f = np.fft.fftfreq(len(y), x[1] - x[0])
    A = np.abs(np.fft.fft(y - offset))
    f_max_2 = f[np.argpartition(A, -2)][-2:]  # f's for largest 2 elements of A
    if f_max_2[0] != 0.0:
        f_max = f_max_2[0]
    else:
        f_max = f_max_2[1]
    return np.abs(f_max)