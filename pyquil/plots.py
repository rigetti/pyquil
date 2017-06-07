import matplotlib.pyplot as plt
import numpy as np

from job_results import RabiResult, RamseyResult, T1Result


def analog_plot(job_result):
    xs, amps = job_result.decode()
    if isinstance(job_result, RabiResult):
        plt.title("Qubit Rabi")
        plt.ylabel("Response Amplitude (mV)")
        plt.xlabel("Pulse Amplitude (mV)")
    elif isinstance(job_result, RamseyResult):
        plt.title("Qubit Ramsey")
        plt.ylabel("Response Amplitude (mV)")
        plt.xlabel("Pulse Delay (us)")
        xs = np.asarray(xs) * 1.e6
    elif isinstance(job_result, T1Result):
        plt.title("Qubit T1")
        plt.ylabel("Response Amplitude (mV)")
        plt.xlabel("Readout Delay (us)")
        xs = np.asarray(xs) * 1.e6
    plt.plot(xs, amps)
    plt.show()
