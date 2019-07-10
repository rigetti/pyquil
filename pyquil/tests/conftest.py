
import shutil

import numpy as np
import pytest
from requests import RequestException

from pyquil.api import (QVMConnection, QVMCompiler, ForestConnection,
                        get_benchmarker, local_qvm)
from pyquil.api._config import PyquilConfig
from pyquil.api._errors import UnknownApiError
from pyquil.api._compiler import QuilcNotRunning, QuilcVersionMismatch
from pyquil.api._qvm import QVMNotRunning, QVMVersionMismatch
from pyquil.device import Device
from pyquil.gates import I
from pyquil.paulis import sX
from pyquil.quil import Program


@pytest.fixture
def isa_dict():
    return {
        "1Q": {
            "0": {
                'type': 'Xhalves'
            },
            "1": {},
            "2": {},
            "3": {
                "dead": True
            }
        },
        "2Q": {
            "0-1": {},
            "1-2": {
                "type": "ISWAP"
            },
            "0-2": {
                "type": "CPHASE"
            },
            "0-3": {
                "dead": True
            }
        }
    }


@pytest.fixture
def specs_dict():
    return {
        '1Q': {
            "0": {
                "f1QRB": 0.99,
                "fRO": 0.93,
                "T1": 20e-6,
                "T2": 15e-6
            },
            "1": {
                "f1QRB": 0.989,
                "fRO": 0.92,
                "T1": 19e-6,
                "T2": 12e-6
            },
            "2": {
                "f1QRB": 0.983,
                "fRO": 0.95,
                "T1": 21e-6,
                "T2": 16e-6
            },
            "3": {
                "f1QRB": 0.988,
                "fRO": 0.94,
                "T1": 18e-6,
                "T2": 11e-6
            }
        },
        '2Q': {
            "0-1": {
                "fBellState": 0.90,
                "fCZ": 0.89,
                "fCZ_std_err": 0.01,
                "fCPHASE": 0.88
            },
            "1-2": {
                "fBellState": 0.91,
                "fCZ": 0.90,
                "fCZ_std_err": 0.12,
                "fCPHASE": 0.89
            },
            "2-0": {
                "fBellState": 0.92,
                "fCZ": 0.91,
                "fCZ_std_err": 0.20,
                "fCPHASE": 0.90
            },
            "0-3": {
                "fBellState": 0.89,
                "fCZ": 0.88,
                "fCZ_std_err": 0.03,
                "fCPHASE": 0.87
            }
        }
    }


@pytest.fixture
def noise_model_dict():
    return {'gates': [{'gate': 'I',
                       'params': (5.0,),
                       'targets': (0, 1),
                       'kraus_ops': [[[[1.]], [[1.0]]]],
                       'fidelity': 1.0},
                      {'gate': 'RX',
                       'params': (np.pi / 2.,),
                       'targets': (0,),
                       'kraus_ops': [[[[1.]], [[1.0]]]],
                       'fidelity': 1.0}],
            'assignment_probs': {'1': [[1.0, 0.0], [0.0, 1.0]],
                                 '0': [[1.0, 0.0], [0.0, 1.0]]},
            }


@pytest.fixture
def device_raw(isa_dict, noise_model_dict, specs_dict):
    return {'isa': isa_dict,
            'noise_model': noise_model_dict,
            'specs': specs_dict,
            'is_online': True,
            'is_retuning': False}


@pytest.fixture
def test_device(device_raw):
    return Device('test_device', device_raw)


@pytest.fixture(scope='session')
def qvm():
    try:
        qvm = QVMConnection(random_seed=52)
        qvm.run(Program(I(0)), [])
        return qvm
    except (RequestException, QVMNotRunning, UnknownApiError) as e:
        return pytest.skip("This test requires QVM connection: {}".format(e))
    except QVMVersionMismatch as e:
        return pytest.skip("This test requires a different version of the QVM: {}".format(e))


@pytest.fixture()
def compiler(test_device):
    try:
        config = PyquilConfig()
        compiler = QVMCompiler(endpoint=config.quilc_url, device=test_device, timeout=1)
        compiler.quil_to_native_quil(Program(I(0)))
        return compiler
    except (RequestException, QuilcNotRunning, UnknownApiError, TimeoutError) as e:
        return pytest.skip("This test requires compiler connection: {}".format(e))
    except QuilcVersionMismatch as e:
        return pytest.skip("This test requires a different version of quilc: {}".format(e))


@pytest.fixture(scope='session')
def forest():
    try:
        connection = ForestConnection()
        connection._wavefunction(Program(I(0)), 52)
        return connection
    except (RequestException, UnknownApiError) as e:
        return pytest.skip("This test requires a Forest connection: {}".format(e))


@pytest.fixture(scope='session')
def benchmarker():
    try:
        bm = get_benchmarker(timeout=2)
        bm.apply_clifford_to_pauli(Program(I(0)), sX(0))
        return bm
    except (RequestException, TimeoutError) as e:
        return pytest.skip("This test requires a running local benchmarker endpoint (ie quilc): {}"
                           .format(e))


@pytest.fixture(scope='session')
def local_qvm_quilc():
    """Execute test with local qvm and quilc running"""
    if shutil.which('qvm') is None or shutil.which('quilc') is None:
        return pytest.skip("This test requires 'qvm' and 'quilc' "
                           "executables to be installed locally.")

    with local_qvm() as context:
        yield context


def _str_to_bool(s):
    """Convert either of the strings 'True' or 'False' to their Boolean equivalent"""
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("Please specify either True or False")


def pytest_addoption(parser):
    parser.addoption("--use-seed", action="store", type=_str_to_bool, default=False,
                     help="run some tests faster using a fixed random seed")


@pytest.fixture()
def use_seed(pytestconfig):
    return pytestconfig.getoption("use_seed")
