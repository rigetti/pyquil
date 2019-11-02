import functools
from operator import mul

import numpy as np
import pytest

from pyquil import Program
from pyquil.experiment._main import _remove_reset_from_program
from pyquil.experiment import (ExperimentSetting, SIC0, SIC1, SIC2, SIC3, TensorProductState,
                               TomographyExperiment, plusX, minusX, plusY, minusY, plusZ, minusZ,
                               read_json, to_json, zeros_state, ExperimentResult)
from pyquil.gates import RESET, X, Y
from pyquil.paulis import sI, sX, sY, sZ


EXPERIMENT_REPR = """
shots: 1
active reset: disabled
symmetrization: -1 (exhaustive)
program:
   X 0
   Y 1
"""


def _generate_random_states(n_qubits, n_terms):
    oneq_states = [SIC0, SIC1, SIC2, SIC3, plusX, minusX, plusY, minusY, plusZ, minusZ]
    all_s_inds = np.random.randint(len(oneq_states), size=(n_terms, n_qubits))
    states = []
    for s_inds in all_s_inds:
        state = functools.reduce(mul, (oneq_states[pi](i) for i, pi in enumerate(s_inds)),
                                 TensorProductState([]))
        states += [state]
    return states


def _generate_random_paulis(n_qubits, n_terms):
    paulis = [sI, sX, sY, sZ]
    all_op_inds = np.random.randint(len(paulis), size=(n_terms, n_qubits))
    operators = []
    for op_inds in all_op_inds:
        op = functools.reduce(mul, (paulis[pi](i) for i, pi in enumerate(op_inds)), sI(0))
        op *= np.random.uniform(-1, 1)
        operators += [op]
    return operators


def test_experiment_setting():
    in_states = _generate_random_states(n_qubits=4, n_terms=7)
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for ist, oop in zip(in_states, out_ops):
        expt = ExperimentSetting(ist, oop)
        assert str(expt) == expt.serializable()
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_state == ist
        assert expt2.out_operator == oop


@pytest.mark.filterwarnings("ignore:ExperimentSetting")
def test_setting_no_in_back_compat():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(TensorProductState(), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == sI()
        assert expt2.out_operator == oop


@pytest.mark.filterwarnings("ignore:ExperimentSetting")
def test_setting_no_in():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(zeros_state(oop.get_qubits()), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == functools.reduce(mul, [sZ(q) for q in oop.get_qubits()], sI())
        assert expt2.out_operator == oop


def test_tomo_experiment():
    expts = [
        ExperimentSetting(TensorProductState(), sX(0) * sY(1)),
        ExperimentSetting(plusZ(0), sZ(0)),
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    assert len(suite) == 2
    for e1, e2 in zip(expts, suite):
        # experiment suite puts in groups of length 1
        assert len(e2) == 1
        e2 = e2[0]
        assert e1 == e2
    prog_str = str(suite).splitlines()[3:5]
    assert prog_str == EXPERIMENT_REPR.splitlines()[4:6]


def test_tomo_experiment_pre_grouped():
    expts = [
        [ExperimentSetting(TensorProductState(), sX(0) * sI(1)),
         ExperimentSetting(TensorProductState(), sI(0) * sX(1))],
        [ExperimentSetting(TensorProductState(), sZ(0) * sI(1)),
         ExperimentSetting(TensorProductState(), sI(0) * sZ(1))],
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    assert len(suite) == 2  # number of groups
    for es1, es2 in zip(expts, suite):
        for e1, e2 in zip(es1, es2):
            assert e1 == e2
    prog_str = str(suite).splitlines()[3:5]
    assert prog_str == EXPERIMENT_REPR.splitlines()[4:6]


def test_tomo_experiment_empty():
    suite = TomographyExperiment([], program=Program(X(0)))
    assert len(suite) == 0
    assert str(suite.program) == 'X 0\n'


def test_experiment_deser(tmpdir):
    expts = [
        [ExperimentSetting(TensorProductState(), sX(0) * sI(1)),
         ExperimentSetting(TensorProductState(), sI(0) * sX(1))],
        [ExperimentSetting(TensorProductState(), sZ(0) * sI(1)),
         ExperimentSetting(TensorProductState(), sI(0) * sZ(1))],
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    to_json(f'{tmpdir}/suite.json', suite)
    suite2 = read_json(f'{tmpdir}/suite.json')
    assert suite == suite2


def test_experiment_result_compat():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9,
        std_err=0.05,
        total_counts=100,
    )
    assert str(er) == 'X0_0→(1+0j)*Z0: 0.9 +- 0.05'


def test_experiment_result():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9,
        std_err=0.05,
        total_counts=100,
    )
    assert str(er) == 'X0_0→(1+0j)*Z0: 0.9 +- 0.05'


DEFGATE_X = """
DEFGATE XGATE:
    0, 1
    1, 0
"""


TRIMMED_PROG = """
DEFGATE XGATE:
    0, 1
    1, 0

X 0
"""


def test_remove_reset_from_program():
    p = Program(DEFGATE_X)
    p += RESET()
    p += X(0)
    new_p = _remove_reset_from_program(p)
    assert '\n' + new_p.out() == TRIMMED_PROG
