import itertools
import random
from math import pi

import numpy as np
import functools
from operator import mul

import pytest

from pyquil.api import WavefunctionSimulator
from pyquil.operator_estimation import ExperimentSetting, TomographyExperiment, to_json, read_json, \
    _all_qubits_diagonal_in_tpb, group_experiments, ExperimentResult, measure_observables, \
    diagonal_basis_commutes, get_diagonalizing_basis, _max_key_overlap_term_pair, \
    commuting_sets_by_zbasis_tomo_expt, tomo_expt_from_diagonal_sets, group_experiments_greedy
from pyquil.paulis import sI, sX, sY, sZ, PauliSum
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.paulis import *


def _generate_random_paulis(n_qubits, n_terms):
    paulis = [sI, sX, sY, sZ]
    all_op_inds = np.random.randint(len(paulis), size=(n_terms, n_qubits))
    operators = []
    for op_inds in all_op_inds:
        op = functools.reduce(mul, (paulis[pi](i) for i, pi in enumerate(op_inds)), sI(0))
        op *= np.random.uniform(-1, 1)
        operators += [op]
    return operators


def test_experiment():
    in_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for iop, oop in zip(in_ops, out_ops):
        expt = ExperimentSetting(iop, oop)
        assert str(expt) == expt.serializable()
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == iop
        assert expt2.out_operator == oop


def test_experiment_no_in():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(sI(), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == sI()
        assert expt2.out_operator == oop


def test_experiment_suite():
    expts = [
        ExperimentSetting(sI(), sX(0) * sY(1)),
        ExperimentSetting(sZ(0), sZ(0)),
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    assert len(suite) == 2
    for e1, e2 in zip(expts, suite):
        # experiment suite puts in groups of length 1
        assert len(e2) == 1
        e2 = e2[0]
        assert e1 == e2
    prog_str = str(suite).splitlines()[0]
    assert prog_str == 'X 0; Y 1'


def test_experiment_suite_pre_grouped():
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    assert len(suite) == 2  # number of groups
    for es1, es2 in zip(expts, suite):
        for e1, e2 in zip(es1, es2):
            assert e1 == e2
    prog_str = str(suite).splitlines()[0]
    assert prog_str == 'X 0; Y 1'


def test_experiment_suite_empty():
    suite = TomographyExperiment([], program=Program(X(0)), qubits=[0])
    assert len(suite) == 0
    assert str(suite.program) == 'X 0\n'


def test_suite_deser(tmpdir):
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    to_json(f'{tmpdir}/suite.json', suite)
    suite2 = read_json(f'{tmpdir}/suite.json')
    assert suite == suite2


def test_all_ops_belong_to_tpb():
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]
    for group in expts:
        for e1, e2 in itertools.combinations(group, 2):
            assert _all_qubits_diagonal_in_tpb(e1.in_operator, e2.in_operator)
            assert _all_qubits_diagonal_in_tpb(e1.out_operator, e2.out_operator)


def test_group_experiments():
    expts = [  # cf above, I removed the inner nesting. Still grouped visually
        ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1)),
        ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1)),
    ]
    suite = TomographyExperiment(expts, Program(), qubits=[0, 1])
    grouped_suite = group_experiments(suite)
    assert len(suite) == 4
    assert len(grouped_suite) == 2


def test_experiment_result():
    er = ExperimentResult(
        setting=ExperimentSetting(sX(0), sZ(0)),
        expectation=0.9,
        stddev=0.05,
    )
    assert str(er) == '(1+0j)*X0→(1+0j)*Z0: 0.9 +- 0.05'


def test_measure_observables(forest):
    expts = [
        ExperimentSetting(sI(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = TomographyExperiment(expts, program=Program(X(0), CNOT(0, 1)), qubits=[0, 1])
    assert len(suite) == 4 * 4
    gsuite = group_experiments(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    for res in measure_observables(qc, gsuite, n_shots=10_000):
        if res.setting.out_operator in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1


def _random_2q_programs(n_progs=10):
    """Generate random programs that consist of single qubit rotations, a CZ, and single
    qubit rotations.
    """
    r = random.Random(52)

    def RI(qubit, angle):
        # throw away angle so we can randomly choose the identity
        return I(qubit)

    def _random_1q_gate(qubit):
        return r.choice([RI, RX, RY, RZ])(qubit=qubit, angle=r.uniform(0, 2 * pi))

    for _ in range(n_progs):
        prog = Program()
        prog += _random_1q_gate(0)
        prog += _random_1q_gate(1)
        prog += CZ(0, 1)
        prog += _random_1q_gate(0)
        prog += _random_1q_gate(1)
        yield prog


def test_measure_observables_many_progs(forest):
    expts = [
        ExperimentSetting(sI(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]

    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 51
    for prog in _random_2q_programs():
        suite = TomographyExperiment(expts, program=prog, qubits=[0, 1])
        assert len(suite) == 4 * 4
        gsuite = group_experiments(suite)
        assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

        wfn = WavefunctionSimulator()
        wfn_exps = {}
        for expt in expts:
            wfn_exps[expt] = wfn.expectation(gsuite.program, PauliSum([expt.out_operator]))

        for res in measure_observables(qc, gsuite, n_shots=1_000):
            np.testing.assert_allclose(wfn_exps[res.setting], res.expectation, atol=0.1)


def test_append():
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]
    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    suite.append(ExperimentSetting(sI(), sY(0) * sX(1)))
    assert (len(str(suite))) > 0


def test_no_complex_coeffs(forest):
    qc = get_qc('2q-qvm')
    suite = TomographyExperiment([ExperimentSetting(sI(), 1.j * sY(0))], program=Program(X(0)),
                                 qubits=[0])
    with pytest.raises(ValueError):
        res = list(measure_observables(qc, suite))


def test_diagonal_basis_commutes():
    assert diagonal_basis_commutes(sZ(0), sZ(0) * sZ(1))
    assert diagonal_basis_commutes(sX(5), sZ(4))
    assert not diagonal_basis_commutes(sX(0), sY(0) * sZ(2))
    # this last example illustrates that a pair of commuting operators
    # need not be diagonal in the same tpb
    assert not diagonal_basis_commutes(sX(1) * sZ(0), sZ(1) * sX(0))


def test_get_diagonalizing_basis():
    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1)]
    assert get_diagonalizing_basis(pauli_terms) == sY(2) * sX(1) * sZ(0)
    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1), sZ(5) * sI(3)]
    assert get_diagonalizing_basis(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_key_overlap_term_pair():
    # test adding a new key-value pair
    expt_setting = ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    expected_dict = {(((0, 'Y'), (1, 'Z'), (2, 'Z'), (4, 'Y'), (5, 'Y'), (6, 'X'), (7, 'Z'), (8, 'Y')),
                      ((1, 'Y'), (3, 'X'), (4, 'Z'), (5, 'Y'), (7, 'Y'), (8, 'X'))):
                     ([PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6')],
                      [PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1')])}
    assert _max_key_overlap_term_pair(expt_setting, {}) == expected_dict
    # test adding to an already existing key
    expt_setting2 = ExperimentSetting(sZ(7), sY(1))
    expected_dict2 = {(((0, 'Y'), (1, 'Z'), (2, 'Z'), (4, 'Y'), (5, 'Y'), (6, 'X'), (7, 'Z'), (8, 'Y')),
                       ((1, 'Y'), (3, 'X'), (4, 'Z'), (5, 'Y'), (7, 'Y'), (8, 'X'))):
                      ([PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'), sZ(7)],
                       [PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'), sY(1)])}
    assert _max_key_overlap_term_pair(expt_setting2, expected_dict) == expected_dict2


def test_commuting_sets_by_zbasis_tomo_expt():
    tomo_expt_settings = [ExperimentSetting(sZ(1) * sX(0), sY(2) * sY(1)),
                          ExperimentSetting(sX(2) * sZ(1), sY(2) * sZ(0))]
    tomo_expt_program = Program(H(0), H(1), H(2))
    tomo_expt_qubits = [0, 1, 2]
    tomo_expt = TomographyExperiment(tomo_expt_settings, tomo_expt_program, tomo_expt_qubits)
    expected_dict = {(((0, 'X'), (1, 'Z'), (2, 'X')), ((0, 'Z'), (1, 'Y'), (2, 'Y'))):
                     ([sZ(1) * sX(0), sX(2) * sZ(1)], [sY(2) * sY(1), sY(2) * sZ(0)])}
    assert expected_dict == commuting_sets_by_zbasis_tomo_expt(tomo_expt)


def test_tomo_expt_from_diagonal_sets():
    ungrouped_tomo_expt = TomographyExperiment(
        [[ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                            PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))],
         [ExperimentSetting(sZ(7), sY(1))]], program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    diag_sets = {(((0, 'Y'), (1, 'Z'), (2, 'Z'), (4, 'Y'), (5, 'Y'), (6, 'X'), (7, 'Z'), (8, 'Y')),
                  ((1, 'Y'), (3, 'X'), (4, 'Z'), (5, 'Y'), (7, 'Y'), (8, 'X'))):
                 ([PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'), sZ(7)],
                  [PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'), sY(1)])}
    grouped_tomo_expt = tomo_expt_from_diagonal_sets(diag_sets, ungrouped_tomo_expt)
    expected_grouped_tomo_expt = TomographyExperiment(
        [[ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                            PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1')), ExperimentSetting(sZ(7), sY(1))]],
        program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    assert grouped_tomo_expt == expected_grouped_tomo_expt


def test_group_experiments_greedy():
    ungrouped_tomo_expt = TomographyExperiment(
        [[ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                            PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))],
         [ExperimentSetting(sZ(7), sY(1))]], program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    grouped_tomo_expt = group_experiments_greedy(ungrouped_tomo_expt)
    expected_grouped_tomo_expt = TomographyExperiment(
        [[ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                            PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1')), ExperimentSetting(sZ(7), sY(1))]],
        program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    assert grouped_tomo_expt == expected_grouped_tomo_expt


def test_identity(forest):
    qc = get_qc('2q-qvm')
    suite = TomographyExperiment([ExperimentSetting(sI(), 0.123 * sI(0))],
                                 program=Program(X(0)), qubits=[0])
    result = list(measure_observables(qc, suite))[0]
    assert result.expectation == 0.123
