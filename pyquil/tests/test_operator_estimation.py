import functools
import itertools
import random
from math import pi
from operator import mul

import numpy as np
import pytest

from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from pyquil.operator_estimation import ExperimentSetting, TomographyExperiment, to_json, read_json, \
    group_experiments, ExperimentResult, measure_observables, SIC0, \
    SIC1, SIC2, SIC3, plusX, minusX, plusY, minusY, plusZ, minusZ, \
    _max_tpb_overlap, _max_weight_operator, _max_weight_state, \
    TensorProductState, zeros_state
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm


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


def test_setting_no_in_back_compat():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(sI(), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == sI()
        assert expt2.out_operator == oop


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


def test_tomo_experiment_pre_grouped():
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


def test_tomo_experiment_empty():
    suite = TomographyExperiment([], program=Program(X(0)), qubits=[0])
    assert len(suite) == 0
    assert str(suite.program) == 'X 0\n'


def test_experiment_deser(tmpdir):
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


@pytest.fixture(params=['clique-removal', 'greedy'])
def grouping_method(request):
    return request.param


def test_group_experiments(grouping_method):
    expts = [  # cf above, I removed the inner nesting. Still grouped visually
        ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1)),
        ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1)),
    ]
    suite = TomographyExperiment(expts, Program(), qubits=[0, 1])
    grouped_suite = group_experiments(suite, method=grouping_method)
    assert len(suite) == 4
    assert len(grouped_suite) == 2


def test_experiment_result_compat():
    er = ExperimentResult(
        setting=ExperimentSetting(sX(0), sZ(0)),
        expectation=0.9,
        stddev=0.05,
        total_counts=100,
    )
    assert str(er) == 'X0_0→(1+0j)*Z0: 0.9 +- 0.05'


def test_experiment_result():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9,
        stddev=0.05,
        total_counts=100,
    )
    assert str(er) == 'X0_0→(1+0j)*Z0: 0.9 +- 0.05'


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


def test_max_weight_operator_1():
    pauli_terms = [sZ(0),
                   sX(1) * sZ(0),
                   sY(2) * sX(1)]
    assert _max_weight_operator(pauli_terms) == sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_2():
    pauli_terms = [sZ(0),
                   sX(1) * sZ(0),
                   sY(2) * sX(1),
                   sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_3():
    pauli_terms = [sZ(0) * sX(5),
                   sX(1) * sZ(0),
                   sY(2) * sX(1),
                   sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) is None


def test_max_weight_state_1():
    states = [plusX(0) * plusZ(1),
              plusX(0),
              plusZ(1),
              ]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_2():
    states = [plusX(1) * plusZ(0),
              plusX(0),
              plusZ(1),
              ]
    assert _max_weight_state(states) is None


def test_max_weight_state_3():
    states = [plusX(0) * minusZ(1),
              plusX(0),
              minusZ(1),
              ]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_4():
    states = [plusX(1) * minusZ(0),
              plusX(0),
              minusZ(1),
              ]
    assert _max_weight_state(states) is None


def test_max_tpb_overlap_1():
    tomo_expt_settings = [ExperimentSetting(sZ(1) * sX(0), sY(2) * sY(1)),
                          ExperimentSetting(sX(2) * sZ(1), sY(2) * sZ(0))]
    tomo_expt_program = Program(H(0), H(1), H(2))
    tomo_expt_qubits = [0, 1, 2]
    tomo_expt = TomographyExperiment(tomo_expt_settings, tomo_expt_program, tomo_expt_qubits)
    expected_dict = {
        ExperimentSetting(plusX(0) * plusZ(1) * plusX(2), sZ(0) * sY(1) * sY(2)): [
            ExperimentSetting(plusZ(1) * plusX(0), sY(2) * sY(1)),
            ExperimentSetting(plusX(2) * plusZ(1), sY(2) * sZ(0))
        ]
    }
    assert expected_dict == _max_tpb_overlap(tomo_expt)


def test_max_tpb_overlap_2():
    expt_setting = ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    p = Program(H(0), H(1), H(2))
    qubits = [0, 1, 2]
    tomo_expt = TomographyExperiment([expt_setting], p, qubits)
    expected_dict = {expt_setting: [expt_setting]}
    assert expected_dict == _max_tpb_overlap(tomo_expt)


def test_max_tpb_overlap_3():
    # add another ExperimentSetting to the above
    expt_setting = ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    expt_setting2 = ExperimentSetting(sZ(7), sY(1))
    p = Program(H(0), H(1), H(2))
    qubits = [0, 1, 2]
    tomo_expt2 = TomographyExperiment([expt_setting, expt_setting2], p, qubits)
    expected_dict2 = {expt_setting: [expt_setting, expt_setting2]}
    assert expected_dict2 == _max_tpb_overlap(tomo_expt2)


def test_group_experiments_greedy():
    ungrouped_tomo_expt = TomographyExperiment(
        [[ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                            PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))],
         [ExperimentSetting(sZ(7), sY(1))]], program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    grouped_tomo_expt = group_experiments(ungrouped_tomo_expt, method='greedy')
    expected_grouped_tomo_expt = TomographyExperiment(
        [[
            ExperimentSetting(TensorProductState.from_str('Z0_7 * Y0_8 * Z0_1 * Y0_4 * '
                                                          'Z0_2 * Y0_5 * Y0_0 * X0_6'),
                              PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1')),
            ExperimentSetting(plusZ(7), sY(1))
        ]],
        program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    assert grouped_tomo_expt == expected_grouped_tomo_expt


def test_expt_settings_diagonal_in_tpb():
    def _expt_settings_diagonal_in_tpb(es1: ExperimentSetting, es2: ExperimentSetting):
        """
        Extends the concept of being diagonal in the same tpb to ExperimentSettings, by
        determining if the pairs of in_states and out_operators are separately diagonal in the same
        tpb
        """
        max_weight_in = _max_weight_state([es1.in_state, es2.in_state])
        max_weight_out = _max_weight_operator([es1.out_operator, es2.out_operator])
        return max_weight_in is not None and max_weight_out is not None

    expt_setting1 = ExperimentSetting(plusZ(1) * plusX(0), sY(1) * sZ(0))
    expt_setting2 = ExperimentSetting(plusY(2) * plusZ(1), sZ(2) * sY(1))
    assert _expt_settings_diagonal_in_tpb(expt_setting1, expt_setting2)
    expt_setting3 = ExperimentSetting(plusX(2) * plusZ(1), sZ(2) * sY(1))
    expt_setting4 = ExperimentSetting(plusY(2) * plusZ(1), sX(2) * sY(1))
    assert not _expt_settings_diagonal_in_tpb(expt_setting2, expt_setting3)
    assert not _expt_settings_diagonal_in_tpb(expt_setting2, expt_setting4)


def test_identity(forest):
    qc = get_qc('2q-qvm')
    suite = TomographyExperiment([ExperimentSetting(plusZ(0), 0.123 * sI(0))],
                                 program=Program(X(0)), qubits=[0])
    result = list(measure_observables(qc, suite))[0]
    assert result.expectation == 0.123


def test_sic_process_tomo(forest):
    qc = get_qc('2q-qvm')
    process = Program(X(0))
    settings = []
    for in_state in [SIC0, SIC1, SIC2, SIC3]:
        for out_op in [sI, sX, sY, sZ]:
            settings += [ExperimentSetting(
                in_state=in_state(q=0),
                out_operator=out_op(q=0)
            )]

    experiment = TomographyExperiment(settings=settings, program=process, qubits=[0])
    results = list(measure_observables(qc, experiment))
    assert len(results) == 4 * 4
