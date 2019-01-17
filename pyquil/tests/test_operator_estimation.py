import itertools
import random
from math import pi
from unittest.mock import Mock
import numpy as np
import functools
from operator import mul
from scipy.stats import bernoulli
import pytest

from pyquil.api import WavefunctionSimulator, QVMConnection
from pyquil.operator_estimation import ExperimentSetting, TomographyExperiment, to_json, read_json, \
    _all_qubits_diagonal_in_tpb, group_experiments, ExperimentResult, measure_observables, \
    remove_imaginary, get_rotation_program_measure, get_parity, estimate_pauli_sum, CommutationError, \
    remove_identity, estimate_locally_commuting_operator, diagonal_basis_commutes, get_diagonalizing_basis, \
    _max_key_overlap, commuting_sets_by_zbasis
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm
from pyquil import Program, get_qc
from pyquil.gates import *


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


def test_identity(forest):
    qc = get_qc('2q-qvm')
    suite = TomographyExperiment([ExperimentSetting(sI(), 0.123 * sI(0))],
                                 program=Program(X(0)), qubits=[0])
    result = list(measure_observables(qc, suite))[0]
    assert result.expectation == 0.123


def test_imaginary_removal():
    """
    remove terms with imaginary coefficients from a pauli sum
    """
    test_term = 0.25 * sX(1) * sZ(2) * sX(3) + 0.25j * sX(1) * sZ(2) * sY(3)
    test_term += -0.25j * sY(1) * sZ(2) * sX(3) + 0.25 * sY(1) * sZ(2) * sY(3)
    true_term = 0.25 * sX(1) * sZ(2) * sX(3) + 0.25 * sY(1) * sZ(2) * sY(3)
    assert remove_imaginary(test_term) == true_term

    test_term = (0.25 + 1j) * sX(0) * sZ(2) + 1j * sZ(2)
    # is_identity in pyquil apparently thinks zero is identity
    assert remove_imaginary(test_term) == 0.25 * sX(0) * sZ(2)

    test_term = 0.25 * sX(0) * sZ(2) + 1j * sZ(2)
    assert remove_imaginary(test_term) == PauliSum([0.25 * sX(0) * sZ(2)])

    with pytest.raises(TypeError):
        remove_imaginary(5)

    with pytest.raises(TypeError):
        remove_imaginary(sX(0))


def test_rotation_programs():
    """
    Testing the generation of post rotations
    """
    test_term = sZ(0) * sX(20) * sI(100) * sY(5)
    rotations_to_do = [RX(np.pi / 2, 5), RY(-np.pi / 2, 20)]
    test_rotation_program = get_rotation_program_measure(test_term)
    # Since the rotations commute, it's sufficient to test membership in the program,
    # without ordering. However, it's true that a more complicated rotation could be performed,
    #  where the elements would not be free to be permuted. We ignore this possibility, for now.
    assert len(rotations_to_do) == len(test_rotation_program)
    for rotation in test_rotation_program:
        assert rotation in rotations_to_do


def test_get_parity():
    """
    Check if our way to compute parity is correct
    """
    single_qubit_results = [[0]] * 50 + [[1]] * 50
    single_qubit_parity_results = list(map(lambda x: -2 * x[0] + 1,
                                           single_qubit_results))

    # just making sure I constructed my test properly
    assert np.allclose(np.array([1] * 50 + [-1] * 50),
                       single_qubit_parity_results)

    test_results = get_parity([sZ(5)], single_qubit_results)
    assert np.allclose(single_qubit_parity_results, test_results[0, :])

    np.random.seed(87655678)
    brv1 = bernoulli(p=0.25)
    brv2 = bernoulli(p=0.4)
    n = 500
    two_qubit_measurements = list(zip(brv1.rvs(size=n), brv2.rvs(size=n)))
    pauli_terms = [sZ(0), sZ(1), sZ(0) * sZ(1)]
    parity_results = np.zeros((len(pauli_terms), n))
    parity_results[0, :] = [-2 * x[0] + 1 for x in two_qubit_measurements]
    parity_results[1, :] = [-2 * x[1] + 1 for x in two_qubit_measurements]
    parity_results[2, :] = [-2 * (sum(x) % 2) + 1 for x in
                            two_qubit_measurements]

    test_parity_results = get_parity(pauli_terms, two_qubit_measurements)
    assert np.allclose(test_parity_results, parity_results)


def test_estimate_pauli_sum():
    """
    Full test of the estimation procedures
    """
    quantum_resource = QVMConnection()

    # type checks
    with pytest.raises(TypeError):
        estimate_pauli_sum('5', {0: 'X', 1: 'Z'}, Program(), 1.0E-3,
                           quantum_resource)

    with pytest.raises(CommutationError):
        estimate_pauli_sum([sX(0), sY(0)], {0: 'X', 1: 'Z'}, Program(), 1.0E-3,
                           quantum_resource)

    with pytest.raises(TypeError):
        estimate_pauli_sum(sX(0), {0: 'X', 1: 'Z'}, Program(), 1.0E-3,
                           quantum_resource)

    # mock out qvm
    np.random.seed(87655678)
    brv1 = bernoulli(p=0.25)
    brv2 = bernoulli(p=0.4)
    n = 500
    two_qubit_measurements = list(zip(brv1.rvs(size=n), brv2.rvs(size=n)))
    pauli_terms = [sZ(0), sZ(1), sZ(0) * sZ(1)]

    fakeQVM = Mock(spec=QVMConnection())
    fakeQVM.run = Mock(return_value=two_qubit_measurements)
    mean, means, cov, estimator_var, shots = estimate_pauli_sum(pauli_terms,
                                                                {0: 'Z', 1: 'Z'},
                                                                Program(),
                                                                1.0E-1,
                                                                fakeQVM,
                                                                symmetrize=False)
    parity_results = np.zeros((len(pauli_terms), n))
    parity_results[0, :] = [-2 * x[0] + 1 for x in two_qubit_measurements]
    parity_results[1, :] = [-2 * x[1] + 1 for x in two_qubit_measurements]
    parity_results[2, :] = [-2 * (sum(x) % 2) + 1 for x in
                            two_qubit_measurements]

    assert np.allclose(np.cov(parity_results, ddof=1), cov)
    assert np.isclose(np.sum(np.mean(parity_results, axis=1)), mean)
    assert np.allclose(np.mean(parity_results, axis=1), means)
    assert np.isclose(shots, n)
    variance_to_beat = np.sum(cov) / (n - 1)
    assert np.isclose(variance_to_beat, estimator_var)

    # Double the shots by ever so slightly decreasing variance bound
    double_two_q_measurements = two_qubit_measurements + two_qubit_measurements
    mean, means, cov, estimator_var, shots = estimate_pauli_sum(pauli_terms,
                                                                {0: 'Z', 1: 'Z'},
                                                                Program(),
                                                                variance_to_beat - 1.0E-8,
                                                                fakeQVM,
                                                                symmetrize=False)

    parity_results = np.zeros((len(pauli_terms), 2 * n))
    parity_results[0, :] = [-2 * x[0] + 1 for x in double_two_q_measurements]
    parity_results[1, :] = [-2 * x[1] + 1 for x in double_two_q_measurements]
    parity_results[2, :] = [-2 * (sum(x) % 2) + 1 for x in
                            double_two_q_measurements]

    assert np.allclose(np.cov(parity_results, ddof=1), cov)
    assert np.isclose(np.sum(np.mean(parity_results, axis=1)), mean)
    assert np.allclose(np.mean(parity_results, axis=1), means)
    assert np.isclose(shots, 2 * n)
    assert np.isclose(np.sum(cov) / (2 * n - 1), estimator_var)


def test_identity_removal():
    test_term = 0.25 * sX(1) * sZ(2) * sX(3) + 0.25j * sX(1) * sZ(2) * sY(3)
    test_term += -0.25j * sY(1) * sZ(2) * sX(3) + 0.25 * sY(1) * sZ(2) * sY(3)
    identity_term = 200 * sI(5)

    new_psum, identity_term_result = remove_identity(identity_term + test_term)
    assert test_term == new_psum
    assert identity_term_result == identity_term


def test_mutation_free_estimation():
    """
    Make sure the estimation routines do not mutate the programs the user sends
    This is accomplished by a deep copy in `estimate_pauli_sum'.
    """
    prog = Program().inst(I(0))
    pauli_sum = sX(0)  # measure in the X-basis

    # set up fake QVM
    fakeQVM = Mock(spec=QVMConnection())
    fakeQVM.run = Mock(return_value=[[0], [1]])

    expected_value, estimator_variance, total_shots = \
        estimate_locally_commuting_operator(prog, PauliSum([pauli_sum]),
                                            1.0E-3, quantum_resource=fakeQVM)

    # make sure RY(-pi/2) 0\nMEASURE 0 [0] was not added to the program the
    # user sees
    assert prog.out() == 'I 0\n'


def test_diagonal_basis_commutes():
    x_term = sX(0) * sX(1)
    z1_term = sZ(1)
    z0_term = sZ(0)
    z0z1_term = sZ(0) * sZ(1)
    assert not diagonal_basis_commutes(x_term, z1_term)
    assert not diagonal_basis_commutes(z0z1_term, x_term)

    assert diagonal_basis_commutes(z1_term, z0_term)
    assert diagonal_basis_commutes(z0z1_term, z0_term)
    assert diagonal_basis_commutes(z0z1_term, z1_term)
    assert diagonal_basis_commutes(z0z1_term, sI(1))
    assert diagonal_basis_commutes(z0z1_term, sI(2))
    assert diagonal_basis_commutes(z0z1_term, sX(5) * sY(7))


def test_get_diagonalizing_basis():
    xxxx_terms = sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + \
        sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    true_term = sX(1) * sX(2) * sX(3) * sX(4)
    assert get_diagonalizing_basis(xxxx_terms.terms) == true_term

    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + \
        sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    assert get_diagonalizing_basis(zzzz_terms.terms) == sZ(1) * sZ(2) * \
        sZ(3) * sZ(4)


def test_max_key_overlap():
    # adds to already existing key
    x0_term = sX(0)
    diag_sets = {((0, 'X'), (1, 'Z')): [sX(0) * sZ(1), sZ(1)], ((0, 'Y'), (1, 'Z')): [sY(0), sZ(1), sY(0) * sZ(1)]}
    d_expected = {((0, 'X'), (1, 'Z')): [sX(0) * sZ(1), sZ(1), sX(0)],
                  ((0, 'Y'), (1, 'Z')): [sY(0), sZ(1), sY(0) * sZ(1)]}
    assert _max_key_overlap(x0_term, diag_sets) == d_expected

    # adds a new key
    x0_term = sX(0)
    diag_sets = {((0, 'Z'), (1, 'Z')): [sZ(0) * sZ(1), sZ(1)], ((0, 'Y'), (1, 'Z')): [sY(0), sZ(1), sY(0) * sZ(1)]}
    d_expected = d_expected = {((0, 'Z'), (1, 'Z')): [sZ(0) * sZ(1), sZ(1)],
                               ((0, 'Y'), (1, 'Z')): [sY(0), sZ(1), sY(0) * sZ(1)],
                               ((0, 'X'),): [sX(0)]}
    assert _max_key_overlap(x0_term, diag_sets) == d_expected


def test_commuting_sets_by_zbasis():
    # complicated coefficients, overlap on single qubit
    coeff1 = 0.012870253243021476
    term1 = PauliTerm.from_list([('X', 1), ('Z', 2), ('Y', 3), ('Y', 5),
                                 ('Z', 6), ('X', 7)],
                                coefficient=coeff1)

    coeff2 = 0.13131672212575296
    term2 = PauliTerm.from_list([('Z', 0), ('Z', 6)],
                                coefficient=coeff2)

    d_result = commuting_sets_by_zbasis(term1 + term2)
    d_expected = {((0, 'Z'), (1, 'X'), (2, 'Z'), (3, 'Y'), (5, 'Y'), (6, 'Z'), (7, 'X')):
                  [coeff1 * sX(1) * sZ(2) * sY(3) * sY(5) * sZ(6) * sX(7), coeff2 * sZ(0) * sZ(6)]}
    assert d_result == d_expected

    # clumping terms relevant for H2 into same diagonal bases
    x_term = sX(0) * sX(1)
    z1_term = sZ(1)
    z2_term = sZ(0)
    zz_term = sZ(0) * sZ(1)
    h2_hamiltonian = zz_term + z2_term + z1_term + x_term
    clumped_terms = commuting_sets_by_zbasis(h2_hamiltonian)
    true_set = {((0, 'X'), (1, 'X')): set([x_term.id()]),
                ((0, 'Z'), (1, 'Z')): set([z1_term.id(), z2_term.id(), zz_term.id()])}

    for key, value in clumped_terms.items():
        assert set(map(lambda x: x.id(), clumped_terms[key])) == true_set[key]

    # clumping 4-qubit terms into same diagonal bases
    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + \
        sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    xzxz_terms = sX(1) * sZ(2) + sX(3) * sZ(4) + \
        sX(1) * sZ(2) * sX(3) * sZ(4) + sX(1) * sX(3) * sZ(4)
    xxxx_terms = sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + \
        sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    yyyy_terms = sY(1) * sY(2) + sY(3) * sY(4) + sY(1) * sY(2) * sY(3) * sY(4)

    pauli_sum = zzzz_terms + xzxz_terms + xxxx_terms + yyyy_terms
    clumped_terms = commuting_sets_by_zbasis(pauli_sum)

    true_set = {((1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'Z')): set(map(lambda x: x.id(), zzzz_terms)),
                ((1, 'X'), (2, 'Z'), (3, 'X'), (4, 'Z')): set(map(lambda x: x.id(), xzxz_terms)),
                ((1, 'X'), (2, 'X'), (3, 'X'), (4, 'X')): set(map(lambda x: x.id(), xxxx_terms)),
                ((1, 'Y'), (2, 'Y'), (3, 'Y'), (4, 'Y')): set(map(lambda x: x.id(), yyyy_terms))}
    for key, value in clumped_terms.items():
        assert set(map(lambda x: x.id(), clumped_terms[key])) == true_set[key]
