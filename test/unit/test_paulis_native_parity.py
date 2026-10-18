"""Parity checks between `pyquil.paulis` and the native `quil.instructions` Pauli classes.

Each test is run against both implementations. The ``pyquil`` variants are expected to pass; the
``quil`` variants document the gaps that block migrating `pyquil.paulis` to the native classes.
"""

import pytest
import quil.instructions as quil_rs

from pyquil import paulis as py_paulis
from pyquil.quilatom import QubitPlaceholder

IMPLS = {
    "pyquil": (py_paulis.PauliTerm, py_paulis.PauliSum),
    "quil": (quil_rs.PauliTerm, quil_rs.PauliSum),
}


@pytest.fixture(params=list(IMPLS))
def impl(request):
    return IMPLS[request.param]


def test_integer_qubits_preserved(impl):
    PauliTerm, _ = impl
    assert PauliTerm("X", 12).get_qubits() == [12]


def test_getitem_by_integer_qubit(impl):
    PauliTerm, _ = impl
    term = PauliTerm("X", 0) * PauliTerm("Y", 1)
    assert term[0] == "X"
    assert term[1] == "Y"
    assert term[5] == "I"


def test_pauli_string_with_qubits(impl):
    PauliTerm, _ = impl
    term = PauliTerm("X", 0) * PauliTerm("Y", 1)
    assert term.pauli_string(qubits=[0, 2]) == "XI"


def test_iter_yields_integer_qubits(impl):
    PauliTerm, _ = impl
    assert list(PauliTerm("X", 0) * PauliTerm("Y", 1)) == [(0, "X"), (1, "Y")]


def test_placeholder_qubit(impl):
    PauliTerm, _ = impl
    q = QubitPlaceholder()
    assert PauliTerm("X", q).get_qubits() == [q]


def test_term_sum_simplifies_like_terms(impl):
    PauliTerm, _ = impl
    total = PauliTerm("X", 0) + PauliTerm("X", 0)
    assert len(total) == 1
    assert total.terms[0].coefficient == 2


def test_term_difference_cancels(impl):
    PauliTerm, _ = impl
    assert len(PauliTerm("X", 0) - PauliTerm("X", 0)) == 0 or all(
        t.coefficient == 0 for t in (PauliTerm("X", 0) - PauliTerm("X", 0)).terms
    )
    assert len((PauliTerm("X", 0) - PauliTerm("X", 0)).terms) <= 1


def test_sum_square_simplifies(impl):
    PauliTerm, PauliSum = impl
    s = PauliSum([PauliTerm("X", 0), PauliTerm("Y", 1, 2.0)])
    # (X0 + 2 Y1)^2 = 5 I + 2 X0 Y1 + 2 Y1 X0 -> 5 I + 4 X0Y1
    assert len((s * s).terms) == 2


def test_scalar_plus_term_is_identity_term(impl):
    PauliTerm, _ = impl
    total = 1 + PauliTerm("X", 0)
    assert len(total) == 2


def test_scalar_minus_term_is_identity_term(impl):
    PauliTerm, _ = impl
    total = 1 - PauliTerm("X", 0)
    assert len(total) == 2


def test_sum_has_simplify(impl):
    PauliTerm, PauliSum = impl
    s = PauliSum([PauliTerm("X", 0), PauliTerm("X", 0)])
    assert len(s.simplify()) == 1


def test_sum_has_get_programs(impl):
    PauliTerm, PauliSum = impl
    programs, coefficients = PauliSum([PauliTerm("X", 0), PauliTerm("Z", 1)]).get_programs()
    assert len(programs) == 2
    assert len(coefficients) == 2


def test_sum_compact_str_round_trip(impl):
    PauliTerm, PauliSum = impl
    s = PauliSum([PauliTerm("X", 1, 2.0), PauliTerm("Z", 2, 1.5)])
    assert PauliSum.from_compact_str(s.compact_str()) == s


def test_term_compact_str_round_trip(impl):
    PauliTerm, _ = impl
    term = PauliTerm("X", 1, 2.0) * PauliTerm("Z", 2)
    assert PauliTerm.from_compact_str(term.compact_str()) == term


def test_id_sort_ops_keyword(impl):
    PauliTerm, _ = impl
    term = PauliTerm("X", 0) * PauliTerm("Y", 1)
    assert term.id(sort_ops=False) == "X0Y1"


def test_from_list_duplicate_qubits_raises(impl):
    PauliTerm, _ = impl
    with pytest.raises(ValueError):
        PauliTerm.from_list([("X", 0), ("Y", 0)])


def test_eq_with_other_type_raises(impl):
    PauliTerm, _ = impl
    with pytest.raises(TypeError):
        PauliTerm("X", 0) == 3  # noqa: B015


def test_coefficient_is_complex(impl):
    PauliTerm, _ = impl
    assert isinstance(PauliTerm("X", 0, 2.0).coefficient, complex)
