from collections.abc import Sequence

import numpy as np
import pytest
from quil.instructions import Qubit
from syrupy.assertion import SnapshotAssertion

from pyquil.quilatom import Add, Frame, Label, LabelPlaceholder, Parameter, QubitPlaceholder


@pytest.mark.parametrize(
    ("qubits", "name"),
    [
        ([Qubit(0)], "FRAME"),
        ([Qubit("One")], "FRAME"),
        ([2], "FRAME"),
        ([Qubit(0), Qubit("One"), 2], "FRAME"),
    ],
    ids=("With-Fixed", "With-Variable", "With-int", "With-Mixed"),
)
class TestFrame:
    @pytest.fixture
    def frame(self, qubits: Sequence[int | Qubit], name: str):
        return Frame(qubits, name)

    def test_out(self, frame: Frame, snapshot: SnapshotAssertion):
        assert frame.out() == snapshot

    def test_str(self, frame: Frame, snapshot: SnapshotAssertion):
        assert str(frame) == snapshot

    def test_qubits(self, frame: Frame, qubits: Sequence[int | Qubit]):
        expected = [Qubit(q) if isinstance(q, int) else q for q in qubits]
        assert frame.qubits == expected

    def test_name(self, frame: Frame, name: str):
        assert frame.name == name

    def test_eq(self, frame: Frame):
        assert frame == frame
        assert frame != Frame([], "definitely-not-eq")


def test_label():
    name = "my-label"
    label = Label(name)
    assert label.name == name
    assert label.out() == str(label) == f"@{name}"
    assert label == Label(name)
    assert hash(label) == hash(Label(name))
    label.name = "new-label"
    assert label.name == "new-label"


def test_label_placeholder():
    prefix = "my-prefix"
    placeholder = LabelPlaceholder(prefix)
    with pytest.raises(RuntimeError):
        placeholder.out()
    assert placeholder.prefix == prefix
    assert placeholder != LabelPlaceholder(prefix)
    assert hash(placeholder) != hash(LabelPlaceholder(prefix))


def test_qubit_placeholder():
    register = QubitPlaceholder.register(2)
    assert len(register) == 2
    assert register[0] != register[1]


def test_arithmetic_with_numpy():
    x = Parameter("x")
    expression = np.float64(1.0) + x
    assert expression == Add(np.float64(1.0), x)
