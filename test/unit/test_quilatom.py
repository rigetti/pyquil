from typing import Sequence, Union
import pytest
from syrupy.assertion import SnapshotAssertion

from pyquil.quilatom import FormalArgument, Frame, Qubit, Label, LabelPlaceholder, QubitPlaceholder


@pytest.mark.parametrize(
    ("qubits", "name"),
    [
        ([Qubit(0)], "FRAME"),
        ([FormalArgument("One")], "FRAME"),
        ([2], "FRAME"),
        ([Qubit(0), FormalArgument("One"), 2], "FRAME"),
    ],
    ids=("With-Qubit", "With-FormalArgument", "With-int", "With-Mixed"),
)
class TestFrame:
    @pytest.fixture
    def frame(self, qubits: Sequence[Union[int, Qubit, FormalArgument]], name: str):
        return Frame(qubits, name)

    def test_out(self, frame: Frame, snapshot: SnapshotAssertion):
        assert frame.out() == snapshot

    def test_str(self, frame: Frame, snapshot: SnapshotAssertion):
        assert str(frame) == snapshot

    def test_qubits(self, frame: Frame, qubits: Sequence[Union[int, Qubit, FormalArgument]]):
        assert frame.qubits == tuple(Qubit(q) if isinstance(q, int) else q for q in qubits)
        frame.qubits = (Qubit(2), FormalArgument("One"))
        assert frame.qubits == (Qubit(2), FormalArgument("One"))

    def test_name(self, frame: Frame, name: str):
        assert frame.name == name
        frame.name = "new name"
        assert frame.name == "new name"

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
    with pytest.raises:
        placeholder.out()
    assert placeholder.prefix == prefix
    assert placeholder != LabelPlaceholder(prefix)
    assert hash(placeholder) != hash(LabelPlaceholder(prefix))


def test_qubit_placeholder():
    register = QubitPlaceholder.register(2)
    assert len(register) == 2
    with pytest.raises(RuntimeError):
        register[0].out()

    assert register[0] != register[1]
