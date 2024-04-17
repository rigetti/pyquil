from typing import List, Iterable, TYPE_CHECKING, Optional
from typing_extensions import Self, override

from quil import program as quil_rs

if TYPE_CHECKING:
    from pyquil.quil import Program

from pyquil.quilbase import _convert_to_py_instruction, _convert_to_py_instructions, AbstractInstruction


class BasicBlock(quil_rs.BasicBlock):
    """
    Represents a basic block in the Program.

    Most functionality is implemented by the `quil` package. See the
    [quil documentation](https://rigetti.github.io/quil-rs/quil/program.html#BasicBlock) for available methods.
    """

    @classmethod
    def _from_rs(cls, block: quil_rs.BasicBlock) -> Self:
        return super().__new__(cls, block)

    @override
    def instructions(self) -> List[AbstractInstruction]:  # type: ignore[override]
        return _convert_to_py_instructions(super().instructions())

    @override
    def terminator(self) -> Optional[AbstractInstruction]:  # type: ignore[override]
        inst = super().terminator()
        if inst is None:
            return None
        return _convert_to_py_instruction(super().terminator())


class ControlFlowGraph:
    """
    Representation of a control flow graph (CFG) for a Quil program.

    The CFG is a directed graph where each node is a basic block and each edge is a control flow transition between two
    basic blocks.
    """

    def __init__(self, program: "Program"):
        self._graph = program._program.control_flow_graph()

    @staticmethod
    def _from_rs(graph: quil_rs.ControlFlowGraph) -> "ControlFlowGraph":
        from pyquil.quil import Program

        py_graph = ControlFlowGraph(Program())
        py_graph._graph = graph

        return py_graph

    def has_dynamic_control_flow(self) -> bool:
        """
        Return True if the program has dynamic control flow, i.e. contains a conditional branch instruction.

        False does not imply that there is only one basic block in the program. Multiple basic blocks may have
        non-conditional control flow among them, in which the execution order is deterministic and does not depend on
        program state. This may be a sequence of basic blocks with fixed JUMPs or without explicit terminators.
        """
        return self._graph.has_dynamic_control_flow()

    def basic_blocks(self) -> Iterable[BasicBlock]:
        """
        Return a list of all the basic blocks in the control flow graph, in order of definition.
        """
        for block in self._graph.basic_blocks():
            yield BasicBlock._from_rs(block)
