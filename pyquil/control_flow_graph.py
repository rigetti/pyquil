from typing import List, Optional
from typing_extensions import Self, override

from quil import program as quil_rs

from pyquil.quilbase import _convert_to_py_instruction, _convert_to_py_instructions, AbstractInstruction


class BasicBlock(quil_rs.BasicBlock):
    """
    Represents a basic block in the Program.

    Most functionality is implemented by the `quil` package. See the
    `quil documentation`_ for documentation and available methods.

    .. _quil documentation: https://rigetti.github.io/quil-rs/quil/program.html#BasicBlock
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


class ControlFlowGraph(quil_rs.ControlFlowGraph):
    """
    Representation of a control flow graph (CFG) for a Quil program.

    The CFG is a directed graph where each node is a basic block and each edge is a control flow transition between two
    basic blocks.

    This class should not be initialized directly. Use :py:meth:~pyquil.quil.Program.control
     flow_graph` to get a CFG for a program.

    Most functionality is implemented by the `quil` package. See the `quil documentation`_ for
    documentation and available methods.

    .. _quil documentation: https://rigetti.github.io/quil-rs/quil/program.html#ControlFlowGraph
    """

    @classmethod
    def _from_rs(cls, graph: quil_rs.ControlFlowGraph) -> Self:
        return super().__new__(cls, graph)

    @override
    def basic_blocks(self) -> List[BasicBlock]:  # type: ignore[override]
        """
        Return a list of all the basic blocks in the control flow graph, in order of definition.
        """
        return [BasicBlock._from_rs(block) for block in super().basic_blocks()]
