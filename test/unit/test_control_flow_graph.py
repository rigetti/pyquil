from pyquil.control_flow_graph import BasicBlock
from pyquil.quilbase import AbstractInstruction
from pyquil.quil import Program


def test_control_flow_graph():
    program = Program(
        """
DEFFRAME 0 "flux_tx_cz":
    TEST: 1

DEFFRAME 1 "flux_tx_iswap":
    TEST: 1

DEFFRAME 1 "flux_tx_cz":
    TEST: 1

DEFFRAME 1 "flux_tx_iswap":
    TEST: 1

DEFFRAME 2 "flux_tx_cz":
    TEST: 1

DEFFRAME 2 "flux_tx_iswap":
    TEST: 1

DEFFRAME 3 "flux_tx_cz":
    TEST: 1

DEFFRAME 3 "flux_tx_iswap":
    TEST: 1

# Simplified version
DEFCAL CZ q0 q1:
    FENCE q0 q1
    SET-PHASE q0 "flux_tx_cz" 0.0
    SET-PHASE q1 "flux_tx_iswap" 0.0
    NONBLOCKING PULSE q0 "flux_tx_cz" erf_square(duration: 6.000000000000001e-08)
    NONBLOCKING PULSE q1 "flux_tx_iswap" erf_square(duration: 6.000000000000001e-08)
    SHIFT-PHASE q0 "flux_tx_cz" 1.0
    SHIFT-PHASE q1 "flux_tx_iswap" 1.0
    FENCE q0 q1

CZ 0 1
CZ 2 3
CZ 0 2
JUMP @END
LABEL @END
"""
    )
    graph = program.control_flow_graph()
    assert not graph.has_dynamic_control_flow()
    blocks = list(graph.basic_blocks())
    assert len(blocks) == 2
    for block in blocks:
        assert isinstance(block, BasicBlock)
        assert isinstance(block.terminator(), (type(None), AbstractInstruction))
        assert all([isinstance(instruction, AbstractInstruction) for instruction in block.instructions()])
        assert isinstance(block.gate_depth(1), int)
