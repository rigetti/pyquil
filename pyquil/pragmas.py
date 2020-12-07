##############################################################################
# Copyright 2016-2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from typing import List, Iterable

from pyquil.quilbase import AbstractInstruction, Pragma
from pyquil.quil import Program


class InitialRewiring(Pragma):
    def __init__(self, rewiring: str):
        self.rewiring = rewiring

    def out(self) -> str:
        return f'PRAGMA INITIAL_REWIRING "{self.rewiring}"'


NAIVE_REWIRING = InitialRewiring("NAIVE")
RANDOM_REWIRING = InitialRewiring("RANDOM")
GREEDY_REWIRING = InitialRewiring("GREEDY")
PARTIAL_REWIRING = InitialRewiring("PARTIAL")


class PreserveBlock(Pragma):
    def __init__(self, *instructions: Iterable[AbstractInstruction]):
        self.instructions = Program(instructions)

    def out(self) -> str:
        return "PRAGMA PRESERVE_BLOCK\n{}\nPRAGMA END_PRESERVE_BLOCK".format(
            "\n".join([str(instr) for instr in self.instructions]),
        )


class CommutingBlocks(Pragma):
    def __init__(self, blocks: List[Iterable[AbstractInstruction]]):
        self.blocks = blocks

    def out(self) -> str:
        blocks = "\n".join([f"PRAGMA BLOCK\n{block}PRAGMA END_BLOCK" for block in self.blocks])
        return f"PRAGMA COMMUTING_BLOCKS\n{blocks}\nPRAGMA END_COMMUTING_BLOCKS"
