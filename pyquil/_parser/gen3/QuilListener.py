# Generated from Quil.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .QuilParser import QuilParser
else:
    from QuilParser import QuilParser

# This class defines a complete listener for a parse tree produced by QuilParser.
class QuilListener(ParseTreeListener):

    # Enter a parse tree produced by QuilParser#quil.
    def enterQuil(self, ctx:QuilParser.QuilContext):
        pass

    # Exit a parse tree produced by QuilParser#quil.
    def exitQuil(self, ctx:QuilParser.QuilContext):
        pass


    # Enter a parse tree produced by QuilParser#allInstr.
    def enterAllInstr(self, ctx:QuilParser.AllInstrContext):
        pass

    # Exit a parse tree produced by QuilParser#allInstr.
    def exitAllInstr(self, ctx:QuilParser.AllInstrContext):
        pass


    # Enter a parse tree produced by QuilParser#instr.
    def enterInstr(self, ctx:QuilParser.InstrContext):
        pass

    # Exit a parse tree produced by QuilParser#instr.
    def exitInstr(self, ctx:QuilParser.InstrContext):
        pass


    # Enter a parse tree produced by QuilParser#gate.
    def enterGate(self, ctx:QuilParser.GateContext):
        pass

    # Exit a parse tree produced by QuilParser#gate.
    def exitGate(self, ctx:QuilParser.GateContext):
        pass


    # Enter a parse tree produced by QuilParser#name.
    def enterName(self, ctx:QuilParser.NameContext):
        pass

    # Exit a parse tree produced by QuilParser#name.
    def exitName(self, ctx:QuilParser.NameContext):
        pass


    # Enter a parse tree produced by QuilParser#qubit.
    def enterQubit(self, ctx:QuilParser.QubitContext):
        pass

    # Exit a parse tree produced by QuilParser#qubit.
    def exitQubit(self, ctx:QuilParser.QubitContext):
        pass


    # Enter a parse tree produced by QuilParser#param.
    def enterParam(self, ctx:QuilParser.ParamContext):
        pass

    # Exit a parse tree produced by QuilParser#param.
    def exitParam(self, ctx:QuilParser.ParamContext):
        pass


    # Enter a parse tree produced by QuilParser#modifier.
    def enterModifier(self, ctx:QuilParser.ModifierContext):
        pass

    # Exit a parse tree produced by QuilParser#modifier.
    def exitModifier(self, ctx:QuilParser.ModifierContext):
        pass


    # Enter a parse tree produced by QuilParser#defGate.
    def enterDefGate(self, ctx:QuilParser.DefGateContext):
        pass

    # Exit a parse tree produced by QuilParser#defGate.
    def exitDefGate(self, ctx:QuilParser.DefGateContext):
        pass


    # Enter a parse tree produced by QuilParser#defGateAsPauli.
    def enterDefGateAsPauli(self, ctx:QuilParser.DefGateAsPauliContext):
        pass

    # Exit a parse tree produced by QuilParser#defGateAsPauli.
    def exitDefGateAsPauli(self, ctx:QuilParser.DefGateAsPauliContext):
        pass


    # Enter a parse tree produced by QuilParser#variable.
    def enterVariable(self, ctx:QuilParser.VariableContext):
        pass

    # Exit a parse tree produced by QuilParser#variable.
    def exitVariable(self, ctx:QuilParser.VariableContext):
        pass


    # Enter a parse tree produced by QuilParser#gatetype.
    def enterGatetype(self, ctx:QuilParser.GatetypeContext):
        pass

    # Exit a parse tree produced by QuilParser#gatetype.
    def exitGatetype(self, ctx:QuilParser.GatetypeContext):
        pass


    # Enter a parse tree produced by QuilParser#matrix.
    def enterMatrix(self, ctx:QuilParser.MatrixContext):
        pass

    # Exit a parse tree produced by QuilParser#matrix.
    def exitMatrix(self, ctx:QuilParser.MatrixContext):
        pass


    # Enter a parse tree produced by QuilParser#matrixRow.
    def enterMatrixRow(self, ctx:QuilParser.MatrixRowContext):
        pass

    # Exit a parse tree produced by QuilParser#matrixRow.
    def exitMatrixRow(self, ctx:QuilParser.MatrixRowContext):
        pass


    # Enter a parse tree produced by QuilParser#pauliTerms.
    def enterPauliTerms(self, ctx:QuilParser.PauliTermsContext):
        pass

    # Exit a parse tree produced by QuilParser#pauliTerms.
    def exitPauliTerms(self, ctx:QuilParser.PauliTermsContext):
        pass


    # Enter a parse tree produced by QuilParser#pauliTerm.
    def enterPauliTerm(self, ctx:QuilParser.PauliTermContext):
        pass

    # Exit a parse tree produced by QuilParser#pauliTerm.
    def exitPauliTerm(self, ctx:QuilParser.PauliTermContext):
        pass


    # Enter a parse tree produced by QuilParser#defCircuit.
    def enterDefCircuit(self, ctx:QuilParser.DefCircuitContext):
        pass

    # Exit a parse tree produced by QuilParser#defCircuit.
    def exitDefCircuit(self, ctx:QuilParser.DefCircuitContext):
        pass


    # Enter a parse tree produced by QuilParser#qubitVariable.
    def enterQubitVariable(self, ctx:QuilParser.QubitVariableContext):
        pass

    # Exit a parse tree produced by QuilParser#qubitVariable.
    def exitQubitVariable(self, ctx:QuilParser.QubitVariableContext):
        pass


    # Enter a parse tree produced by QuilParser#circuitQubit.
    def enterCircuitQubit(self, ctx:QuilParser.CircuitQubitContext):
        pass

    # Exit a parse tree produced by QuilParser#circuitQubit.
    def exitCircuitQubit(self, ctx:QuilParser.CircuitQubitContext):
        pass


    # Enter a parse tree produced by QuilParser#circuitGate.
    def enterCircuitGate(self, ctx:QuilParser.CircuitGateContext):
        pass

    # Exit a parse tree produced by QuilParser#circuitGate.
    def exitCircuitGate(self, ctx:QuilParser.CircuitGateContext):
        pass


    # Enter a parse tree produced by QuilParser#circuitMeasure.
    def enterCircuitMeasure(self, ctx:QuilParser.CircuitMeasureContext):
        pass

    # Exit a parse tree produced by QuilParser#circuitMeasure.
    def exitCircuitMeasure(self, ctx:QuilParser.CircuitMeasureContext):
        pass


    # Enter a parse tree produced by QuilParser#circuitResetState.
    def enterCircuitResetState(self, ctx:QuilParser.CircuitResetStateContext):
        pass

    # Exit a parse tree produced by QuilParser#circuitResetState.
    def exitCircuitResetState(self, ctx:QuilParser.CircuitResetStateContext):
        pass


    # Enter a parse tree produced by QuilParser#circuitInstr.
    def enterCircuitInstr(self, ctx:QuilParser.CircuitInstrContext):
        pass

    # Exit a parse tree produced by QuilParser#circuitInstr.
    def exitCircuitInstr(self, ctx:QuilParser.CircuitInstrContext):
        pass


    # Enter a parse tree produced by QuilParser#circuit.
    def enterCircuit(self, ctx:QuilParser.CircuitContext):
        pass

    # Exit a parse tree produced by QuilParser#circuit.
    def exitCircuit(self, ctx:QuilParser.CircuitContext):
        pass


    # Enter a parse tree produced by QuilParser#measure.
    def enterMeasure(self, ctx:QuilParser.MeasureContext):
        pass

    # Exit a parse tree produced by QuilParser#measure.
    def exitMeasure(self, ctx:QuilParser.MeasureContext):
        pass


    # Enter a parse tree produced by QuilParser#addr.
    def enterAddr(self, ctx:QuilParser.AddrContext):
        pass

    # Exit a parse tree produced by QuilParser#addr.
    def exitAddr(self, ctx:QuilParser.AddrContext):
        pass


    # Enter a parse tree produced by QuilParser#defLabel.
    def enterDefLabel(self, ctx:QuilParser.DefLabelContext):
        pass

    # Exit a parse tree produced by QuilParser#defLabel.
    def exitDefLabel(self, ctx:QuilParser.DefLabelContext):
        pass


    # Enter a parse tree produced by QuilParser#label.
    def enterLabel(self, ctx:QuilParser.LabelContext):
        pass

    # Exit a parse tree produced by QuilParser#label.
    def exitLabel(self, ctx:QuilParser.LabelContext):
        pass


    # Enter a parse tree produced by QuilParser#halt.
    def enterHalt(self, ctx:QuilParser.HaltContext):
        pass

    # Exit a parse tree produced by QuilParser#halt.
    def exitHalt(self, ctx:QuilParser.HaltContext):
        pass


    # Enter a parse tree produced by QuilParser#jump.
    def enterJump(self, ctx:QuilParser.JumpContext):
        pass

    # Exit a parse tree produced by QuilParser#jump.
    def exitJump(self, ctx:QuilParser.JumpContext):
        pass


    # Enter a parse tree produced by QuilParser#jumpWhen.
    def enterJumpWhen(self, ctx:QuilParser.JumpWhenContext):
        pass

    # Exit a parse tree produced by QuilParser#jumpWhen.
    def exitJumpWhen(self, ctx:QuilParser.JumpWhenContext):
        pass


    # Enter a parse tree produced by QuilParser#jumpUnless.
    def enterJumpUnless(self, ctx:QuilParser.JumpUnlessContext):
        pass

    # Exit a parse tree produced by QuilParser#jumpUnless.
    def exitJumpUnless(self, ctx:QuilParser.JumpUnlessContext):
        pass


    # Enter a parse tree produced by QuilParser#resetState.
    def enterResetState(self, ctx:QuilParser.ResetStateContext):
        pass

    # Exit a parse tree produced by QuilParser#resetState.
    def exitResetState(self, ctx:QuilParser.ResetStateContext):
        pass


    # Enter a parse tree produced by QuilParser#wait.
    def enterWait(self, ctx:QuilParser.WaitContext):
        pass

    # Exit a parse tree produced by QuilParser#wait.
    def exitWait(self, ctx:QuilParser.WaitContext):
        pass


    # Enter a parse tree produced by QuilParser#memoryDescriptor.
    def enterMemoryDescriptor(self, ctx:QuilParser.MemoryDescriptorContext):
        pass

    # Exit a parse tree produced by QuilParser#memoryDescriptor.
    def exitMemoryDescriptor(self, ctx:QuilParser.MemoryDescriptorContext):
        pass


    # Enter a parse tree produced by QuilParser#offsetDescriptor.
    def enterOffsetDescriptor(self, ctx:QuilParser.OffsetDescriptorContext):
        pass

    # Exit a parse tree produced by QuilParser#offsetDescriptor.
    def exitOffsetDescriptor(self, ctx:QuilParser.OffsetDescriptorContext):
        pass


    # Enter a parse tree produced by QuilParser#classicalUnary.
    def enterClassicalUnary(self, ctx:QuilParser.ClassicalUnaryContext):
        pass

    # Exit a parse tree produced by QuilParser#classicalUnary.
    def exitClassicalUnary(self, ctx:QuilParser.ClassicalUnaryContext):
        pass


    # Enter a parse tree produced by QuilParser#classicalBinary.
    def enterClassicalBinary(self, ctx:QuilParser.ClassicalBinaryContext):
        pass

    # Exit a parse tree produced by QuilParser#classicalBinary.
    def exitClassicalBinary(self, ctx:QuilParser.ClassicalBinaryContext):
        pass


    # Enter a parse tree produced by QuilParser#logicalBinaryOp.
    def enterLogicalBinaryOp(self, ctx:QuilParser.LogicalBinaryOpContext):
        pass

    # Exit a parse tree produced by QuilParser#logicalBinaryOp.
    def exitLogicalBinaryOp(self, ctx:QuilParser.LogicalBinaryOpContext):
        pass


    # Enter a parse tree produced by QuilParser#arithmeticBinaryOp.
    def enterArithmeticBinaryOp(self, ctx:QuilParser.ArithmeticBinaryOpContext):
        pass

    # Exit a parse tree produced by QuilParser#arithmeticBinaryOp.
    def exitArithmeticBinaryOp(self, ctx:QuilParser.ArithmeticBinaryOpContext):
        pass


    # Enter a parse tree produced by QuilParser#move.
    def enterMove(self, ctx:QuilParser.MoveContext):
        pass

    # Exit a parse tree produced by QuilParser#move.
    def exitMove(self, ctx:QuilParser.MoveContext):
        pass


    # Enter a parse tree produced by QuilParser#exchange.
    def enterExchange(self, ctx:QuilParser.ExchangeContext):
        pass

    # Exit a parse tree produced by QuilParser#exchange.
    def exitExchange(self, ctx:QuilParser.ExchangeContext):
        pass


    # Enter a parse tree produced by QuilParser#convert.
    def enterConvert(self, ctx:QuilParser.ConvertContext):
        pass

    # Exit a parse tree produced by QuilParser#convert.
    def exitConvert(self, ctx:QuilParser.ConvertContext):
        pass


    # Enter a parse tree produced by QuilParser#load.
    def enterLoad(self, ctx:QuilParser.LoadContext):
        pass

    # Exit a parse tree produced by QuilParser#load.
    def exitLoad(self, ctx:QuilParser.LoadContext):
        pass


    # Enter a parse tree produced by QuilParser#store.
    def enterStore(self, ctx:QuilParser.StoreContext):
        pass

    # Exit a parse tree produced by QuilParser#store.
    def exitStore(self, ctx:QuilParser.StoreContext):
        pass


    # Enter a parse tree produced by QuilParser#classicalComparison.
    def enterClassicalComparison(self, ctx:QuilParser.ClassicalComparisonContext):
        pass

    # Exit a parse tree produced by QuilParser#classicalComparison.
    def exitClassicalComparison(self, ctx:QuilParser.ClassicalComparisonContext):
        pass


    # Enter a parse tree produced by QuilParser#nop.
    def enterNop(self, ctx:QuilParser.NopContext):
        pass

    # Exit a parse tree produced by QuilParser#nop.
    def exitNop(self, ctx:QuilParser.NopContext):
        pass


    # Enter a parse tree produced by QuilParser#include.
    def enterInclude(self, ctx:QuilParser.IncludeContext):
        pass

    # Exit a parse tree produced by QuilParser#include.
    def exitInclude(self, ctx:QuilParser.IncludeContext):
        pass


    # Enter a parse tree produced by QuilParser#pragma.
    def enterPragma(self, ctx:QuilParser.PragmaContext):
        pass

    # Exit a parse tree produced by QuilParser#pragma.
    def exitPragma(self, ctx:QuilParser.PragmaContext):
        pass


    # Enter a parse tree produced by QuilParser#pragma_name.
    def enterPragma_name(self, ctx:QuilParser.Pragma_nameContext):
        pass

    # Exit a parse tree produced by QuilParser#pragma_name.
    def exitPragma_name(self, ctx:QuilParser.Pragma_nameContext):
        pass


    # Enter a parse tree produced by QuilParser#numberExp.
    def enterNumberExp(self, ctx:QuilParser.NumberExpContext):
        pass

    # Exit a parse tree produced by QuilParser#numberExp.
    def exitNumberExp(self, ctx:QuilParser.NumberExpContext):
        pass


    # Enter a parse tree produced by QuilParser#powerExp.
    def enterPowerExp(self, ctx:QuilParser.PowerExpContext):
        pass

    # Exit a parse tree produced by QuilParser#powerExp.
    def exitPowerExp(self, ctx:QuilParser.PowerExpContext):
        pass


    # Enter a parse tree produced by QuilParser#mulDivExp.
    def enterMulDivExp(self, ctx:QuilParser.MulDivExpContext):
        pass

    # Exit a parse tree produced by QuilParser#mulDivExp.
    def exitMulDivExp(self, ctx:QuilParser.MulDivExpContext):
        pass


    # Enter a parse tree produced by QuilParser#parenthesisExp.
    def enterParenthesisExp(self, ctx:QuilParser.ParenthesisExpContext):
        pass

    # Exit a parse tree produced by QuilParser#parenthesisExp.
    def exitParenthesisExp(self, ctx:QuilParser.ParenthesisExpContext):
        pass


    # Enter a parse tree produced by QuilParser#variableExp.
    def enterVariableExp(self, ctx:QuilParser.VariableExpContext):
        pass

    # Exit a parse tree produced by QuilParser#variableExp.
    def exitVariableExp(self, ctx:QuilParser.VariableExpContext):
        pass


    # Enter a parse tree produced by QuilParser#addrExp.
    def enterAddrExp(self, ctx:QuilParser.AddrExpContext):
        pass

    # Exit a parse tree produced by QuilParser#addrExp.
    def exitAddrExp(self, ctx:QuilParser.AddrExpContext):
        pass


    # Enter a parse tree produced by QuilParser#signedExp.
    def enterSignedExp(self, ctx:QuilParser.SignedExpContext):
        pass

    # Exit a parse tree produced by QuilParser#signedExp.
    def exitSignedExp(self, ctx:QuilParser.SignedExpContext):
        pass


    # Enter a parse tree produced by QuilParser#addSubExp.
    def enterAddSubExp(self, ctx:QuilParser.AddSubExpContext):
        pass

    # Exit a parse tree produced by QuilParser#addSubExp.
    def exitAddSubExp(self, ctx:QuilParser.AddSubExpContext):
        pass


    # Enter a parse tree produced by QuilParser#functionExp.
    def enterFunctionExp(self, ctx:QuilParser.FunctionExpContext):
        pass

    # Exit a parse tree produced by QuilParser#functionExp.
    def exitFunctionExp(self, ctx:QuilParser.FunctionExpContext):
        pass


    # Enter a parse tree produced by QuilParser#function.
    def enterFunction(self, ctx:QuilParser.FunctionContext):
        pass

    # Exit a parse tree produced by QuilParser#function.
    def exitFunction(self, ctx:QuilParser.FunctionContext):
        pass


    # Enter a parse tree produced by QuilParser#sign.
    def enterSign(self, ctx:QuilParser.SignContext):
        pass

    # Exit a parse tree produced by QuilParser#sign.
    def exitSign(self, ctx:QuilParser.SignContext):
        pass


    # Enter a parse tree produced by QuilParser#number.
    def enterNumber(self, ctx:QuilParser.NumberContext):
        pass

    # Exit a parse tree produced by QuilParser#number.
    def exitNumber(self, ctx:QuilParser.NumberContext):
        pass


    # Enter a parse tree produced by QuilParser#imaginaryN.
    def enterImaginaryN(self, ctx:QuilParser.ImaginaryNContext):
        pass

    # Exit a parse tree produced by QuilParser#imaginaryN.
    def exitImaginaryN(self, ctx:QuilParser.ImaginaryNContext):
        pass


    # Enter a parse tree produced by QuilParser#realN.
    def enterRealN(self, ctx:QuilParser.RealNContext):
        pass

    # Exit a parse tree produced by QuilParser#realN.
    def exitRealN(self, ctx:QuilParser.RealNContext):
        pass


