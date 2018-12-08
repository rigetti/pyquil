##############################################################################
# Copyright 2018 Rigetti Computing
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
import ast
import sys
import functools
import inspect

from pyquil import gates
from pyquil.quil import Program
from pyquil.quilatom import Addr

if sys.version_info < (3, 7):
    from pyquil.external.contextvars import ContextVar
else:
    from contextvars import ContextVar

_program_context = ContextVar('program')


def program_context() -> Program:
    """
    Returns the current program in context. Will only work from inside calls to @magicquil.
    If called from outside @magicquil will throw a ValueError

    :return: program if one exists
    """
    return _program_context.get()


def I(qubit) -> None:
    program_context().inst(gates.I(qubit))


def X(qubit) -> None:
    program_context().inst(gates.X(qubit))


def H(qubit) -> None:
    program_context().inst(gates.H(qubit))


def CNOT(qubit1, qubit2) -> None:
    program_context().inst(gates.CNOT(qubit1, qubit2))


def MEASURE(qubit) -> Addr:
    program_context().inst(gates.MEASURE(qubit, qubit))
    return Addr(qubit)


def _if_statement(test, if_function, else_function) -> None:
    """
    Evaluate an if statement within a @magicquil block.

    If the test value is a Quil Addr then unwind it into quil code equivalent to an if then statement using jumps. Both
    sides of the if statement need to be evaluated and placed into separate Programs, which is why we create new
    program contexts for their evaluation.

    If the test value is not a Quil Addr then fall back to what Python would normally do with an if statement.

    Params are:
        if <test>:
            <if_function>
        else:
            <else_function>

    NB: This function must be named exactly _if_statement and be in scope for the ast transformer
    """
    if isinstance(test, Addr):
        token = _program_context.set(Program())
        if_function()
        if_program = _program_context.get()
        _program_context.reset(token)

        if else_function:
            token = _program_context.set(Program())
            else_function()
            else_program = _program_context.get()
            _program_context.reset(token)
        else:
            else_program = None

        program = _program_context.get()
        program.if_then(test, if_program, else_program)
    else:
        if test:
            if_function()
        elif else_function:
            else_function()


_EMPTY_ARGUMENTS = ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[])


class _IfTransformer(ast.NodeTransformer):
    """
    Transformer that unwraps the if and else branches into separate inner functions and then wraps them in a call to
    _if_statement. For example:

        if 1 + 1 == 2:
            print('math works')
        else:
            print('something is broken')

    would be transformed into:

        def _if_branch():
            print('math works')
        def _else_branch():
            print('something is broken')
        _if_statement(1 + 1 == 2, _if_branch, _else_branch)
    """
    def visit_If(self, node):
        # Must recursively visit the body of both the if and else bodies to handle any nested if and else statements
        # This also conveniently handles elif since those are just treated as a nested if/else within the else branch
        # See: https://greentreesnakes.readthedocs.io/en/latest/nodes.html#If
        node = self.generic_visit(node)

        if_function = ast.FunctionDef(name='_if_branch', body=node.body, decorator_list=[], args=_EMPTY_ARGUMENTS)
        else_function = ast.FunctionDef(name='_else_branch', body=node.orelse, decorator_list=[], args=_EMPTY_ARGUMENTS)

        if_function_name = ast.Name(id='_if_branch', ctx=ast.Load())
        else_function_name = ast.Name(id='_else_branch', ctx=ast.Load())

        if node.orelse:
            return [
                if_function,
                else_function,
                ast.Expr(ast.Call(
                    func=ast.Name(id='_if_statement', ctx=ast.Load()),
                    args=[node.test, if_function_name, else_function_name],
                    keywords=[]))
            ]
        else:
            return [
                if_function,
                ast.Expr(ast.Call(
                    func=ast.Name(id='_if_statement', ctx=ast.Load()),
                    args=[node.test, if_function_name, ast.NameConstant(None)],
                    keywords=[]))
            ]


def _rewrite_function(f):
    """
    Rewrite a function so that any if/else branches are intercepted and their behavior can be overridden. This is
    accomplished using 3 steps:

    1. Get the source of the function and then rewrite the AST using _IfTransformer
    2. Do some small fixups to the tree to make sure
        a) the function doesn't have the same name, and
        b) the decorator isn't called recursively on the transformed function as well
    3. Bring the variables from the call site back into scope

    :param f: Function to rewrite
    :return: Rewritten function
    """
    source = inspect.getsource(f)
    tree = ast.parse(source)
    _IfTransformer().visit(tree)

    ast.fix_missing_locations(tree)
    tree.body[0].name = f.__name__ + '_patched'
    tree.body[0].decorator_list = []

    compiled = compile(tree, filename='<ast>', mode='exec')
    # The first f_back here gets to the body of magicquil() and the second f_back gets to the user's call site which
    # is what we want. If we didn't add these manually to the globals it wouldn't be possible to call other @magicquil
    # functions from within a @magicquil function.
    prev_globals = inspect.currentframe().f_back.f_back.f_globals
    # For reasons I don't quite understand it's critical to add locals() here otherwise the function will disappear and
    # we won't be able to return it below
    exec(compiled, {**prev_globals, **globals()}, locals())
    return locals()[f.__name__ + '_patched']


def magicquil(f):
    """
    Decorator to enable a more convenient syntax for writing quil programs. With this decorator there is no need to
    keep track of a Program object and regular Python if/else branches can be used for classical control flow.

    Example usage:

        @magicquil
        def fast_reset(q1):
            reg1 = MEASURE(q1)
            if reg1:
                X(q1)
            else:
                I(q1)

        my_program = fast_reset(0)  # this will be a Program object
    """
    rewritten_function = _rewrite_function(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if _program_context.get(None) is not None:
            rewritten_function(*args, **kwargs)
            program = _program_context.get()
        else:
            token = _program_context.set(Program())
            rewritten_function(*args, **kwargs)
            program = _program_context.get()
            _program_context.reset(token)
        return program

    return wrapper
