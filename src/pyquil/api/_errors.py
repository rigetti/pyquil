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

import logging
import sys
from types import TracebackType
from typing import Callable, Type

from pyquil.api._logger import logger


def exception_handler(
    exception_type: Type[BaseException],
    exception: BaseException,
    traceback: TracebackType,
    debug_hook: Callable[..., None] = sys.excepthook,
) -> None:
    """
    This allows us to suppress tracebacks for UserMessageError outside of debug mode
      by overriding the default exception handler.
    """
    if logger.level > logging.DEBUG and exception_type is UserMessageError:
        exception.__traceback__ = None
    debug_hook(exception_type, exception, traceback)


sys.excepthook = exception_handler  # type: ignore


class ApiError(RuntimeError):
    def __init__(self, server_status: str, explanation: str):
        super(ApiError, self).__init__(self, server_status)
        self.server_status = server_status
        self.explanation = explanation

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        return "{}\n{}".format(self.server_status, self.explanation)


class QVMError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The QVM returned the above error. This could be due to a bug in the server or a
bug in your code. If you suspect this to be a bug in pyQuil or Rigetti Forest,
then please describe the problem in a GitHub issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(QVMError, self).__init__(server_status, explanation)


class QUILCError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
QUILC returned the above error. This could be due to a bug in the server or a
bug in your code. If you suspect this to be a bug in pyQuil or Rigetti Forest,
then please describe the problem in a GitHub issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(QUILCError, self).__init__(server_status, explanation)


class TooManyQubitsError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
You requested too many qubits on the QVM."""
        super(TooManyQubitsError, self).__init__(server_status, explanation)


class UserMessageError(Exception):
    """
    A special class of error which only displays its traceback when the program
      is run in debug mode (eg, with `LOG_LEVEL=DEBUG`).

    The purpose of this is to improve the user experience, reducing noise in
      the case of errors for which the cause is known.
    """

    def __init__(self, message: str):
        if logger.level <= logging.DEBUG:
            super().__init__(message)
        else:
            self.message = message

    def __str__(self) -> str:
        if logger.level <= logging.DEBUG:
            return super(UserMessageError, self).__str__()
        else:
            return f"ERROR: {self.message}"

    def __repr__(self) -> str:
        if logger.level <= logging.DEBUG:
            return super(UserMessageError, self).__repr__()
        else:
            return f"UserMessageError: {str(self)}"


class UnknownApiError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The server has failed to return a proper response. Please describe the problem
and copy the above message into a GitHub issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(UnknownApiError, self).__init__(server_status, explanation)


error_mapping = {
    "quilc_error": QUILCError,
    "qvm_error": QVMError,
}
