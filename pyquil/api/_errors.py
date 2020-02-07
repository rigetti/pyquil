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
    if logger.level <= logging.DEBUG or exception_type is not UserMessageError:
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


class CancellationError(ApiError):
    def __init__(self, server_status: str):
        explanation = "Please try resubmitting the job again."
        super(CancellationError, self).__init__(server_status, explanation)


class DeviceOfflineError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The device you requested is offline. Use the following code to check for the
currently available devices:

    from pyquil.api import get_devices
    print(get_devices())"""
        super(DeviceOfflineError, self).__init__(server_status, explanation)


class DeviceRetuningError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The device you requested is temporarily down for retuning. Use the following
code to check for the currently available devices:

    from pyquil.api import get_devices
    print(get_devices())"""
        super(DeviceRetuningError, self).__init__(server_status, explanation)


class InvalidInputError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The server returned the above error because something was wrong with the HTTP
request sent to it. This could be due to a bug in the server or a bug in your
code. If you suspect this to be a bug in pyQuil or Rigetti Forest, then please
describe the problem in a GitHub issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(InvalidInputError, self).__init__(server_status, explanation)


class InvalidUserError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
There was an issue validating your Forest account!
Have you run the `pyquil-config-setup` command yet?

If you do not yet have a Forest account then sign up for one at:
    https://forest.rigetti.com"""
        super(InvalidUserError, self).__init__(server_status, explanation)


class JobNotFoundError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The above job may have been deleted manually or due to some bug in the server.
If you suspect this to be a bug then please describe the problem in a Github
issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(JobNotFoundError, self).__init__(server_status, explanation)


class MissingPermissionsError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
Your account may not be whitelisted for QPU access. To request the appropriate
permissions please read the information located at:
    https://forest.rigetti.com"""
        super(MissingPermissionsError, self).__init__(server_status, explanation)


class QPUError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The QPU returned the above error. This could be due to a bug in the server or a
bug in your code. If you suspect this to be a bug in pyQuil or Rigetti Forest,
then please describe the problem in a GitHub issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(QPUError, self).__init__(server_status, explanation)


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
You requested too many qubits on the QVM. More qubits are available when you use
the queue. Pass the use_queue parameter to QVMConnection to enable additional
qubits (however, each program will take longer to run). For example:

    qvm = QVMConnection(use_queue=True)
    qvm.run(twenty_qubit_program)

See https://go.rigetti.com/connections for more info."""
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
            logger.error(message)


class UnknownApiError(ApiError):
    def __init__(self, server_status: str):
        explanation = """
The server has failed to return a proper response. Please describe the problem
and copy the above message into a GitHub issue at:
    https://github.com/rigetti/pyquil/issues"""
        super(UnknownApiError, self).__init__(server_status, explanation)


# NB: Some errors are not included here if they are only returned by async endpoints
# The source of truth for this mapping is the _errors.py file on the server
error_mapping = {
    "device_offline": DeviceOfflineError,
    "device_retuning": DeviceRetuningError,
    "invalid_input": InvalidInputError,
    "invalid_user": InvalidUserError,
    "job_not_found": JobNotFoundError,
    "missing_permissions": MissingPermissionsError,
    "quilc_error": QUILCError,
    "qvm_error": QVMError,
}
