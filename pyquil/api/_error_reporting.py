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
"""
Module for automatically generating error reports helpful for diagnosing pyQuil errors.

IMPORTANT NOTE: THIS MODULE USES GLOBAL STATE AND IS NOT ESPECIALLY THREAD-SAFE.
                If your threaded code is experiencing pyQuil errors, you'll have to track
                your own state and not use this convenient decorator.
"""
import inspect
import json
import logging
import os
import sys
from datetime import datetime, date
from functools import wraps
from typing import List, Dict, Any, Callable, Optional

from pyquil.quil import Program
from pyquil.version import __version__

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass, is_dataclass, asdict
else:
    from dataclasses import dataclass, is_dataclass, asdict

_log = logging.getLogger(__name__)


@dataclass
class ErrorReport:
    """
    Dump of the current state of a pyQuil program.
    """

    stack_trace: List["StacktraceFrame"]
    timestamp: date
    call_log: Dict[str, "CallLogValue"]
    exception: Exception  # noqa: E701
    system_info: Dict[str, str]


@dataclass
class StacktraceFrame:
    """
    Expanded frame in a stacktrace, suitable for JSON export.
    """

    name: str
    filename: str
    line_number: int
    locals: Dict[str, str]


@dataclass(eq=True, frozen=True)
class CallLogKey:
    """
    Entry in the call log list, suitable for JSON export.
    """

    name: str
    args: List[str]
    kwargs: Dict[str, Any]

    def __hash__(self) -> int:
        finger_print = (
            (self.name,) + tuple(self.args) + tuple(sorted(self.kwargs.items(), key=lambda i: i[0]))
        )
        return hash(finger_print)

    def __repr__(self) -> str:
        ret = self.name + "("
        for item in self.args:
            ret += repr(item) + ", "
        for k, v in self.kwargs.items():
            ret += k + "=" + repr(v) + ", "
        ret += ")"

        return ret


@dataclass
class CallLogValue:
    """
    Entry in the call log list, suitable for JSON export.
    """

    timestamp_in: date
    timestamp_out: Optional[date]
    return_value: Optional[str]


def json_serialization_helper(o: object) -> Any:
    if is_dataclass(o):
        return asdict(o)
    elif isinstance(o, datetime):
        return o.isoformat()
    elif isinstance(o, Exception):
        return repr(o)
    else:
        raise TypeError("unable to serialize object {}".format(o))


def generate_system_info() -> Dict[str, str]:
    system_info = {"python_version": sys.version, "pyquil_version": __version__}

    return system_info


def serialize_object_for_logging(o: object) -> str:
    if isinstance(o, Program):
        return str(o)
    else:
        return repr(o)


def flatten_log(log: Dict[CallLogKey, CallLogValue]) -> Dict[str, CallLogValue]:
    return {repr(k): v for k, v in log.items()}


class ErrorContext(object):
    """
    Tracks information relevant to error reporting.
    """

    log: Dict[CallLogKey, CallLogValue] = {}
    filename = "pyquil_error.log"

    def generate_report(self, exception: Exception, trace: List[inspect.FrameInfo]) -> ErrorReport:
        """
        Handle an error generated in a routine decorated with the pyQuil error handler.

        :param exception: Exception object that generated this error.
        :param trace: inspect.trace object from the frame that caught the error.
        :return: ErrorReport object
        """
        stack_trace = [
            StacktraceFrame(
                name=item.function,
                filename=item.filename,
                line_number=item.lineno,
                locals={
                    k: serialize_object_for_logging(v) for (k, v) in item.frame.f_locals.items()
                },
            )
            for item in trace
        ]

        system_info = generate_system_info()

        report = ErrorReport(
            stack_trace=stack_trace,
            timestamp=datetime.utcnow(),
            exception=exception,
            system_info=system_info,
            call_log=flatten_log(self.log),
        )

        return report

    def dump_error(self, exception: Exception, trace: List[inspect.FrameInfo]) -> None:
        warn_msg = """
>>> PYQUIL_PROTECT <<<
An uncaught exception was raised in a function wrapped in pyquil_protect.  We are writing out a
log file to "{}".

Along with a description of what you were doing when the error occurred, send this file to
Rigetti Computing support by email at support@rigetti.com for assistance.
>>> PYQUIL_PROTECT <<<
""".format(
            os.path.abspath(self.filename)
        )

        _log.warning(warn_msg)

        report = self.generate_report(exception, trace)

        # overwrite any existing log file
        fh = open(self.filename, "w")
        fh.write(json.dumps(report, default=json_serialization_helper))
        fh.close()


global_error_context: Optional[ErrorContext] = None


def pyquil_protect(
    func: Callable[..., Any], log_filename: str = "pyquil_error.log"
) -> Callable[..., Any]:
    """
    A decorator that sets up an error context, captures errors, and tears down the context.
    """

    def pyquil_protect_wrapper(*args: Any, **kwargs: Any) -> Any:
        global global_error_context

        old_error_context = global_error_context
        global_error_context = ErrorContext()
        global_error_context.filename = log_filename

        try:
            val = func(*args, **kwargs)
            global_error_context = old_error_context
            return val
        except Exception as e:
            assert global_error_context is not None
            global_error_context.dump_error(e, inspect.trace())
            global_error_context = old_error_context
            raise

    return pyquil_protect_wrapper


def _record_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that logs a call into the global error context.

    This is probably for internal use only.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global global_error_context

        # log a call as about to take place
        if global_error_context is not None:
            key = CallLogKey(
                name=func.__name__,
                args=[serialize_object_for_logging(arg) for arg in args],
                kwargs={k: serialize_object_for_logging(v) for k, v in kwargs.items()},
            )

            pre_entry = CallLogValue(
                timestamp_in=datetime.utcnow(), timestamp_out=None, return_value=None
            )
            global_error_context.log[key] = pre_entry

        val = func(*args, **kwargs)

        # poke the return value of that call in
        if global_error_context is not None:
            post_entry = CallLogValue(
                timestamp_in=pre_entry.timestamp_in,
                timestamp_out=datetime.utcnow(),
                return_value=serialize_object_for_logging(val),
            )
            global_error_context.log[key] = post_entry

        return val

    return wrapper
