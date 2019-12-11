##############################################################################
# Copyright 2016-2019 Rigetti Computing
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
import os
import sys

logging.basicConfig(format='%(levelname)s - %(message)s',)

level = os.getenv('LOG_LEVEL', 'INFO').upper()

logger = logging.getLogger(__name__)
logger.setLevel(level)

logger.debug(f"Log level: {logger.level}")


def exception_handler(exception_type, exception, traceback, debug_hook=sys.excepthook):
    """
    This allows us to suppress tracebacks for UserMessageError outside of debug mode
      by overriding the default exception handler.
    """
    if logger.level <= logging.DEBUG or exception_type is not UserMessageError:
        debug_hook(exception_type, exception, traceback)


sys.excepthook = exception_handler


class UserMessageError(Exception):
    """
    A special class of error which only displays its traceback when the program
      is run in debug mode (eg, with `LOG_LEVEL=DEBUG`).

    The purpose of this is to improve the user experience, reducing noise in
      the case of errors for which the cause is known.
    """
    def __init__(self, message):
        if logger.level <= logging.DEBUG:
            super().__init__(message)
        else:
            logger.error(message)
