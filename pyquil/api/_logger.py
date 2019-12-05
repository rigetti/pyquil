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

import logging
import os
import sys

logging.basicConfig(format='%(levelname)s - %(message)s',)

level = os.getenv('LOG_LEVEL', 'INFO').upper()

logger = logging.getLogger(__name__)
logger.setLevel(level)


class UserMessageError(Exception):
    """
    A special class of error which only displays its traceback when the program
      is run in debug mode (eg, with `LOG_LEVEL=DEBUG`).

    The purpose of this is to improve the user experience, reducing noise in
      the case of errors for which the cause is known
    """
    def __init__(self, message):
        if logger.level <= logging.DEBUG:
            super().__init__(message)
        else:
            logger.error(message)
            sys.exit(1)
