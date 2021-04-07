##############################################################################
# Copyright 2016-2021 Rigetti Computing
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
from contextlib import contextmanager
from typing import Iterator

import httpx
from qcs_api_client.client import QCSClientConfiguration, build_sync_client


@contextmanager
def qcs_client(*, client_configuration: QCSClientConfiguration, request_timeout: float = 5.0) -> Iterator[httpx.Client]:
    """
    Build a QCS client.

    :param client_configuration: Client configuration.
    :param request_timeout: Time limit for requests, in seconds.
    """
    with build_sync_client(
        configuration=client_configuration, client_kwargs={"timeout": request_timeout}
    ) as client:  # type: httpx.Client
        yield client
