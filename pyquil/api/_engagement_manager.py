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
import threading
from datetime import datetime
from typing import Dict, NamedTuple, Optional, TYPE_CHECKING

from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import EngagementWithCredentials, CreateEngagementRequest
from qcs_api_client.operations.sync import create_engagement
from qcs_api_client.types import UNSET
from qcs_api_client.util.errors import QCSHTTPStatusError

from pyquil.api._qcs_client import qcs_client

if TYPE_CHECKING:
    import httpx


class QPUUnavailableError(Exception):
    """
    Exception raised when a QPU is unavailable.
    """

    retry_after: Optional[int]
    """The number of seconds after which to retry the engagement request."""

    def __init__(self, retry_after: Optional[str]) -> None:
        if retry_after is not None:
            super().__init__(f"QPU unavailable. Please retry after {retry_after}s.")
        else:
            super().__init__("QPU unavailable.")


class EngagementCacheKey(NamedTuple):
    quantum_processor_id: str
    endpoint_id: Optional[str]


class EngagementManager:
    """
    Fetches (and caches) engagements for use when accessing a QPU.
    """

    _lock: threading.Lock
    """Lock used to ensure that only one engagement request is in flight at once."""

    def __init__(self, *, client_configuration: QCSClientConfiguration) -> None:
        """
        Instantiate a new engagement manager.

        :param client_configuration: Client configuration, used for refreshing engagements.
        """
        self._client_configuration = client_configuration
        self._cached_engagements: Dict[EngagementCacheKey, EngagementWithCredentials] = {}
        self._lock = threading.Lock()

    def get_engagement(
        self, *, quantum_processor_id: str, request_timeout: float = 10.0, endpoint_id: Optional[str] = None
    ) -> EngagementWithCredentials:
        """
        Gets an engagement for the given quantum processor endpoint.

        If an engagement was already fetched previously and remains valid, it will be returned instead
        of creating a new engagement.

        :param quantum_processor_id: Quantum processor being engaged.
        :param request_timeout: Timeout for request, in seconds.
        :param endpoint_id: Optional ID of the endpoint to use for engagement. If provided, it must
            correspond to an endpoint serving the provided Quantum Processor.
        :return: Fetched or cached engagement.
        :raises QPUUnavailableError: raised when the QPU is unavailable due, and provides a suggested
            number of seconds to wait until retrying.
        :raises QCSHTTPStatusError: raised when creating an engagement fails for a reason that is not
            due to QPU unavailability.
        """
        key = EngagementCacheKey(quantum_processor_id, endpoint_id)

        with self._lock:
            if not self._engagement_valid(self._cached_engagements.get(key)):
                with qcs_client(
                    client_configuration=self._client_configuration, request_timeout=request_timeout
                ) as client:  # type: httpx.Client
                    request = CreateEngagementRequest(
                        quantum_processor_id=quantum_processor_id, endpoint_id=endpoint_id or UNSET
                    )
                    response = create_engagement(client=client, json_body=request)
                    try:
                        self._cached_engagements[key] = response.parsed
                    except QCSHTTPStatusError as e:
                        if response.status_code == 503:
                            raise QPUUnavailableError(retry_after=response.headers.get("Retry-After")) from e
                        raise e
            return self._cached_engagements[key]

    @staticmethod
    def _engagement_valid(engagement: Optional[EngagementWithCredentials]) -> bool:
        if engagement is None:
            return False

        return all(
            [
                engagement.credentials.client_public != "",
                engagement.credentials.client_secret != "",
                engagement.credentials.server_public != "",
                parsedate(engagement.expires_at) > datetime.now(tzutc()),
                engagement.address != "",
            ]
        )
