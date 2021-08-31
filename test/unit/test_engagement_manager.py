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
import json
from typing import List, Optional

import httpx
import respx
from pyquil.api import QCSClientConfiguration
from pyquil.api._engagement_manager import EngagementManager
from qcs_api_client.models import EngagementCredentials, EngagementWithCredentials

DEFAULT_ENDPOINT_ID = "some-endpoint"

@respx.mock
def test_get_engagement__refreshes_engagement_when_cached_engagement_expired(
    client_configuration: QCSClientConfiguration,
):
    engagement_manager = EngagementManager(client_configuration=client_configuration)
    cache_engagement(
        engagement_manager, expired_engagement(quantum_processor_id="some-processor"), client_configuration,
    )
    respx.post(
        url=f"{client_configuration.profile.api_url}/v1/engagements",
        json={"quantumProcessorId": "some-processor"},
    ).respond(json=unexpired_engagement(quantum_processor_id="some-processor").to_dict())

    engagement = engagement_manager.get_engagement(quantum_processor_id="some-processor")

    assert engagement == unexpired_engagement(quantum_processor_id="some-processor")


@respx.mock
def test_get_engagement__refreshes_engagement_when_cached_engagement_expired__using_endpoint_id(
    client_configuration: QCSClientConfiguration,
):
    """
    Assert that endpoint ID is correctly used to engage against an endpoint when the cached engagement has expired.
    """
    engagement_manager = EngagementManager(client_configuration=client_configuration)
    cache_engagement(
        engagement_manager,
        expired_engagement(quantum_processor_id="some-processor", endpoint_id="custom-endpoint"),
        client_configuration,
    )
    respx.post(
        url=f"{client_configuration.profile.api_url}/v1/engagements",
        json={"quantumProcessorId": "some-processor", "endpointId": "custom-endpoint"},
    ).respond(json=unexpired_engagement(quantum_processor_id="some-processor").to_dict())

    engagement = engagement_manager.get_engagement(quantum_processor_id="some-processor")

    assert engagement == unexpired_engagement(quantum_processor_id="some-processor")


@respx.mock
def test_get_engagement__reuses_engagement_when_cached_engagement_unexpired(
    client_configuration: QCSClientConfiguration,
):
    engagement_manager = EngagementManager(client_configuration=client_configuration)
    cached_engagement = cache_engagement(
        engagement_manager,
        unexpired_engagement(quantum_processor_id="some-processor"),
        client_configuration,
    )
    network_calls_before = respx.calls.call_count

    engagement = engagement_manager.get_engagement(quantum_processor_id="some-processor")
    network_calls_after = respx.calls.call_count

    assert network_calls_before == network_calls_after
    assert engagement is cached_engagement


@respx.mock
def test_get_engagement__reuses_engagement_when_cached_engagement_unexpired__using_endpoint_id(
    client_configuration: QCSClientConfiguration,
):
    """
    Assert that endpoint ID is correctly used to engage against an endpoint.
    """
    engagement_manager = EngagementManager(client_configuration=client_configuration)
    cached_engagement = cache_engagement(
        engagement_manager,
        unexpired_engagement(quantum_processor_id="some-processor", endpoint_id="custom-endpoint"),
        client_configuration,
    )
    network_calls_before = respx.calls.call_count

    engagement = engagement_manager.get_engagement(quantum_processor_id="some-processor", endpoint_id="custom-endpoint")
    network_calls_after = respx.calls.call_count

    assert network_calls_before == network_calls_after
    assert engagement is cached_engagement


def cache_engagement(
    engagement_manager: EngagementManager,
    engagement: EngagementWithCredentials,
    client_configuration: QCSClientConfiguration,
) -> EngagementWithCredentials:
    mock_engagement(client_configuration=client_configuration, engagement=engagement)

    if engagement.endpoint_id == DEFAULT_ENDPOINT_ID:
        endpoint_id = None
    else:
        endpoint_id = engagement.endpoint_id

    cached_engagement = engagement_manager.get_engagement(
        quantum_processor_id=engagement.quantum_processor_id, endpoint_id=endpoint_id
    )

    assert cached_engagement == engagement
    return cached_engagement


def mock_engagement(engagement: EngagementWithCredentials, *, client_configuration: QCSClientConfiguration):
    """
    Apply and respond with an engagement when it matches.
    """

    if engagement.endpoint_id == DEFAULT_ENDPOINT_ID:
        respx.post(
            url=f"{client_configuration.profile.api_url}/v1/engagements",
            json={"quantumProcessorId": engagement.quantum_processor_id}
        ).respond(json=engagement.to_dict())

    respx.post(
        url=f"{client_configuration.profile.api_url}/v1/engagements",
        json={"endpointId": engagement.endpoint_id}
    ).respond(json=engagement.to_dict())


def make_engagement(
    *, quantum_processor_id: str, endpoint_id: Optional[str] = None, expires_at: str
) -> EngagementWithCredentials:
    return EngagementWithCredentials(
        address="tcp://example.com/qpu",
        credentials=EngagementCredentials(
            client_public="client-public-123",
            client_secret="client-secret-123",
            server_public="server-public-123",
        ),
        endpoint_id=endpoint_id or DEFAULT_ENDPOINT_ID,
        expires_at=expires_at,
        quantum_processor_id=quantum_processor_id,
        user_id="some-user",
        minimum_priority=42,
    )


def unexpired_engagement(*, quantum_processor_id: str, endpoint_id: Optional[str] = None) -> EngagementWithCredentials:
    return make_engagement(
        quantum_processor_id=quantum_processor_id, endpoint_id=endpoint_id, expires_at="9999-01-01T00:00:00Z"
    )


def expired_engagement(*, quantum_processor_id: str, endpoint_id: Optional[str] = None) -> EngagementWithCredentials:
    return make_engagement(
        quantum_processor_id=quantum_processor_id, endpoint_id=endpoint_id, expires_at="1970-01-01T00:00:00Z"
    )
