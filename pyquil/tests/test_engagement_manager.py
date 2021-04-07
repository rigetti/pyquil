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

from pytest_httpx import HTTPXMock
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials

from pyquil.api._engagement_manager import EngagementManager
from pyquil.api import QCSClientConfiguration


def test_get_engagement__refreshes_engagement_when_cached_engagement_expired(
    client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock
):
    engagement_manager = EngagementManager(client_configuration=client_configuration)
    cache_engagement(
        engagement_manager, expired_engagement(quantum_processor_id="some-processor"), client_configuration, httpx_mock
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{client_configuration.profile.api_url}/v1/engagements",
        match_content=json.dumps({"quantumProcessorId": "some-processor"}).encode(),
        json=unexpired_engagement(quantum_processor_id="some-processor").to_dict(),
    )

    engagement = engagement_manager.get_engagement(quantum_processor_id="some-processor")

    assert engagement == unexpired_engagement(quantum_processor_id="some-processor")


def test_get_engagement__reuses_engagement_when_cached_engagement_unexpired(
    client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock
):
    engagement_manager = EngagementManager(client_configuration=client_configuration)
    cached_engagement = cache_engagement(
        engagement_manager,
        unexpired_engagement(quantum_processor_id="some-processor"),
        client_configuration,
        httpx_mock,
    )
    network_calls_before = len(httpx_mock.get_requests())

    engagement = engagement_manager.get_engagement(quantum_processor_id="some-processor")
    network_calls_after = len(httpx_mock.get_requests())

    assert network_calls_before == network_calls_after
    assert engagement is cached_engagement


def cache_engagement(
    engagement_manager: EngagementManager,
    engagement: EngagementWithCredentials,
    client_configuration: QCSClientConfiguration,
    httpx_mock: HTTPXMock,
) -> EngagementWithCredentials:
    httpx_mock.add_response(
        method="POST",
        url=f"{client_configuration.profile.api_url}/v1/engagements",
        match_content=json.dumps({"quantumProcessorId": engagement.quantum_processor_id}).encode(),
        json=engagement.to_dict(),
    )

    cached_engagement = engagement_manager.get_engagement(quantum_processor_id=engagement.quantum_processor_id)

    assert cached_engagement == engagement
    return cached_engagement


def make_engagement(*, quantum_processor_id: str, expires_at: str) -> EngagementWithCredentials:
    return EngagementWithCredentials(
        address="tcp://example.com/qpu",
        credentials=EngagementCredentials(
            client_public="client-public-123",
            client_secret="client-secret-123",
            server_public="server-public-123",
        ),
        endpoint_id="some-endpoint",
        expires_at=expires_at,
        quantum_processor_id=quantum_processor_id,
        user_id="some-user",
        minimum_priority=42,
    )


def unexpired_engagement(*, quantum_processor_id: str) -> EngagementWithCredentials:
    return make_engagement(quantum_processor_id=quantum_processor_id, expires_at="9999-01-01T00:00:00Z")


def expired_engagement(*, quantum_processor_id: str) -> EngagementWithCredentials:
    return make_engagement(quantum_processor_id=quantum_processor_id, expires_at="1970-01-01T00:00:00Z")
