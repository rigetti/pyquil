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

from test.unit.utils import patch_rpcq_client

try:
    from unittest.mock import AsyncMock
except ImportError:  # 3.7 requires this backport of AsyncMock
    from mock import AsyncMock

from qcs_sdk import QCSClient
from _pytest.monkeypatch import MonkeyPatch
import pytest
from pytest import raises
from pytest_mock import MockerFixture

from pyquil.api._compiler_client import (
    CompilerClient,
    CompileToNativeQuilRequest,
    CompileToNativeQuilResponse,
)
from pyquil.external.rpcq import CompilerISA, compiler_isa_to_target_quantum_processor


def test_init__sets_base_url_and_timeout(monkeypatch: MonkeyPatch):
    host = "tcp://localhost:1234"
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_QUILC_URL", host)
    client_configuration = QCSClient.load()

    compiler_client = CompilerClient(client_configuration=client_configuration, request_timeout=3.14)

    assert compiler_client.base_url == host
    assert compiler_client.timeout == 3.14


def test_init__validates_compiler_url(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_QUILC_URL", "not-http-or-tcp://example.com")
    client_configuration = QCSClient.load()

    with raises(
        ValueError,
        match="Expected compiler URL 'not-http-or-tcp://example.com' to start with 'tcp://'",
    ):
        CompilerClient(client_configuration=client_configuration)


def test_sets_timeout_on_requests(mocker: MockerFixture):
    client_configuration = QCSClient.load()
    compiler_client = CompilerClient(client_configuration=client_configuration, request_timeout=0.1)

    patch_rpcq_client(mocker=mocker, return_value={})

    with compiler_client._rpcq_client() as client:
        assert client.timeout == compiler_client.timeout


@pytest.mark.skip  # cannot mock `qcs_sdk` here
def test_get_version__returns_version(mocker: MockerFixture):
    client_configuration = QCSClient.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    version_mock = AsyncMock(return_value="1.2.3")
    get_quilc_version_mock = mocker.patch("qcs_sdk.get_quilc_version", version_mock)

    assert compiler_client.get_version() == "1.2.3"
    assert get_quilc_version_mock.called


def test_compile_to_native_quil__returns_native_quil(
    aspen8_compiler_isa: CompilerISA,
    mocker: MockerFixture,
):
    client_configuration = QCSClient.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    request = CompileToNativeQuilRequest(
        program="DECLARE ro BIT",
        target_quantum_processor=compiler_isa_to_target_quantum_processor(aspen8_compiler_isa),
        protoquil=True,
    )

    assert compiler_client.compile_to_native_quil(request) == CompileToNativeQuilResponse(
        native_program="DECLARE ro BIT[1]\n",
        metadata=None,
    )
