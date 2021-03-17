import re
from contextlib import contextmanager
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Optional, Any, Callable, Iterator, cast

import httpx
import rpcq
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.client import QCSClientConfiguration, build_sync_client
from qcs_api_client.models import (
    EngagementWithCredentials,
    CreateEngagementRequest,
    EngagementCredentials,
)
from qcs_api_client.operations.sync import create_engagement
from qcs_api_client.types import Response
from rpcq import ClientAuthConfig

from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
from pyquil.api._logger import logger


class Client:
    """
    Class for housing application configuration and interacting with network resources.
    """

    _http: Optional[httpx.Client]
    _config: QCSClientConfiguration
    _has_default_config: bool
    _engagement: Optional[EngagementWithCredentials] = None

    def __init__(
        self,
        *,
        http: Optional[httpx.Client] = None,
        configuration: Optional[QCSClientConfiguration] = None,
    ):
        """
        Instantiate a client.

        :param http: optional underlying HTTP client. If none is provided, a default
               client will be created using ``configuration``.
        :param configuration: QCS configuration.
        """
        self._config = configuration or QCSClientConfiguration.load()
        self._has_default_config = configuration is None
        self._http = http

    def post_json(self, url: str, json: Any, timeout: float = 5) -> httpx.Response:
        """
        Post JSON to a URL. Will raise an exception for response statuses >= 400.

        :param url: URL to post to.
        :param json: JSON body of request.
        :param timeout: Time limit for request, in seconds.
        :return: HTTP response corresponding to request.
        """
        logger.debug("Sending POST request to %s. Body: %s", url, json)
        with self._http_client() as http:  # type: httpx.Client
            res = http.post(url, json=json, timeout=timeout)
            if res.status_code >= 400:
                raise _parse_error(res)
        return res

    def qcs_request(self, request_fn: Callable[..., Response[Any]], **kwargs: Any) -> Any:
        """
        Execute a QCS request.

        :param request_fn: Request function (from ``qcs_api_client.operations.sync``).
        :param kwargs: Arguments to pass to request function.
        :return: HTTP response corresponding to request.
        """
        with self._http_client() as http:  # type: httpx.Client
            return request_fn(client=http, **kwargs).parsed

    def compiler_rpcq_request(
        self, method_name: str, *args: Any, timeout: Optional[float] = None, **kwargs: Any
    ) -> Any:
        """
        Execute a remote function against the Quil compiler.

        :param method_name: Method name.
        :param args: Arguments that will be passed to the remote function.
        :param timeout: Optional time limit for request, in seconds.
        :param kwargs: Keyword arguments that will be passed to the remote function.
        :return: Result from remote function.
        """
        return self._rpcq_request(self.quilc_url, method_name, *args, timeout=timeout, **kwargs)

    def processor_rpcq_request(
        self,
        quantum_processor_id: str,
        method_name: str,
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a remote function against a processor endpoint. If there is no current engagement,
        or if it is invalid, a new engagement will be requested for the given processor.

        :param quantum_processor_id: Processor to engage.
        :param method_name: Method name.
        :param args: Arguments that will be passed to the remote function.
        :param timeout: Optional time limit for request, in seconds.
        :param kwargs: Keyword arguments that will be passed to the remote function.
        :return: Result from remote function.
        """
        # TODO(andrew): handle multiple engagements at once (per processor)
        if not _engagement_valid(quantum_processor_id, self._engagement):
            self._engagement = self._create_engagement(quantum_processor_id)

        assert self._engagement is not None

        return self._rpcq_request(
            self._engagement.address,
            method_name,
            *args,
            timeout=timeout,
            auth_config=_to_auth_config(self._engagement.credentials),
            **kwargs,
        )

    def _rpcq_request(
        self,
        endpoint: str,
        method_name: str,
        *args: Any,
        timeout: Optional[float] = None,
        auth_config: Optional[ClientAuthConfig] = None,
        **kwargs: Any,
    ) -> Any:
        client = rpcq.Client(endpoint, auth_config=auth_config)
        response = client.call(method_name, *args, rpc_timeout=timeout, **kwargs)  # type: ignore
        client.close()  # type: ignore
        return response

    def reset(self) -> None:
        """
        Clears current engagement and reloads configuration (if not overridden in constructor).
        """
        self._engagement = None
        if self._has_default_config:
            self._config = QCSClientConfiguration.load()

    @property
    def qvm_url(self) -> str:
        """
        QVM URL from client configuration.
        """
        return self._config.profile.applications.pyquil.qvm_url

    @property
    def quilc_url(self) -> str:
        """
        Quil compiler URL from client configuration.
        """
        return self._config.profile.applications.pyquil.quilc_url

    def qvm_version(self) -> str:
        """
        Get QVM version string.
        """
        response = self.post_json(self.qvm_url, {"type": "version"})
        split_version_string = response.text.split()
        try:
            qvm_version = split_version_string[0]
        except ValueError:
            raise TypeError(f"Malformed version string returned by the QVM: {response.text}")
        return qvm_version

    @contextmanager
    def _http_client(self) -> Iterator[httpx.Client]:
        if self._http is None:
            with build_sync_client(configuration=self._config) as client:  # type: httpx.Client
                yield client
        else:
            yield self._http

    def _create_engagement(self, quantum_processor_id: str) -> EngagementWithCredentials:
        return cast(
            EngagementWithCredentials,
            self.qcs_request(
                create_engagement,
                json_body=CreateEngagementRequest(quantum_processor_id=quantum_processor_id),
            ),
        )


def _engagement_valid(quantum_processor_id: str, engagement: Optional[EngagementWithCredentials]) -> bool:
    if engagement is None:
        return False

    return all(
        [
            engagement.credentials.client_public != "",
            engagement.credentials.client_secret != "",
            engagement.credentials.server_public != "",
            parsedate(engagement.expires_at) > datetime.now(tzutc()),
            engagement.address != "",
            engagement.quantum_processor_id == quantum_processor_id,
        ]
    )


def _to_auth_config(credentials: EngagementCredentials) -> ClientAuthConfig:
    return rpcq.ClientAuthConfig(
        client_secret_key=credentials.client_secret.encode(),
        client_public_key=credentials.client_public.encode(),
        server_public_key=credentials.server_public.encode(),
    )


def _parse_error(res: httpx.Response) -> ApiError:
    """
    Errors should contain a "status" field with a human readable explanation of
    what went wrong as well as a "error_type" field indicating the kind of error that can be mapped
    to a Python type.

    There's a fallback error UnknownApiError for other types of exceptions (network issues, api
    gateway problems, etc.)
    """
    try:
        body = res.json()
    except JSONDecodeError:
        raise UnknownApiError(res.text)

    if "error_type" not in body:
        raise UnknownApiError(str(body))

    error_type = body["error_type"]
    status = body["status"]

    if re.search(r"[0-9]+ qubits were requested, but the QVM is limited to [0-9]+ qubits.", status):
        return TooManyQubitsError(status)

    error_cls = error_mapping.get(error_type, UnknownApiError)
    return error_cls(status)
