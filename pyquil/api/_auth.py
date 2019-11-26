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

import json
import re
import time
import requests
from os.path import exists, expanduser, isfile, join

from pyquil.api._logger import logger, UserMessageError


class EngagementFailedError(UserMessageError):
    pass


class AuthClient:
    """
    Encapsulates the operations required for authorization & encryption
      with the QPU.

    Two operations are involved in authorization:
      * Requesting & storing a user authentication token, used to authenticate calls
        to Dispatch and other Rigetti services
      * Requesting a Curve ZeroMQ keypair for connection to the QPU. The response to
        this request also comes with service endpoints: compiler server and QPU

    The authentication tokens are of the standard JWT format and are issued by Forest Server.
      The refresh token is only used to renew the access token, which is used for all transactions
      and is valid for a short period of time.
    """
    def __init__(self, config):
        self.config = config
        self.client_public_key = None
        self.client_secret_key = None
        self.server_public_key = None
        self._engagement = None
        self._auth_tokens = None

    @property
    def auth_token_refresh_endpoint(self):
        """
        The endpoint to which PyQuil can send a long-lived refresh token 
          in order to get a new, valid access token
        """
        return join(self.config.forest_url, 'auth/idp/oauth2/v1/token')

    @property
    def auth_tokens(self):
        """
        Memoized authentication tokens; read from disk if not present
        """
        if not self._auth_tokens:
            self._auth_tokens = self.fetch_auth_tokens_from_disk()
        return self._auth_tokens

    @property
    def access_token(self):
        """
        The temporarily-valid access token granted by Forest Server
        """
        return self.auth_tokens.get('access_token')

    @property
    def refresh_token(self):
        """
        The long-lived authorization token used by PyQuil to refresh the access token
        """
        return self.auth_tokens.get('refresh_token')

    @property
    def engagement(self):
        """
        Returns memoized engagement information, if still valid - or requests a new engagement,
          and stores and returns that.
        """
        if not (self._engagement and self._engagement.is_valid()):
            self.engage()
        return self._engagement

    def fetch_auth_tokens_from_disk(self) -> dict:
        """
        Fetches the Rigetti API auth token from the configured location on disk
        """
        if isfile(self.config.auth_token_path):
            logger.debug(
                f"Loading auth tokens from {self.config.auth_token_path}")
            with open(self.config.auth_token_path, 'r') as f:
                tokens = json.load(f)
                return tokens
        return {}

    def write_auth_tokens_to_disk(self, metadata={}):
        """
        Write the auth tokens to the configured locations on disk
        """
        with open(self.config.auth_token_path, 'w') as f:
            logger.debug(
                f"Writing new auth tokens to {self.config.auth_token_path}")
            json.dump(self.auth_tokens, f)

    def refresh_auth_token(self):
        """
        Get a new auth token from the auth server
        """
        current_tokens = self.fetch_auth_tokens_from_disk()
        refresh_token = current_tokens.get('refresh_token')
        if not refresh_token:
            raise UserMessageError(
                f"""No refresh token available - visit {self.config.qcs_ui_url}/auth/token
                 to get one, and download it to {self.config.auth_token_path}"""
            )

        headers = {
            'accept': 'application/json',
            'cache-control': 'no-cache',
            'content-type': 'application/x-www-form-urlencoded',
        }

        body = {
            'grant_type': 'refresh_token',
            'scopes': ['offline_access', 'email', 'profile', 'openid'],
            'refresh_token': refresh_token,
            'redirect_uri': self.config.qcs_ui_url
        }

        current_time = time.time()
        logger.debug(
            f"Refreshing auth tokens from {self.auth_token_refresh_endpoint}")
        response = requests.post(self.auth_token_refresh_endpoint,
                                 data=body,
                                 headers=headers)

        if response.ok:
            auth_tokens = response.json()
            logger.debug(f"Received auth tokens in response: {auth_tokens}")
            expires_at = current_time + auth_tokens.get('expires_in', 0)
            self._auth_tokens = dict(**auth_tokens,
                                     **dict(expires_at=expires_at))
            self.write_auth_tokens_to_disk()
        else:
            raise UserMessageError(f"Failed to refresh token: {response.__dict__}")

    def engage(self, lattice_name):
        """
        The heart of the PyQuil authorization process, `engage` makes a request to
          the dispatch server for the information needed to communicate with the QPU.

        This is a standard GraphQL request, authenticated using the access token
          retrieved from Forest Server.

        The response includes the endpoints to the QPU and QPU Compiler Server, 
          along with the set of keys necessary to connect to the QPU and the time at
          which that key set expires.
        """
        query = '''
          mutation Engage($name: String!) {
            engage(input: { lattice: { name: $name }}) {
              success
              message
              engagement {
                type
                qpu {
                    endpoint
                    credentials {
                        clientPublic
                        clientSecret
                        serverPublic
                    }
                }
                compiler {
                    endpoint
                }
                expiresAt
              }
            }
          }
        '''
        logger.info(f"Requesting engagement from {self.config.dispatch_url}")
        variables = dict(name=lattice_name)
        query_response = self._send_graphql_with_auth(self.config.dispatch_url,
                                                      query, variables)
        logger.debug(
            f"Received response to engagement request: {query_response}")

        if query_response.get('errors'):
            raise UserMessageError(
                f"Failed to engage: {','.join(map(lambda error: error['message'], query_response.get('errors', [])))}"
            )

        if query_response is not None:
            engagement_response = query_response.get('data',
                                                     {}).get('engage', None)
            if engagement_response and engagement_response.get(
                    'success') is True:
                logger.info(f"Engagement successful")
                engagement_data = engagement_response.get('engagement', {})
                self._engagement = Engagement(
                    client_secret_key=engagement_data.get('qpu', {}).get(
                        'credentials', {}).get('clientSecret',
                                               '').encode('utf-8'),
                    client_public_key=engagement_data.get('qpu', {}).get(
                        'credentials', {}).get('clientPublic',
                                               '').encode('utf-8'),
                    server_public_key=engagement_data.get('qpu', {}).get(
                        'credentials', {}).get('serverPublic',
                                               '').encode('utf-8'),
                    expires_at=engagement_data.get('expiresAt', {}),
                    qpu_endpoint=engagement_data.get('qpu',
                                                     {}).get('endpoint'),
                    qpu_compiler_endpoint=engagement_data.get(
                        'compiler', {}).get('endpoint'))
            else:
                raise EngagementFailedError(
                    f"Unable to engage {lattice_name}: {engagement_response.get('message', 'No message')}"
                )

        return self._engagement

    def _send_graphql_with_auth(self, *args, **kwargs):
        """
        If the auth token is expired on the first attempt, refresh it and then
            perform a single retry using the new token.
        """
        if self.access_token is None:
            raise UserMessageError(f"""
                No access token found: this could be because you don't have a valid token at
                {self.config.auth_token_path}, or because there are problems with your account.
                """)
        headers = dict(authorization=f'Bearer {self.access_token}')
        auth_kwargs = dict(**kwargs, **{'headers': headers})

        response = self._send_graphql(*args, **auth_kwargs)
        if response.get('errors') and any([
                error.get('extensions', {}).get('code') == 'AUTH_TOKEN_EXPIRED'
                for error in response.get('errors')
        ]):
            self.refresh_auth_token()
            headers = dict(authorization=f'Bearer {self.access_token}')
            auth_kwargs = dict(**kwargs, **{'headers': headers})
            response = self._send_graphql(*args, **auth_kwargs)

        return response

    def _send_graphql(self, endpoint, query, variables, headers={}):
        """
        A simple GraphQL client
        """
        data = {'query': query, 'variables': variables}

        headers = dict(
            {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            }, **headers)

        logger.debug(f"Sending request to {endpoint}: {data}")
        response = requests.post(endpoint,
                                 data=json.dumps(data),
                                 headers=headers)
        try:
            result = response.json()
            logger.debug(f"Received response from {endpoint}: {result}")
            return result
        except Exception as e:
            logger.error(f"Unable to parse json response from endpoint {endpoint}:",
                         response.text)
            raise e


class Engagement:
    """
    An Engagement stores all the information retrieved via an engagement request sent to
      the dispatch server.
    """
    def __init__(self, client_public_key: bytes, client_secret_key: bytes,
                 server_public_key: bytes, expires_at, qpu_endpoint,
                 qpu_compiler_endpoint):
        self.client_public_key = client_public_key
        self.client_secret_key = client_secret_key
        self.server_public_key = server_public_key
        self.expires_at = float(expires_at) if expires_at else None
        self.qpu_endpoint = qpu_endpoint
        self.qpu_compiler_endpoint = qpu_compiler_endpoint
        logger.debug(f"New engagement created: {self}")

    def is_valid(self) -> bool:
        """
        Return true if an engagement is valid for use, false if it is missing required
          fields
        """
        return all([
            self.client_public_key is not None,
            self.client_secret_key is not None,
            self.server_public_key is not None,
            (self.expires_at is None
             or self.expires_at > time.time()), self.qpu_endpoint is not None
        ])

    def __str__(self):
        return (f"""
            Client public key: {self.client_public_key}
            Client secret key: {self.client_secret_key}
            Server public key: {self.server_public_key}
            Expiration time: {self.expires_at}
            QPU Endpoint: {self.qpu_endpoint}
            QPU Compiler Endpoint: {self.qpu_compiler_endpoint}
            """)
