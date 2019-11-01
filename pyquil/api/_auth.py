import json
import re
import time
import requests
from os.path import exists, expanduser, isfile, join


class EngagementFailedError(RuntimeError):
    pass


class AuthClient:
    """
    Encapsulates the operations required for authorization & encryption
      with the QPU
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
        return join(self.config.forest_url, 'auth/idp/v1/token')

    @property
    def auth_tokens(self):
        if not self._auth_tokens:
            self._auth_tokens = self.fetch_auth_tokens_from_disk()
        return self._auth_tokens

    @property
    def access_token(self):
        return self.auth_tokens.get('access_token')

    @property
    def refresh_token(self):
        return self.auth_tokens.get('refresh_token')

    @property
    def engagement(self):
        if not (self._engagement and self._engagement.is_valid()):
            # TODO: needs a fix, to provide lattice_name
            self.engage()
        return self._engagement

    def fetch_auth_tokens_from_disk(self) -> dict:
        """
        Fetches the Rigetti API auth token from disk
        """
        if isfile(self.config.auth_token_path):
            with open(self.config.auth_token_path, 'r') as f:
                tokens = json.load(f)
                print(tokens)
                return tokens
        return {}

    def write_auth_tokens_to_disk(self, metadata={}):
        """
        Write the auth tokens to config files on disk
        """
        with open(self.config.auth_token_path, 'w') as f:
            json.dump(self.auth_tokens, f)

    def refresh_auth_token(self):
        """
        Get a new auth token from the auth server
        """
        print("Refreshing --------------")
        current_tokens = self.fetch_auth_tokens_from_disk()
        refresh_token = current_tokens.get('refresh_token')
        if not refresh_token:
            raise RuntimeError(
                f'No refresh token available - visit {self.config.qcs_ui_url} and follow the instructions under \'Account\' to get one.'
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
        response = requests.post(self.auth_token_refresh_endpoint,
                                 data=body,
                                 headers=headers)

        if response.ok:
            auth_tokens = response.json()
            expires_at = current_time + auth_tokens.get('expires_in', 0)
            self._auth_tokens = dict(**auth_tokens,
                                     **dict(expires_at=expires_at))
            self.write_auth_tokens_to_disk()
        else:
            raise RuntimeError("Failed to refresh token: ", response.__dict__)

    def engage(self, lattice_name):
        query = '''
          mutation Engage($name: String!) {
            engage(input: { lattice: { name: $name }}) {
              success
              message
              engagement {
                type
                clientKeys {
                  secret
                  public
                }
                serverKeys {
                  secret
                  public
                }
                endpoints {
                  compiler
                  qpu
                }
                expiresAt
              }
            }
          }
        '''
        variables = dict(name=lattice_name)
        query_response = self._send_graphql_with_auth(self.config.dispatch_url,
                                                      query, variables)

        print(query_response)

        if query_response.get('errors'):
            raise RuntimeError(
                f"Errors: {','.join(map(lambda error: error['message'], query_response.get('errors', [])))}"
            )

        if query_response is not None:
            engagement_response = query_response.get('data',
                                                     {}).get('engage', None)
            if engagement_response and engagement_response.get(
                    'success') is True:
                engagement_data = engagement_response.get('engagement', {})
                self._engagement = Engagement(
                    client_secret_key=engagement_data.get(
                        'clientKeys', {}).get('secret', '').encode('utf-8'),
                    client_public_key=engagement_data.get(
                        'clientKeys', {}).get('public', '').encode('utf-8'),
                    server_public_key=engagement_data.get(
                        'serverKeys', {}).get('public', '').encode('utf-8'),
                    expires_at=engagement_data.get('expiresAt', {}),
                    qpu_endpoint=engagement_data.get('endpoints',
                                                     {}).get('qpu'),
                    qpu_compiler_endpoint=engagement_data.get(
                        'endpoints', {}).get('compiler'))
            else:
                raise EngagementFailedError(
                    f"Unable to engage {lattice_name}:",
                    engagement_response.get('message', 'No message'))

        return self._engagement

    def _send_graphql_with_auth(self, *args, **kwargs):
        """
        If the auth token is expired on the first attempt, refresh it and then 
            perform a single retry
        """
        if self.access_token is None:
            raise RuntimeError('No access token found')
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

        response = requests.post(endpoint,
                                 data=json.dumps(data),
                                 headers=headers)
        try:
            result = response.json()
            return result
        except Exception as e:
            print("Unable to parse json from:", response.text())
            raise e


class Engagement:
    def __init__(self, client_public_key: bytes, client_secret_key: bytes,
                 server_public_key: bytes, expires_at, qpu_endpoint,
                 qpu_compiler_endpoint):
        self.client_public_key = client_public_key
        self.client_secret_key = client_secret_key
        self.server_public_key = server_public_key
        self.expires_at = float(expires_at) if expires_at else None
        self.qpu_endpoint = qpu_endpoint
        self.qpu_compiler_endpoint = qpu_compiler_endpoint

    def is_valid(self):
        return all([
            self.client_public_key is not None,
            self.client_secret_key is not None,
            self.server_public_key is not None,
            (self.expires_at is None
             or self.expires_at > time.time()), self.qpu_endpoint is not None
        ])
