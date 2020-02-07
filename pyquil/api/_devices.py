##############################################################################
# Copyright 2018 Rigetti Computing
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
from typing import Dict, List, Optional, cast

from requests.exceptions import MissingSchema

from pyquil.api._base_connection import get_json, get_session, ForestConnection
from pyquil.api._config import PyquilConfig
from pyquil.device._main import Device


def list_devices(connection: Optional[ForestConnection] = None) -> List[str]:
    """
    Query the Forest 2.0 server for a list of underlying QPU devices.

    NOTE: These can't directly be used to manufacture pyQuil Device objects, but this gives a list
          of legal values that can be supplied to list_lattices to filter its (potentially very
          noisy) output.

    :return: A list of device names.
    """
    # For the record, the dictionary stored in "devices" that we're getting back is keyed on device
    # names and has this structure in its values:
    #
    # {
    #   "is_online":   a boolean indicating the availability of the device,
    #   "is_retuning": a boolean indicating whether the device is busy retuning,
    #   "specs":       a Specs object describing the entire device, serialized as a dictionary,
    #   "isa":         an ISA object describing the entire device, serialized as a dictionary,
    #   "noise_model": a NoiseModel object describing the entire device, serialized as a dictionary
    # }
    if connection is None:
        connection = ForestConnection()

    session = connection.session
    assert connection.forest_cloud_endpoint is not None
    url = connection.forest_cloud_endpoint + "/devices"
    return sorted(get_json(session, url)["devices"].keys())


def list_lattices(
    device_name: Optional[str] = None,
    num_qubits: Optional[int] = None,
    connection: Optional[ForestConnection] = None,
) -> Dict[str, str]:
    """
    Query the Forest 2.0 server for its knowledge of lattices.  Optionally filters by underlying
    device name and lattice qubit count.

    :return: A dictionary keyed on lattice names and valued in dictionaries of the
        form::

            {
                "device_name": device_name,
                "qubits": num_qubits
            }
    """
    if connection is None:
        connection = ForestConnection()
    session = connection.session
    assert connection.forest_cloud_endpoint is not None
    url = connection.forest_cloud_endpoint + "/lattices"
    try:
        response = get_json(
            session, url, params={"device_name": device_name, "num_qubits": num_qubits}
        )

        return cast(Dict[str, str], response["lattices"])
    except Exception as e:
        raise ValueError(
            """
        list_lattices encountered an error when querying the Forest 2.0 endpoint.

        Some common causes for this error include:

        * You don't have valid user authentication information.  Very likely this is because you
          haven't yet been invited to try QCS.  We plan on making our device information publicly
          accessible soon, but in the meanwhile, you'll have to use default QVM configurations and
          to use `list_quantum_computers` with `qpus = False`.

        * You do have user authentication credentials, but they are invalid. You can visit
          https://qcs.rigetti.com/auth/token and save to ~/.qcs/user_auth_token to update your
          authentication credentials. Alternatively, you may provide the path to your credentials in
          your config file or with the USER_AUTH_TOKEN_PATH environment variable::

              [Rigetti Forest]
              user_auth_token_path = ~/.qcs/my_auth_credentials

        * You're missing an address for the Forest 2.0 server endpoint, or the address is invalid.
          This too can be set through the environment variable FOREST_URL or by changing the
          following lines in the QCS config file::

              [Rigetti Forest]
              url = https://forest-server.qcs.rigetti.com

        For the record, here's the original exception: {}
        """.format(
                repr(e)
            )
        )


def get_lattice(lattice_name: Optional[str] = None) -> Device:
    """
    Construct a Device object to match the Forest 2.0 server's understanding of the named lattice.

    :param lattice_name: Name of the desired lattice.
    :return: A Device object.
    """
    raw_lattice = _get_raw_lattice_data(lattice_name)

    return Device(raw_lattice["name"], raw_lattice)


def _get_raw_lattice_data(lattice_name: Optional[str] = None) -> Dict[str, str]:
    """
    Produces a dictionary of raw data for a lattice as queried from the Forest 2.0 server.

    Returns a dictionary of the form::

        {
            "name":        the name of the lattice as a string,
            "device_name": the name of the device, given as a string, that the lattice lies on,
            "specs":       a Specs object, serialized as a dictionary,
            "isa":         an ISA object, serialized as a dictionary,
            "noise_model": a NoiseModel object, serialized as a dictionary
        }
    """

    config = PyquilConfig()
    session = get_session(config=config)

    try:
        res = get_json(session, f"{config.forest_url}/lattices/{lattice_name}")
    except MissingSchema:
        raise ValueError(
            f"Error finding lattice `{lattice_name}` at Forest 2.0 server "
            f"""endpoint `{config.forest_url}`.

    Most likely, you're missing an address for the Forest 2.0 server endpoint, or the
    address is invalid. This can be set through the environment variable FOREST_URL or
    by changing the following lines in the QCS config file (by default, at ~/.qcs_config)::

       [Rigetti Forest]
       url = https://rigetti.com/valid/forest/url"""
        )
    return cast(Dict[str, str], res["lattice"])
