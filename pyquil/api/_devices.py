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
from pyquil.api._base_connection import get_session, get_json
from pyquil.api._config import PyquilConfig


def list_devices():
    """
    Query the Forest 2.0 server for its knowledge of QPUs.

    :return: A dictionary, keyed on device names. Each value is a dictionary of the form
    {
        "is_online":   a boolean indicating the availability of the device,
        "is_retuning": a boolean indicating whether the device is busy retuning,
        "specs":       a Specs object describing the entire device, serialized as a dictionary,
        "isa":         an ISA object describing the entire device, serialized as a dictionary,
        "noise_model": a NoiseModel object describing the entire device, serialized as a dictionary
    }
    """
    session = get_session()
    config = PyquilConfig()

    return get_json(session, config.forest_url + "/devices")["devices"]


def get_device(name: str):
    """
    Construct a Device object from a QCS-available device.

    :param name: Name of the desired device.
    :return: A Device object.
    """
    raise NotImplementedError('QPU devices will be available at a later date, please use the QVM')
