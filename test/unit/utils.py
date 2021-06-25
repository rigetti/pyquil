from typing import Any
from unittest.mock import MagicMock

from pytest_mock import MockerFixture
from qcs_api_client.client import QCSClientConfiguration

from pyquil import Program
from pyquil.api._abstract_compiler import AbstractCompiler
from pyquil.parser import parse
from pyquil.quantum_processor import AbstractQuantumProcessor

# Valid, sample Z85-encoded keys specified by zmq curve for testing:
#   http://api.zeromq.org/master:zmq-curve#toc4
CLIENT_PUBLIC_KEY = "Yne@$w-vo<fVvi]a<NY6T1ed:M$fCG*[IaLV{hID"
CLIENT_SECRET_KEY = "D:)Q[IlAW!ahhC2ac:9*A}h:p?([4%wOTJ%JR%cs"
SERVER_PUBLIC_KEY = "rq:rM>}U?@Lns47E1%kR.o@n%FcmmsL/@{H8]yf7"
SERVER_SECRET_KEY = "JTKVSB%%)wK0E.X)V>+}o?pNmC{O&4W4b!Ni{Lh6"


def patch_rpcq_client(*, mocker: MockerFixture, return_value: Any) -> MagicMock:
    """Patch rpcq.Client and return the MagicMock object it's patched with.

    :param return_value: The response that should come back from calling client.call()
    :return the instance of MagicMock standing in for rpc.Client that other functions will use.
    """

    client = mocker.MagicMock()
    client.call.return_value = return_value

    def _construct(endpoint, timeout=None, auth_config=None):
        client.endpoint = endpoint
        client.timeout = timeout
        client.auth_config = auth_config
        return client

    constructor = mocker.patch("rpcq.Client", side_effect=_construct)

    return client


def parse_equals(quil_string, *instructions):
    expected = list(instructions)
    actual = parse(quil_string)
    assert expected == actual


class DummyCompiler(AbstractCompiler):
    def __init__(self, quantum_processor: AbstractQuantumProcessor, client_configuration: QCSClientConfiguration):
        super().__init__(quantum_processor=quantum_processor, timeout=10, client_configuration=client_configuration)

    def get_version_info(self):
        return {}

    def quil_to_native_quil(self, program: Program, *, protoquil=None):
        return program

    def native_quil_to_executable(self, nq_program: Program):
        return nq_program
