import asyncio
import os
import signal
import time
from socket import socket
from contextlib import contextmanager
from multiprocessing import Process

import rpcq
from qcs_api_client.client import QCSClientConfiguration

from pyquil import Program
from pyquil.api._abstract_compiler import AbstractCompiler
from pyquil.parser import parse
from pyquil.quantum_processor import AbstractQuantumProcessor


@contextmanager
def run_rpcq_server(server: rpcq.Server, port: int):
    def run_server():
        server.run(endpoint=f"tcp://*:{port}", loop=asyncio.new_event_loop())

    def check_server():
        connected = False
        tries = 0
        exception = None

        while not connected and tries < 2:
            s = socket()
            try:
                s.connect(("localhost", port))
                connected = True
            except Exception as ex:
                exception = ex
                time.sleep(0.25)
            finally:
                s.close()

            tries += 1

        if not connected:
            raise Exception(f"Unable to connect to test rpcq server on port {port}: {exception}")

    proc = Process(target=run_server)
    try:
        proc.start()
        check_server()
        yield
    finally:
        os.kill(proc.pid, signal.SIGINT)


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
