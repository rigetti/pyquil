"""
Changes the qcs-sdk-python dependency to qcs-sdk-python-grpc-web.
"""

from io import TextIOWrapper
from os.path import dirname, realpath, join
import toml

workspace_path = dirname(dirname(realpath(__file__)))

def write(f: TextIOWrapper, data):
    f.seek(0)
    f.write(toml.dumps(data))
    f.truncate()

with open(join(workspace_path, "pyproject.toml"), "r+") as f:
    data = toml.load(f)
    deps = data["tool"]["poetry"]["dependencies"]

    deps["qcs-sdk-python-grpc-web"] = deps["qcs-sdk-python"]
    del deps["qcs-sdk-python"]

    write(f, data)