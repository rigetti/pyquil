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

    # renames the published package name, but not the import name
    data["project"] = { "name": "pyquil-grpc-web" }
    data["tool"]["poetry"]["name"] = "pyquil-grpc-web"

    # use the same dependency definition, but change the name
    # to the grpc-web version of the package.
    deps = data["tool"]["poetry"]["dependencies"]
    deps["qcs-sdk-python-grpc-web"] = deps["qcs-sdk-python"]
    del deps["qcs-sdk-python"]

    write(f, data)

# The `__package__` will refer to `pyquil`, but this
# package is now `pyquil_grpc_web` - this is the simplest
# way to make that overwrite.
with open(join(workspace_path, "pyquil", "_version.py"), "r+") as f:
    updated = f.read().replace("__package__", '"pyquil_grpc_web"')
    f.seek(0)
    f.truncate()
    f.write(updated)