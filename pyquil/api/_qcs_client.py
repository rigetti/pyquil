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
import asyncio

import qcs_sdk


def default_qcs_client() -> qcs_sdk.QcsClient:
    """
    Load a QCS Client.

    Raises:
        ``QcsLoadError``: If the client fails to load.
    """

    _ensure_event_loop()

    async def _load():
        return await qcs_sdk.QcsClient.load()

    return asyncio.get_event_loop().run_until_complete(_load())


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError as ex:
        if len(ex.args) > 0 and "There is no current event loop in thread" in ex.args[0]:
            asyncio.set_event_loop(asyncio.new_event_loop())
        else:
            raise
