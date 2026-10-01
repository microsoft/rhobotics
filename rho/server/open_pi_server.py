## Adapted from openpi/serving/websocket_policy_server.py
## Used in combination with the rho_client (adapted from openpi_client)
## Used until we can implement our own client class

import asyncio
import hmac
import http
import logging
import time

import websockets.asyncio.server as _server
import websockets.frames
from rho_client.msgpack_numpy import Packer, unpackb

from rho.eval.policy_interface import PolicyInterface

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGE_SIZE = 64 * 1024 * 1024
_INTERNAL_ERROR_MESSAGE = "Internal server error"


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """  # noqa: E501

    def __init__(
        self,
        policy_interface: PolicyInterface,
        env,
        host,
        port,
        *,
        api_key: str | None = None,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    ) -> None:
        if api_key == "":
            raise ValueError("api_key must be non-empty when authentication is enabled")
        if max_message_size <= 0:
            raise ValueError("max_message_size must be positive")

        self.policy_interface = policy_interface
        self.env = env
        self._host = host
        self._port = port
        self._api_key = api_key
        self._max_message_size = max_message_size
        self._metadata = {
            "action_type": env.policy_action_type,
            "execution_horizon": policy_interface.execution_horizon,
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_size,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                input = unpackb(await websocket.recv())

                obs = self.env.process_input(input)

                infer_time = time.monotonic()
                action = self.policy_interface.get_action_chunk(obs)
                infer_time = time.monotonic() - infer_time

                action = self.env.process_output(action)

                output = {}
                output["infer_ms"] = [infer_time * 1000]
                output["action"] = action

                await websocket.send(packer.pack(output))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.exception("Inference failed for connection %s", websocket.remote_address)
                try:
                    await websocket.send(_INTERNAL_ERROR_MESSAGE)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason=_INTERNAL_ERROR_MESSAGE,
                    )
                except websockets.ConnectionClosed:
                    pass
                break

    def _process_request(
        self,
        connection: _server.ServerConnection,
        request: _server.Request,
    ) -> _server.Response | None:
        health_response = _health_check(connection, request)
        if health_response is not None:
            return health_response

        if self._api_key is None:
            return None

        authorization = request.headers.get("Authorization")
        expected = f"Api-Key {self._api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
