import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig

client_module = pytest.importorskip("rho_client.websocket_client_policy")
WebsocketClientPolicy = client_module.WebsocketClientPolicy
msgpack_numpy = client_module.msgpack_numpy


@pytest.fixture
def client(monkeypatch):
    connection = MagicMock()
    connection.recv.return_value = msgpack_numpy.packb({"action": np.zeros((4, 2), dtype=np.float32)})
    monkeypatch.setattr(WebsocketClientPolicy, "_wait_for_server", lambda self: (connection, {}))
    return WebsocketClientPolicy()


def test_inference_without_reset_preserves_existing_flag(client):
    observation = {"_reset_": False, "task": ["move"]}

    client.infer(observation)

    assert msgpack_numpy.unpackb(client._ws.send.call_args.args[0]) == observation


@pytest.mark.parametrize("observation", [{"task": ["move"]}, {"task": ["move"], "_reset_": False}])
def test_reset_marks_only_next_inference_without_mutating_observation(client, observation):
    original = dict(observation)
    client.reset()
    client.reset()
    client._ws.send.assert_not_called()

    result = client.infer(observation)
    request = msgpack_numpy.unpackb(client._ws.send.call_args.args[0])

    assert request == {**original, "_reset_": True}
    assert observation == original
    assert result["action"].shape == (4, 2)

    client.infer(observation)

    assert msgpack_numpy.unpackb(client._ws.send.call_args.args[0]) == original


@pytest.mark.parametrize("failure", ["send", "receive", "server", "decode"])
def test_reset_remains_pending_after_failed_inference(client, failure):
    client.reset()
    if failure == "send":
        client._ws.send.side_effect = OSError("send failed")
    elif failure == "receive":
        client._ws.recv.side_effect = OSError("receive failed")
    elif failure == "server":
        client._ws.recv.return_value = "server failed"
    else:
        client._ws.recv.return_value = b"\xc1"

    with pytest.raises((OSError, RuntimeError, ValueError)):
        client.infer({"task": ["move"]})

    client._ws.send.side_effect = None
    client._ws.recv.side_effect = None
    client._ws.recv.return_value = msgpack_numpy.packb({"action": []})
    client.infer({"task": ["move"]})
    assert msgpack_numpy.unpackb(client._ws.send.call_args.args[0])["_reset_"] is True

    client.infer({"task": ["move"]})
    assert "_reset_" not in msgpack_numpy.unpackb(client._ws.send.call_args.args[0])


@pytest.mark.parametrize("eval_mode", ["standard", "rtc"])
def test_reset_clears_server_episode_state_over_websocket(eval_mode):
    from websockets.asyncio.server import serve

    from rho.server.open_pi_server import WebsocketPolicyServer

    policy = SimpleNamespace(
        config=SimpleNamespace(
            chunk_size=4,
            n_action_steps=2,
            delta_indices_dict={"observation.state": [-1, 0]},
            feature_dict={},
        ),
        model=SimpleNamespace(action_expert=SimpleNamespace(enable_gradient_checkpointing=True)),
        sample_actions=MagicMock(return_value={"actions": torch.ones(1, 4, 2)}),
        sample_actions_rtc=MagicMock(return_value={"actions": torch.ones(1, 4, 2)}),
    )
    config = PolicyInterfaceConfig(
        policy=policy,
        device="cpu",
        eval_mode=eval_mode,
        inference_delay=1,
        beta=10,
    )
    config.input_transforms = lambda obs: obs
    config.output_transforms = lambda obs: obs
    policy_interface = PolicyInterface(config)
    environment = SimpleNamespace(
        policy_action_type="POSITION",
        process_input=lambda obs: {
            key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
            for key, value in obs.items()
        },
        process_output=lambda actions: actions.squeeze(0).numpy(),
    )
    server = WebsocketPolicyServer(policy_interface, environment, "127.0.0.1", 0)

    def exercise_client(port):
        client = WebsocketClientPolicy(host="127.0.0.1", port=port)
        try:

            def observation(value, executed=0):
                return {
                    "observation.state": np.full((1, 1, 2), value, dtype=np.float32),
                    "num_actions_executed": executed,
                }

            client.infer(observation(1))
            client.infer(observation(2, executed=2))
            assert len(policy_interface.obs_queue["observation.state"]) == 2
            if eval_mode == "rtc":
                assert policy_interface.prev_action_chunk is not None
                assert policy.sample_actions_rtc.call_args.kwargs["prev_actions"] is not None

            client.reset()
            client.infer(observation(7))

            history = list(policy_interface.obs_queue["observation.state"])
            assert len(history) == 1
            torch.testing.assert_close(history[0], torch.full((1, 1, 2), 7.0))
            if eval_mode == "rtc":
                assert policy.sample_actions_rtc.call_args.kwargs["prev_actions"] is None

            client.infer(observation(8, executed=2))
            assert len(policy_interface.obs_queue["observation.state"]) == 2
        finally:
            client._ws.close()

    async def run_round_trip():
        async with serve(server._handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            await asyncio.to_thread(exercise_client, port)

    asyncio.run(run_round_trip())
