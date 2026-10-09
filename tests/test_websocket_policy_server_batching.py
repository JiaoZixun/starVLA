import asyncio
import contextlib

import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _BatchRecordingPolicy:
    def __init__(self):
        self.calls = []

    def predict_action(self, examples, **kwargs):
        self.calls.append((examples, kwargs))
        values = [ord(example["lang"][0]) for example in examples]
        return {"actions": np.asarray(values, dtype=np.int32)[:, None, None]}


def test_concurrent_requests_are_batched_and_outputs_are_split_per_client():
    policy = _BatchRecordingPolicy()
    server = WebsocketPolicyServer(
        policy,
        max_inference_batch_size=2,
        inference_batch_wait_ms=50,
    )

    async def exercise_batcher():
        server._inference_queue = asyncio.Queue()
        worker = asyncio.create_task(server._inference_batch_worker())
        try:
            first, second = await asyncio.gather(
                server._predict_action({"examples": [{"lang": "apple"}], "use_ddim": True}),
                server._predict_action({"examples": [{"lang": "banana"}], "use_ddim": True}),
            )
            return first, second
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    first, second = asyncio.run(exercise_batcher())

    assert len(policy.calls) == 1
    batched_examples, options = policy.calls[0]
    assert [example["lang"] for example in batched_examples] == ["apple", "banana"]
    assert options == {"use_ddim": True}
    np.testing.assert_array_equal(first["actions"].reshape(-1), [ord("a")])
    np.testing.assert_array_equal(second["actions"].reshape(-1), [ord("b")])
