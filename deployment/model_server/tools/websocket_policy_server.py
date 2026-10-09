# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import pickle
import time
import traceback
from collections import deque
from functools import partial

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        max_inference_batch_size: int = 1,
        inference_batch_wait_ms: float = 10.0,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._max_inference_batch_size = max(1, int(max_inference_batch_size))
        self._inference_batch_wait_seconds = max(0.0, float(inference_batch_wait_ms) / 1000.0)
        self._inference_queue: asyncio.Queue | None = None
        self._inference_lock: asyncio.Lock | None = None
        self._batch_count = 0
        self._batch_sample_count = 0
        self._last_batch_stats_log = time.monotonic()
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._inference_queue = asyncio.Queue()
        self._inference_lock = asyncio.Lock()
        batch_worker = None
        if self._max_inference_batch_size > 1:
            batch_worker = asyncio.create_task(self._inference_batch_worker())
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
            ) as server:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
        finally:
            if batch_worker is not None:
                batch_worker.cancel()
                try:
                    await batch_worker
                except asyncio.CancelledError:
                    pass

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                ret = await self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    async def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                output_dict = await self._predict_action(payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }

    async def _predict_action(self, payload: dict) -> dict:
        """Run inference without blocking the websocket event loop.

        When batching is enabled, requests with matching inference options are
        combined into one framework call. The framework already accepts a list
        of examples and returns one action chunk per example.
        """
        if self._max_inference_batch_size <= 1 or self._inference_queue is None:
            if self._inference_lock is None:
                return await asyncio.to_thread(partial(self._policy.predict_action, **payload))
            async with self._inference_lock:
                return await asyncio.to_thread(partial(self._policy.predict_action, **payload))

        future = asyncio.get_running_loop().create_future()
        await self._inference_queue.put((payload, future))
        return await future

    @staticmethod
    def _batch_options(payload: dict) -> bytes:
        # LIBERO sends small scalar inference options; pickle also handles
        # nested option values without relying on equality for NumPy arrays.
        try:
            return pickle.dumps(
                {key: value for key, value in payload.items() if key != "examples"},
                protocol=5,
            )
        except Exception:
            # An unpickleable option can still be served safely as a singleton.
            return f"singleton:{id(payload)}".encode()

    async def _inference_batch_worker(self) -> None:
        assert self._inference_queue is not None
        pending = deque()
        while True:
            first = pending.popleft() if pending else await self._inference_queue.get()
            first_payload, _ = first
            signature = self._batch_options(first_payload)
            batch = [first]
            while pending and len(batch) < self._max_inference_batch_size:
                item = pending[0]
                if self._batch_options(item[0]) != signature:
                    break
                batch.append(pending.popleft())
            deadline = asyncio.get_running_loop().time() + self._inference_batch_wait_seconds

            while len(batch) < self._max_inference_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._inference_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if self._batch_options(item[0]) == signature:
                    batch.append(item)
                else:
                    pending.append(item)

            self._batch_count += 1
            self._batch_sample_count += len(batch)
            now = time.monotonic()
            if now - self._last_batch_stats_log >= 30.0:
                mean_batch = self._batch_sample_count / self._batch_count
                logging.info(
                    "Dynamic inference batching: %d model calls / %d requests in the last interval "
                    "(mean batch %.2f, max %d)",
                    self._batch_count,
                    self._batch_sample_count,
                    mean_batch,
                    self._max_inference_batch_size,
                )
                self._batch_count = 0
                self._batch_sample_count = 0
                self._last_batch_stats_log = now

            try:
                outputs = await asyncio.to_thread(self._run_inference_batch, batch)
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
            else:
                for (_, future), output in zip(batch, outputs):
                    if not future.done():
                        future.set_result(output)

    def _run_inference_batch(self, batch: list[tuple[dict, asyncio.Future]]) -> list[dict]:
        if len(batch) == 1:
            return [self._policy.predict_action(**batch[0][0])]

        merged_payload = dict(batch[0][0])
        merged_examples = []
        offsets = []
        for payload, _ in batch:
            examples = payload.get("examples")
            if not isinstance(examples, list) or not examples:
                raise ValueError("Batched inference requires a non-empty 'examples' list in every request")
            start = len(merged_examples)
            merged_examples.extend(examples)
            offsets.append((start, len(merged_examples)))
        merged_payload["examples"] = merged_examples
        output = self._policy.predict_action(**merged_payload)

        split_outputs = []
        for start, end in offsets:
            split = {}
            for key, value in output.items():
                shape = getattr(value, "shape", None)
                if shape is not None and len(shape) > 0 and int(shape[0]) == len(merged_examples):
                    split[key] = value[start:end]
                else:
                    split[key] = value
            split_outputs.append(split)
        return split_outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
