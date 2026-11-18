"""
Edge case coverage tests for distributed swarm consensus (v0.3.0).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import pytest

from agent_lab.consensus import SwarmProposal
from agent_lab.distributed import (
    DistributedCoordinator,
    DistributedSwarmNode,
    MessageType,
    NetworkMessage,
)


def test_distributed_sync_request_and_response(tmp_path: Path):
    async def run_test():
        secret = "sync-secret"
        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            host="127.0.0.1",
            port=0,
            secret=secret,
        )
        await coordinator.start()
        url = f"ws://127.0.0.1:{coordinator.server_port}"

        worker = DistributedSwarmNode(
            node_id="sync-worker",
            coordinator_url=url,
            secret=secret,
        )
        await worker.connect()

        try:
            # Send sync request
            sync_req = NetworkMessage(
                msg_id="sync_req_1",
                msg_type=MessageType.SYNC_REQ,
                sender="sync-worker",
            )
            sync_req.sign(secret)
            await worker.ws.send(sync_req.to_json())

            # Read sync response
            raw_resp = await worker.ws.recv()
            resp = NetworkMessage.from_json(raw_resp)
            assert resp.msg_type == MessageType.SYNC_RESP
            assert "committed_versions" in resp.payload
        finally:
            await worker.disconnect()
            await coordinator.stop()

    asyncio.run(run_test())


def test_distributed_unauthenticated_message_dropped(tmp_path: Path):
    async def run_test():
        secret = "auth-guard-secret"
        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            host="127.0.0.1",
            port=0,
            secret=secret,
        )
        await coordinator.start()
        url = f"ws://127.0.0.1:{coordinator.server_port}"

        # Connect worker with WRONG secret
        worker = DistributedSwarmNode(
            node_id="bad-auth-worker",
            coordinator_url=url,
            secret="completely-wrong-secret",
        )
        connected = await worker.connect()
        # Handshake should not succeed because coordinator rejects signature
        assert connected is False

        await worker.disconnect()
        await coordinator.stop()

    asyncio.run(run_test())


def test_distributed_node_vote_while_disconnected_raises():
    node = DistributedSwarmNode(
        node_id="offline-node",
        coordinator_url="ws://127.0.0.1:9999",
    )
    with pytest.raises(RuntimeError, match="Cannot vote while disconnected"):
        asyncio.run(node.send_vote("rnd_1", "prop_1", True))


def test_distributed_node_listen_one_disconnected():
    node = DistributedSwarmNode(
        node_id="offline-node",
        coordinator_url="ws://127.0.0.1:9999",
    )
    res = asyncio.run(node.listen_one(timeout=0.1))
    assert res is None


def test_distributed_node_slashing_propagation(tmp_path: Path):
    async def run_test():
        secret = "slash-secret"
        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            host="127.0.0.1",
            port=0,
            secret=secret,
            quorum_ratio=0.5,
        )
        await coordinator.start()
        url = f"ws://127.0.0.1:{coordinator.server_port}"

        worker = DistributedSwarmNode(
            node_id="byzantine-node",
            coordinator_url=url,
            secret=secret,
        )
        await worker.connect()

        try:
            # Simulate slash broadcast from coordinator
            slash_msg = NetworkMessage(
                msg_id="slash_test",
                msg_type=MessageType.SLASH,
                sender="coordinator",
                payload={"slashed_validator": "byzantine-node", "reason": "Equivocation"},
            )
            await coordinator.broadcast(slash_msg)

            # Node listens and detects itself slashed
            msg = await worker.listen_one(timeout=1.0)
            assert msg is not None
            assert msg.msg_type == MessageType.SLASH
            assert worker.slashed is True
        finally:
            await worker.disconnect()
            await coordinator.stop()

    asyncio.run(run_test())
