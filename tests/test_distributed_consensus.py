import asyncio
import hashlib
import time
from pathlib import Path

from agent_lab.consensus import SwarmProposal
from agent_lab.distributed import (
    DistributedCoordinator,
    DistributedSwarmNode,
    MessageType,
    NetworkMessage,
    PeerInfo,
    deserialize_proposal,
    serialize_proposal,
)


def test_network_message_canonical_and_hmac():
    msg = NetworkMessage(
        msg_id="m1",
        msg_type=MessageType.PROPOSAL,
        sender="worker-1",
        round_id="rnd_1",
        payload={"task": "build_service", "params": {"retries": 3}},
    )
    secret = "super-secret-key-1234"
    msg.sign(secret)

    assert msg.signature is not None
    assert msg.verify(secret) is True
    assert msg.verify("wrong-secret") is False

    # Tampering payload should invalidate HMAC
    msg.payload["task"] = "malicious_injection"
    assert msg.verify(secret) is False


def test_proposal_serialization_roundtrip():
    content = b"print('production zero copy')\n"
    content_hash = hashlib.sha256(content).hexdigest()
    proposal = SwarmProposal(
        proposal_id="prop_roundtrip",
        proposer="agent-alpha",
        description="Verify roundtrip serialization",
        read_set={"main.py": "abc123"},
        write_set={"main.py": content_hash},
        file_mutations={"main.py": content},
    )

    serialized = serialize_proposal(proposal)
    assert isinstance(serialized["file_mutations"]["main.py"], str)

    restored = deserialize_proposal(serialized)
    assert restored.proposal_id == proposal.proposal_id
    assert restored.file_mutations["main.py"] == content


def test_distributed_handshake_and_heartbeat(tmp_path: Path):
    async def run_test():
        secret = "cluster-shared-secret"
        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            host="127.0.0.1",
            port=0,
            secret=secret,
        )
        await coordinator.start()
        port = coordinator.server_port
        url = f"ws://127.0.0.1:{port}"

        worker = DistributedSwarmNode(
            node_id="worker-node-1",
            coordinator_url=url,
            secret=secret,
            weight=1.5,
        )

        try:
            connected = await worker.connect()
            assert connected is True
            assert worker.is_connected is True
            assert "worker-node-1" in coordinator.peers
            assert coordinator.peers["worker-node-1"].weight == 1.5

            # Send heartbeat
            await worker.send_heartbeat()
            ack = await worker.listen_one(timeout=2.0)
            assert ack is not None
            assert ack.msg_type == MessageType.HEARTBEAT_ACK
        finally:
            await worker.disconnect()
            await coordinator.stop()

    asyncio.run(run_test())


def test_distributed_bft_quorum_consensus_success(tmp_path: Path):
    async def run_test():
        secret = "bft-secret-token"
        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            host="127.0.0.1",
            port=0,
            secret=secret,
            quorum_ratio=0.66,
        )
        await coordinator.start()
        url = f"ws://127.0.0.1:{coordinator.server_port}"

        # Connect two worker nodes
        workers = []
        for i in range(1, 3):
            w = DistributedSwarmNode(
                node_id=f"validator-{i}",
                coordinator_url=url,
                secret=secret,
                weight=1.0,
                vote_callback=lambda p: (True, "Consensus verified"),
            )
            await w.connect()
            workers.append(w)

        try:
            async def worker_loop(node: DistributedSwarmNode):
                while node.is_connected:
                    await node.listen_one(timeout=0.5)

            tasks = [asyncio.create_task(worker_loop(w)) for w in workers]

            file_content = b"export const BFT_ENABLED = true;\n"
            file_hash = hashlib.sha256(file_content).hexdigest()
            proposal = SwarmProposal(
                proposal_id="prop_bft_01",
                proposer="coordinator",
                description="Enable BFT flags",
                read_set={},
                write_set={"config.ts": file_hash},
                file_mutations={"config.ts": file_content},
            )

            res = await coordinator.broadcast_proposal(proposal=proposal, timeout=2.0)

            assert res.committed is True
            assert res.quorum_reached is True
            assert "config.ts" in res.applied_files
            assert (tmp_path / "config.ts").read_bytes() == file_content

            for t in tasks:
                t.cancel()
        finally:
            for w in workers:
                await w.disconnect()
            await coordinator.stop()

    asyncio.run(run_test())


def test_distributed_bft_rejection_and_rollback(tmp_path: Path):
    async def run_test():
        secret = "bft-reject-token"
        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            host="127.0.0.1",
            port=0,
            secret=secret,
            quorum_ratio=0.66,
        )
        await coordinator.start()
        url = f"ws://127.0.0.1:{coordinator.server_port}"

        workers = []
        for i in range(1, 3):
            w = DistributedSwarmNode(
                node_id=f"strict-validator-{i}",
                coordinator_url=url,
                secret=secret,
                weight=1.0,
                vote_callback=lambda p: (False, "Security policy violation"),
            )
            await w.connect()
            workers.append(w)

        try:
            async def worker_loop(node: DistributedSwarmNode):
                while node.is_connected:
                    await node.listen_one(timeout=0.5)

            tasks = [asyncio.create_task(worker_loop(w)) for w in workers]

            proposal = SwarmProposal(
                proposal_id="prop_rejected",
                proposer="coordinator",
                description="Malicious change",
                read_set={},
                write_set={"bad.py": "1234"},
                file_mutations={"bad.py": b"evil_code()"},
            )

            res = await coordinator.broadcast_proposal(proposal=proposal, timeout=2.0)

            assert res.committed is False
            assert res.quorum_reached is False
            assert not (tmp_path / "bad.py").exists()

            for t in tasks:
                t.cancel()
        finally:
            for w in workers:
                await w.disconnect()
            await coordinator.stop()

    asyncio.run(run_test())
