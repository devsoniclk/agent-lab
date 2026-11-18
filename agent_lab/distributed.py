"""
agent_lab.distributed: Distributed WebSocket Consensus Transport for
multi-host autonomous agent swarms with Byzantine Fault Tolerance (BFT),
cryptographic HMAC message authentication, and optimistic disk synchronization.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import websockets

from agent_lab.consensus import (
    SwarmConsensusResult,
    SwarmCoordinator,
    SwarmProposal,
    SwarmVote,
)
from agent_lab.registry import AnchorAgentRegistryClient

logger = logging.getLogger("agent_lab.distributed")


class MessageType(str, Enum):
    """Protocol message types for distributed swarm consensus."""
    HANDSHAKE = "HANDSHAKE"
    HANDSHAKE_ACK = "HANDSHAKE_ACK"
    HEARTBEAT = "HEARTBEAT"
    HEARTBEAT_ACK = "HEARTBEAT_ACK"
    PROPOSAL = "PROPOSAL"
    VOTE = "VOTE"
    COMMIT = "COMMIT"
    REJECT = "REJECT"
    SLASH = "SLASH"
    SYNC_REQ = "SYNC_REQ"
    SYNC_RESP = "SYNC_RESP"


@dataclass
class NetworkMessage:
    """Envelope for all distributed consensus protocol messages."""
    msg_id: str
    msg_type: str
    sender: str
    round_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None

    def canonical_bytes(self) -> bytes:
        """Deterministically serialize message attributes for HMAC signing."""
        canonical_dict = {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type,
            "sender": self.sender,
            "round_id": self.round_id,
            "payload": self.payload,
            "timestamp": round(self.timestamp, 4),
        }
        return json.dumps(canonical_dict, sort_keys=True).encode("utf-8")

    def sign(self, secret: str) -> None:
        """Sign message using HMAC-SHA256."""
        if not secret:
            self.signature = None
            return
        mac = hmac.new(secret.encode("utf-8"), self.canonical_bytes(), hashlib.sha256)
        self.signature = mac.hexdigest()

    def verify(self, secret: str) -> bool:
        """Verify HMAC-SHA256 message signature."""
        if not secret:
            return True
        if not self.signature:
            return False
        expected = hmac.new(secret.encode("utf-8"), self.canonical_bytes(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def to_json(self) -> str:
        """Serialize envelope to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> NetworkMessage:
        """Deserialize envelope from JSON string."""
        data = json.loads(raw)
        return cls(
            msg_id=data["msg_id"],
            msg_type=data["msg_type"],
            sender=data["sender"],
            round_id=data.get("round_id"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            signature=data.get("signature"),
        )


@dataclass
class PeerInfo:
    """Metadata tracking a connected swarm validator node."""
    node_id: str
    weight: float = 1.0
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    status: str = "HEALTHY"  # HEALTHY, DEGRADED, SLASHED, DISCONNECTED


def serialize_proposal(proposal: SwarmProposal) -> Dict[str, Any]:
    """Convert SwarmProposal to JSON-safe dictionary with hex-encoded byte mutations."""
    mutations_hex = {k: v.hex() for k, v in proposal.file_mutations.items()}
    return {
        "proposal_id": proposal.proposal_id,
        "proposer": proposal.proposer,
        "description": proposal.description,
        "read_set": proposal.read_set,
        "write_set": proposal.write_set,
        "file_mutations": mutations_hex,
        "zk_receipt": proposal.zk_receipt,
        "timestamp": proposal.timestamp,
    }


def deserialize_proposal(data: Dict[str, Any]) -> SwarmProposal:
    """Convert JSON dictionary back to SwarmProposal restoring bytes."""
    mutations_bytes = {k: bytes.fromhex(v) for k, v in data.get("file_mutations", {}).items()}
    return SwarmProposal(
        proposal_id=data["proposal_id"],
        proposer=data["proposer"],
        description=data.get("description", ""),
        read_set=data.get("read_set", {}),
        write_set=data.get("write_set", {}),
        file_mutations=mutations_bytes,
        zk_receipt=data.get("zk_receipt"),
        timestamp=data.get("timestamp", time.time()),
    )


class DistributedCoordinator:
    """
    Central or Leader node that hosts a WebSocket server, manages validator
    swarms, handles OCC verification, collects consensus votes, and broadcasts commits.
    """

    def __init__(
        self,
        workspace_root: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        secret: str = "",
        quorum_ratio: float = 2.0 / 3.0,
        registry_client: Optional[AnchorAgentRegistryClient] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.host = host
        self.port = port
        self.secret = secret
        self.quorum_ratio = quorum_ratio
        self.registry_client = registry_client

        self.coordinator = SwarmCoordinator(
            workspace_root=str(self.workspace_root),
            quorum_ratio=self.quorum_ratio,
            registry_client=self.registry_client,
        )

        self.peers: Dict[str, PeerInfo] = {}
        self.connections: Dict[str, Any] = {}  # node_id -> websocket
        self._server: Optional[Any] = None
        self._running = False
        self._active_rounds: Dict[str, Dict[str, SwarmVote]] = {}
        self._round_events: Dict[str, asyncio.Event] = {}
        self._round_history: List[SwarmConsensusResult] = []

    @property
    def server_port(self) -> int:
        """Return the actual bound port (useful when dynamically bound to port 0)."""
        if self._server and hasattr(self._server, "sockets") and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self.port

    async def start(self) -> None:
        """Start the WebSocket consensus server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info(f"Distributed Coordinator listening on ws://{self.host}:{self.server_port}")

    async def stop(self) -> None:
        """Gracefully close all peer connections and stop server."""
        self._running = False
        for node_id, ws in list(self.connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self.connections.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("Distributed Coordinator stopped.")

    async def _handle_connection(self, websocket: Any) -> None:
        """Process incoming client connections and message loop."""
        remote_node_id: Optional[str] = None
        try:
            async for raw in websocket:
                try:
                    msg = NetworkMessage.from_json(raw)
                except Exception as err:
                    logger.warning(f"Malformed message received: {err}")
                    continue

                # Verify HMAC signature
                if self.secret and not msg.verify(self.secret):
                    logger.warning(f"HMAC signature verification failed for sender '{msg.sender}'")
                    try:
                        await websocket.close(code=4001, reason="Invalid HMAC signature")
                    except Exception:
                        pass
                    break

                if msg.msg_type == MessageType.HANDSHAKE:
                    remote_node_id = msg.sender
                    weight = float(msg.payload.get("weight", 1.0))

                    if self.registry_client is not None:
                        eligible, reason = self.registry_client.verify_agent_eligibility(remote_node_id)
                        if not eligible:
                            logger.warning(f"Rejecting ineligible peer '{remote_node_id}': {reason}")
                            rej_msg = NetworkMessage(
                                msg_id=f"rej_{int(time.time()*1000)}",
                                msg_type=MessageType.REJECT,
                                sender="coordinator",
                                payload={"error": "INELIGIBLE_VALIDATOR", "reason": reason},
                            )
                            if self.secret:
                                rej_msg.sign(self.secret)
                            try:
                                await websocket.send(rej_msg.to_json())
                                await websocket.close(code=4003, reason=reason)
                            except Exception:
                                pass
                            break

                    self.peers[remote_node_id] = PeerInfo(
                        node_id=remote_node_id,
                        weight=weight,
                        connected_at=time.time(),
                        last_seen=time.time(),
                    )
                    self.connections[remote_node_id] = websocket
                    ack = NetworkMessage(
                        msg_id=f"ack_{int(time.time()*1000)}",
                        msg_type=MessageType.HANDSHAKE_ACK,
                        sender="coordinator",
                        payload={"status": "CONNECTED", "quorum_ratio": self.quorum_ratio},
                    )
                    ack.sign(self.secret)
                    await websocket.send(ack.to_json())

                elif msg.msg_type == MessageType.HEARTBEAT:
                    if msg.sender in self.peers:
                        self.peers[msg.sender].last_seen = time.time()
                        ack = NetworkMessage(
                            msg_id=f"hb_ack_{int(time.time()*1000)}",
                            msg_type=MessageType.HEARTBEAT_ACK,
                            sender="coordinator",
                            payload={"server_time": time.time()},
                        )
                        ack.sign(self.secret)
                        await websocket.send(ack.to_json())

                elif msg.msg_type == MessageType.VOTE:
                    round_id = msg.round_id or msg.payload.get("round_id")
                    if round_id and round_id in self._active_rounds:
                        vote = SwarmVote(
                            voter=msg.sender,
                            proposal_id=msg.payload.get("proposal_id", ""),
                            decision=bool(msg.payload.get("decision", True)),
                            weight=float(msg.payload.get("weight", 1.0)),
                            signature=msg.signature,
                            reason=msg.payload.get("reason"),
                            timestamp=msg.timestamp,
                        )
                        self._active_rounds[round_id][msg.sender] = vote
                        # Check if all active non-slashed peers have voted
                        active_voters = {
                            n for n, p in self.peers.items()
                            if p.status != "SLASHED" and n in self.connections
                        }
                        if active_voters.issubset(set(self._active_rounds[round_id].keys())):
                            event = self._round_events.get(round_id)
                            if event:
                                event.set()

                elif msg.msg_type == MessageType.SYNC_REQ:
                    # Provide state summary to reconnecting worker
                    resp = NetworkMessage(
                        msg_id=f"sync_{int(time.time()*1000)}",
                        msg_type=MessageType.SYNC_RESP,
                        sender="coordinator",
                        payload={
                            "committed_versions": self.coordinator.committed_versions,
                            "slashed_validators": list(self.coordinator.slashed_validators),
                            "recent_rounds": len(self._round_history),
                        },
                    )
                    resp.sign(self.secret)
                    await websocket.send(resp.to_json())

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if remote_node_id and remote_node_id in self.connections:
                del self.connections[remote_node_id]
                if remote_node_id in self.peers:
                    self.peers[remote_node_id].status = "DISCONNECTED"

    async def broadcast(self, message: NetworkMessage, exclude: Optional[str] = None) -> int:
        """Broadcast a message to all active peer connections."""
        message.sign(self.secret)
        raw = message.to_json()
        count = 0
        for node_id, ws in list(self.connections.items()):
            if exclude and node_id == exclude:
                continue
            try:
                await ws.send(raw)
                count += 1
            except Exception as e:
                logger.debug(f"Failed sending broadcast to {node_id}: {e}")
        return count

    async def broadcast_proposal(
        self,
        proposal: SwarmProposal,
        timeout: float = 3.0,
    ) -> SwarmConsensusResult:
        """
        Broadcast a proposal to all connected validators, collect remote votes,
        evaluate BFT quorum and OCC constraints, and dispatch commit/reject messages.
        """
        start_time = time.time()
        round_id = f"rnd_{int(start_time*1000)}"

        self._active_rounds[round_id] = {}
        round_event = asyncio.Event()
        self._round_events[round_id] = round_event

        # 1. Broadcast Proposal message
        msg = NetworkMessage(
            msg_id=f"prop_{int(start_time*1000)}",
            msg_type=MessageType.PROPOSAL,
            sender="coordinator",
            round_id=round_id,
            payload={
                "round_id": round_id,
                "proposal": serialize_proposal(proposal),
            },
        )
        await self.broadcast(msg)

        # 2. Wait for votes up to timeout
        try:
            await asyncio.wait_for(round_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        # 3. Assemble votes and validator weights
        votes_dict = self._active_rounds.pop(round_id, {})
        self._round_events.pop(round_id, None)

        votes: List[SwarmVote] = list(votes_dict.values())
        validators: Dict[str, float] = {
            peer_id: peer.weight
            for peer_id, peer in self.peers.items()
            if peer.status != "SLASHED"
        }

        # 4. Execute atomic BFT consensus evaluation
        result = self.coordinator.run_consensus_round(
            proposal=proposal,
            votes=votes,
            validators=validators,
        )

        # 5. Broadcast slash notices if any equivocations occurred
        for slashed in result.equivocations:
            if slashed in self.peers:
                self.peers[slashed].status = "SLASHED"
            slash_msg = NetworkMessage(
                msg_id=f"slash_{int(time.time()*1000)}",
                msg_type=MessageType.SLASH,
                sender="coordinator",
                round_id=round_id,
                payload={
                    "slashed_validator": slashed,
                    "reason": "Equivocation double voting",
                    "slashing_receipts": result.slashing_receipts,
                },
            )
            await self.broadcast(slash_msg)

        # 6. Broadcast Commit or Reject result
        outcome_type = MessageType.COMMIT if result.committed else MessageType.REJECT
        outcome_msg = NetworkMessage(
            msg_id=f"res_{int(time.time()*1000)}",
            msg_type=outcome_type,
            sender="coordinator",
            round_id=round_id,
            payload={
                "round_id": round_id,
                "proposal_id": proposal.proposal_id,
                "committed": result.committed,
                "quorum_reached": result.quorum_reached,
                "applied_files": result.applied_files,
                "conflicts": result.conflicts,
            },
        )
        await self.broadcast(outcome_msg)

        self._round_history.append(result)
        return result


class DistributedSwarmNode:
    """
    Worker or Validator node that connects to a DistributedCoordinator over WebSockets,
    receives proposals, verifies local constraints, and casts signed votes.
    """

    def __init__(
        self,
        node_id: str,
        coordinator_url: str,
        secret: str = "",
        weight: float = 1.0,
        vote_callback: Optional[Callable[[SwarmProposal], Tuple[bool, Optional[str]]]] = None,
    ) -> None:
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.secret = secret
        self.weight = weight
        self.vote_callback = vote_callback

        self.ws: Optional[Any] = None
        self._running = False
        self._connected = False
        self.received_proposals: List[SwarmProposal] = []
        self.received_commits: List[str] = []
        self.slashed = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ws is not None

    async def connect(self) -> bool:
        """Connect to the coordinator and send initial handshake."""
        try:
            self.ws = await websockets.connect(self.coordinator_url)
            self._running = True

            handshake = NetworkMessage(
                msg_id=f"hs_{int(time.time()*1000)}",
                msg_type=MessageType.HANDSHAKE,
                sender=self.node_id,
                payload={"weight": self.weight},
            )
            handshake.sign(self.secret)
            await self.ws.send(handshake.to_json())

            # Read handshake ack with timeout
            raw_ack = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
            ack = NetworkMessage.from_json(raw_ack)
            if ack.msg_type == MessageType.HANDSHAKE_ACK:
                self._connected = True
                return True
        except Exception as e:
            logger.warning(f"Connection to coordinator failed: {e}")
            self._connected = False
        return False

    async def disconnect(self) -> None:
        """Disconnect cleanly from coordinator."""
        self._running = False
        self._connected = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    async def send_vote(
        self,
        round_id: str,
        proposal_id: str,
        decision: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Cast and dispatch a signed vote to the coordinator."""
        if not self.ws or not self._connected:
            raise RuntimeError("Cannot vote while disconnected")

        vote_msg = NetworkMessage(
            msg_id=f"vote_{int(time.time()*1000)}",
            msg_type=MessageType.VOTE,
            sender=self.node_id,
            round_id=round_id,
            payload={
                "round_id": round_id,
                "proposal_id": proposal_id,
                "decision": decision,
                "weight": self.weight,
                "reason": reason,
            },
        )
        vote_msg.sign(self.secret)
        await self.ws.send(vote_msg.to_json())

    async def send_heartbeat(self) -> None:
        """Send a heartbeat ping to coordinator."""
        if not self.ws or not self._connected:
            return
        hb = NetworkMessage(
            msg_id=f"hb_{int(time.time()*1000)}",
            msg_type=MessageType.HEARTBEAT,
            sender=self.node_id,
        )
        hb.sign(self.secret)
        await self.ws.send(hb.to_json())

    async def listen_one(self, timeout: Optional[float] = None) -> Optional[NetworkMessage]:
        """Receive and process a single message from coordinator."""
        if not self.ws or not self._connected:
            return None
        try:
            if timeout is not None:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            else:
                raw = await self.ws.recv()
            msg = NetworkMessage.from_json(raw)

            # HMAC verification
            if self.secret and not msg.verify(self.secret):
                return None

            if msg.msg_type == MessageType.PROPOSAL:
                raw_prop = msg.payload.get("proposal", {})
                proposal = deserialize_proposal(raw_prop)
                self.received_proposals.append(proposal)

                # Automatic voting evaluation if callback registered
                round_id = msg.round_id or msg.payload.get("round_id", "")
                if self.vote_callback and round_id:
                    decision, reason = self.vote_callback(proposal)
                    await self.send_vote(
                        round_id=round_id,
                        proposal_id=proposal.proposal_id,
                        decision=decision,
                        reason=reason,
                    )

            elif msg.msg_type == MessageType.COMMIT:
                self.received_commits.append(msg.payload.get("proposal_id", ""))

            elif msg.msg_type == MessageType.SLASH:
                if msg.payload.get("slashed_validator") == self.node_id:
                    self.slashed = True

            return msg
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            return None
