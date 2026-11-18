"""
agent-lab: Unified autonomous AI agent safety orchestrator,
BFT swarm consensus coordinator, and evaluation benchmark harness.
"""

from agent_lab.config import AgentLabConfig
from agent_lab.orchestrator import AgentLabOrchestrator, AgentSession
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmConsensusResult, SwarmVote
from agent_lab.eval import (
    BenchmarkHarness,
    PassAtKEstimator,
    ASTVerifier,
    RegressionTracker,
    TrajectoryEvaluator,
    StatisticalInterval,
    BenchmarkTask,
)
from agent_lab.telemetry import TelemetryCollector, TelemetryMetrics, TelemetryEvent
from agent_lab.distributed import (
    DistributedCoordinator,
    DistributedSwarmNode,
    NetworkMessage,
    MessageType,
    PeerInfo,
)

__version__ = "0.3.0"
__all__ = [
    "AgentLabConfig",
    "AgentLabOrchestrator",
    "AgentSession",
    "SwarmCoordinator",
    "SwarmProposal",
    "SwarmVote",
    "SwarmConsensusResult",
    "BenchmarkHarness",
    "PassAtKEstimator",
    "ASTVerifier",
    "RegressionTracker",
    "TrajectoryEvaluator",
    "StatisticalInterval",
    "BenchmarkTask",
    "TelemetryCollector",
    "TelemetryMetrics",
    "TelemetryEvent",
    "DistributedCoordinator",
    "DistributedSwarmNode",
    "NetworkMessage",
    "MessageType",
    "PeerInfo",
]
