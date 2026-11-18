"""
agent-lab: Unified autonomous AI agent safety orchestrator,
BFT swarm consensus coordinator, and evaluation benchmark harness.
"""

from agent_lab.config import AgentLabConfig
from agent_lab.orchestrator import AgentLabOrchestrator, AgentSession
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmConsensusResult
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

__version__ = "0.2.0"
__all__ = [
    "AgentLabConfig",
    "AgentLabOrchestrator",
    "AgentSession",
    "SwarmCoordinator",
    "SwarmProposal",
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
]
