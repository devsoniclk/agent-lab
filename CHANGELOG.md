# Changelog

All notable changes to `agent-lab` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-18

### Added
- **Distributed WebSocket BFT Consensus Transport (`DistributedCoordinator` & `DistributedSwarmNode`)**: Full-duplex asynchronous transport allowing multi-host autonomous worker swarms to participate in distributed Byzantine Fault Tolerant consensus rounds across local networks and remote clusters.
- **Cryptographic HMAC-SHA256 Authentication**: Canonical message envelope (`NetworkMessage`) with deterministic serialization and HMAC signing, preventing unauthorized injection, tampering, and message forgery across distributed swarms.
- **Dynamic Peer Lifecycle & Health Management**: Peer registration (`HANDSHAKE` / `HANDSHAKE_ACK`), keepalive pings (`HEARTBEAT` / `HEARTBEAT_ACK`), peer disconnection recovery, and state synchronization (`SYNC_REQ` / `SYNC_RESP`).
- **Distributed Quorum & Slashing Broadcast**: Coordinator broadcasts proposals, aggregates asynchronous remote votes, checks OCC conflict boundaries, slashes equivocating Byzantine nodes, commits atomic disk mutations, and broadcasts `COMMIT` / `REJECT` and `SLASH` notifications across the network.
- **Distributed Swarm CLI (`agent-lab swarm`)**: Added `swarm serve` to launch the coordinator hub, `swarm worker` to connect validator nodes, and `swarm broadcast-proposal` to dispatch proposals and verify remote consensus from the CLI.
- **FastMCP Distributed Swarm Tools**: Added `agentlab_swarm_status` and `agentlab_swarm_broadcast_proposal` tools.
- **Expanded Test Suite**: Added 16 new unit and integration tests covering distributed consensus, network authentication, handshake/heartbeat loops, quorum verification, rejection rollbacks, slashing propagation, and CLI/MCP flows (62 total tests, 100% pass rate, 93% code coverage).

## [0.2.0] - 2026-11-18

### Added
- **Tool-Use Trajectory Evaluator (`TrajectoryEvaluator`)**: Evaluates agent tool calling sequences for multiset precision, recall, and F1 score against expected tool sets. Implements sequence ordering constraints (e.g., prerequisite verification before dependent execution), loop/repetition detection (flagging 3+ consecutive identical calls), and execution step efficiency tracking.
- **Statistical Confidence Intervals (`StatisticalInterval`)**: Added closed-form Wilson score binomial confidence intervals for stochastic evaluation metrics and pass rates with zero external dependencies.
- **Cross-Agent Telemetry & Interactive Terminal Dashboard (`TelemetryCollector`)**: Real-time operational tracking of sessions, micro-checkpoints, rollbacks, BFT consensus votes, slashed equivocations, Solana simulations, and trajectory F1 scores. Features interactive Rich terminal dashboards (`agent-lab telemetry show`) and Markdown/JSON export (`agent-lab telemetry export`).
- **Standardized HumanEval Mini Benchmark Suite**: Added `benchmarks/humaneval_mini.json` comprising 10 curated algorithm, parsing, geometry, and statistics coding tasks with complete prompts, entry points, assertions, and canonical solutions.
- **Benchmark Suite Runner (`BenchmarkHarness.run_suite`)**: Automated execution and statistical scoring across benchmark suites with confidence intervals and task-level diagnostics.
- **Expanded CLI & MCP Tools**: Added `eval trajectory`, `eval suite`, `telemetry show`, `telemetry export` commands and exposed `agentlab_eval_trajectory`, `agentlab_telemetry_get_summary`, and `agentlab_benchmark_run_suite` via FastMCP.
- **PyPI Distribution Ready**: Configured and validated production wheel and source distribution packaging (`agent_lab-0.2.0-py3-none-any.whl`).
- **Test Suite Expansion**: Added 22 new tests covering trajectory evaluation, loop detection, telemetry, dashboards, and CLI commands (46 total tests, 100% pass rate, 97% code coverage).

## [0.1.0] - 2026-11-18

### Added
- **Core Meta-Orchestrator (`AgentLabOrchestrator`)**: Unified autonomous agent execution session manager coordinating transactional filesystem rollback (`rollback`), pre-flight Solana transaction simulation (`solana-tx-lens`), and cryptographic deliberation receipts (`agent-witness`).
- **BFT Swarm Consensus Coordinator (`SwarmCoordinator`)**: Multi-agent 2-phase Byzantine Fault Tolerant consensus engine with Optimistic Concurrency Control (OCC), read-set/write-set tracking, and automated atomic rollback on quorum rejection or equivocation.
- **Autonomous Benchmark & Evaluation Harness (`BenchmarkHarness`)**: Pass@k evaluation pipeline implementing Chen et al.'s unbiased estimator, deterministic AST syntax & lint verification, sandboxed code execution, and regression tracking.
- **FastMCP Server**: Comprehensive MCP server exposing 10 high-level tools for transactional agent sessions, checkpoints, BFT consensus rounds, simulation pre-flight, and benchmark execution.
- **Production CLI (`agent-lab` / `agentlab`)**: Commands for `init`, `session`, `checkpoint`, `rollback`, `consensus`, `eval`, `verify`, and `doctor`.
- **CI/CD Automation**: GitHub Actions workflows for continuous integration and automated security scanning.
