# Changelog

All notable changes to `agent-lab` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-18

### Added
- **Core Meta-Orchestrator (`AgentLabOrchestrator`)**: Unified autonomous agent execution session manager coordinating transactional filesystem rollback (`rollback`), pre-flight Solana transaction simulation (`solana-tx-lens`), and cryptographic deliberation receipts (`agent-witness`).
- **BFT Swarm Consensus Coordinator (`SwarmCoordinator`)**: Multi-agent 2-phase Byzantine Fault Tolerant consensus engine with Optimistic Concurrency Control (OCC), read-set/write-set tracking, and automated atomic rollback on quorum rejection or equivocation.
- **Autonomous Benchmark & Evaluation Harness (`BenchmarkHarness`)**: Pass@k evaluation pipeline implementing Chen et al.'s unbiased estimator, deterministic AST syntax & lint verification, sandboxed code execution, and regression tracking.
- **FastMCP Server**: Comprehensive MCP server exposing 10 high-level tools for transactional agent sessions, checkpoints, BFT consensus rounds, simulation pre-flight, and benchmark execution.
- **Production CLI (`agent-lab` / `agentlab`)**: Commands for `init`, `session`, `checkpoint`, `rollback`, `consensus`, `eval`, `verify`, and `doctor`.
- **CI/CD Automation**: GitHub Actions workflows for continuous integration and automated security scanning.
