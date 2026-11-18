# Changelog

All notable changes to `agent-lab` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-18 - Portfolio Ecosystem Integration & Cross-Repository Unified Fleet Simulation

### Added
- **Portfolio Ecosystem Integration Engine (`agent_lab.portfolio`)**:
  - `PortfolioDiscovery`: Deep auto-discovery of adjacent and installed portfolio packages (`rollback`, `solana-tx-lens`, `agent-witness`, `agent-lab`), resolving versions, runtime availability, file paths, and active capability matrices.
  - `PortfolioDoctor`: End-to-end diagnostic runner executing real health probes across all 4 portfolio repositories:
    - `rollback` (v0.10.0): Verifies transactional micro-checkpoints, mutating file edits, and clean atomic rollbacks with hash equality.
    - `solana-tx-lens` (v0.10.0): Verifies headless instruction execution, compute unit consumption, and dynamic priority fee prediction.
    - `agent-witness` (v0.10.0): Verifies RFC 6962 Merkle tree proofs, leaf verification, and SP1 zkVM deliberation attestation receipts.
    - `agent-lab` (v1.1.0): Runs full fleet diagnostics verifying memory, routing, BFT consensus, ZK verification, and tournament arena.
  - `CrossRepoFleetSimulator`: Unified end-to-end multi-agent execution pipeline orchestrating all 4 repositories:
    1. Workspace micro-checkpointing via `rollback.core.create_session`.
    2. Semantic procedural memory pre-fetching via `agent_lab.memory.CrossAgentMemoryStore`.
    3. Multi-model cascade routing via `agent_lab.router.AdaptiveModelRouter`.
    4. Adversarial corruption simulation and atomic state recovery via `rollback.revert()`, proving 0 lingering corrupted files.
    5. On-chain pre-flight simulation and priority fee estimation via `solana_tx_lens.simulator.SolanaSimulator` and `solana_tx_lens.priority_fee.DynamicFeeOptimizer`.
    6. Deliberation witnessing, Merkle proofs, and SP1 zkVM attestation via `agent_witness.receipt.WitnessEngine` and `agent_witness.zkvm.ZKVMDeliberationEngine`.
    7. Tournament match scoring and Elo updates via `agent_lab.arena.ArenaLeaderboard`.
  - `UnifiedPortfolioReceipt`: Cryptographically sealed audit receipt capturing rollback verification, Solana compute units, witness Merkle root, ZK journal digest, and SHA-256 integrity seal.
  - `PortfolioBenchmarkRunner`: Sequential throughput benchmark achieving > 140 cross-repository simulations/second with sub-6ms average latency.
- **CLI Command Group (`agent-lab portfolio`)**:
  - `agent-lab portfolio status`: Formatted table displaying discovery status, installed versions, and capabilities of all 4 repositories.
  - `agent-lab portfolio doctor`: Deep health diagnostic runner across all 4 repositories with `--json` export.
  - `agent-lab portfolio simulate`: Executes unified cross-repository mission with rich summary cards and verified receipt.
  - `agent-lab portfolio benchmark`: High-throughput benchmark reporting operations/sec and latency percentiles (p50, p95, p99).
- **FastMCP Protocol Suite Expansion (47 tools total)**:
  - Added 4 native MCP tools:
    - `agentlab_portfolio_get_status`: Returns discovery status, versions, and capabilities as JSON.
    - `agentlab_portfolio_run_doctor`: Runs cross-repository diagnostics report.
    - `agentlab_portfolio_simulate_mission`: Executes the cross-repository unified simulation.
    - `agentlab_portfolio_benchmark`: Runs the cross-repository throughput benchmark.
- **Testing & Verification**:
  - Added 25 new tests in `tests/test_portfolio.py`, `tests/test_cli_v11.py`, `tests/test_mcp_v11.py`, and `tests/test_v11_coverage_edges.py`.
  - `agent-lab`: 258/258 passed (100%), 93% total project code coverage, 98% portfolio module coverage.
  - Full portfolio pass rate: **611/611 automated tests passing across all 4 repositories** (`rollback`: 86, `solana-tx-lens`: 137, `agent-witness`: 130, `agent-lab`: 258).

## [1.0.0] - 2026-11-18 - General Availability (GA)

### Added
- **Fleet Diagnostics & Unified Operational Engine (`agent_lab.fleet`)**:
  - `FleetDiagnostics`: Deep self-diagnostics runner verifying operational status, roundtrip latency, and health invariant satisfaction across all 6 core sub-engines (semantic memory, adaptive router, BFT consensus, ZK deliberation, tournament arena, and evaluation harness).
  - `FleetHealthReport` & `FleetSubsystemCheck`: Structured diagnostics reporting model supporting rich terminal tables and JSON exports.
  - `FleetMissionRunner`: Full-lifecycle meta-orchestration pipeline executing end-to-end multi-agent missions combining:
    1. Cross-agent semantic memory pre-fetching and contextual search.
    2. Adaptive multi-model cascade fallback routing with automatic circuit breaker trips on provider 429/500 errors.
    3. Byzantine Fault Tolerant (BFT) deliberation consensus with OCC validation across validator quorums.
    4. Zero-Knowledge execution and deliberation attestation (RISC Zero & Succinct SP1 zkVMs).
    5. Deterministic AST syntax verification and trajectory F1 compliance scoring with loop detection.
    6. Autonomous episodic memory reinforcement and adaptive Elo arena rating adjustments.
  - `FleetExecutionReceipt`: Cryptographically sealed mission receipt containing canonical JSON digest and verifiable SHA-256 seal.
  - `MissionConfig`: Declarative mission specification with custom code proposals, target paths, and failover simulation.
- **Fleet CLI Command Group (`agent-lab fleet`)**:
  - `agent-lab fleet doctor`: Runs instant diagnostic probes across all sub-engines.
  - `agent-lab fleet run-mission`: Executes full-lifecycle multi-agent missions with rich summary panels and JSON outputs.
  - `agent-lab fleet benchmark`: Benchmarks sequential fleet throughput (1,000+ missions/second) and latency.
- **FastMCP Protocol Extensions**:
  - Added 4 new native tools: `agentlab_fleet_run_diagnostics`, `agentlab_fleet_run_mission`, `agentlab_fleet_benchmark`, `agentlab_fleet_verify_receipt` (expanding the FastMCP suite to 45 tools).
- **Expanded Test Suite**:
  - Added 24 new unit, CLI, MCP, and edge-case integration tests across `tests/test_fleet.py`, `tests/test_cli_v10.py`, `tests/test_mcp_v10.py`, and `tests/test_v10_coverage_edges.py` (233 total tests, 100% pass rate, 92% code coverage, 98% fleet module coverage).
- **Production Package Distribution**:
  - Upgraded PyPI packaging metadata to `Development Status :: 5 - Production/Stable`.
  - Built production wheel `agent_lab-1.0.0-py3-none-any.whl` and source distribution `agent_lab-1.0.0.tar.gz`.

## [0.9.0] - 2026-11-18

### Added
- **Multi-Agent Tournament Arena & Head-to-Head Evaluation Engine (`agent_lab.arena`)**:
  - `EloRatingSystem`: Adaptive multi-agent Elo rating engine with dynamic K-factors (provisional vs. established), logistic win probability expectation ($E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$), and Wilson score confidence intervals.
  - `BradleyTerryEstimator`: Iterative Luce / Bradley-Terry maximum likelihood skill estimation solving for latent capability parameters from head-to-head match matrices.
  - `AgentMatchScore` & `MatchOutcome`: Unified composite scoring metric balancing task pass rate (50%), syntax AST validity (20%), trajectory rule compliance (15%), and token efficiency (15%).
  - `ArenaLeaderboard`: Persistent state and rankings tracker recording Elo scores, match histories, win/loss/draw records, and win-rate Wilson confidence intervals with full JSON serialization.
- **Collaborative Swarm Workflow Benchmark Harness (`SwarmWorkflowHarness`)**:
  - Benchmarks end-to-end multi-agent collaborative DAG workflows (e.g. Specification -> Codegen -> Review -> BFT Consensus).
  - Evaluates stage completion, latency per agent, token consumption, BFT consensus rounds, and coordination efficiency.
- **Arena CLI Command Group (`agent-lab arena`)**:
  - `agent-lab arena match`: Runs head-to-head evaluation match between two agents and updates Elo ratings.
  - `agent-lab arena tournament`: Executes multi-agent round-robin tournament across all participant pairs.
  - `agent-lab arena ratings`: Displays current Elo leaderboard standings and win-rate statistics.
  - `agent-lab arena benchmark-swarm`: Benchmarks collaborative multi-agent DAG workflow execution.
  - `agent-lab arena export` & `import`: Manages file-based JSON leaderboard backups and restoration.
- **FastMCP Protocol Extensions**:
  - Added 4 new native tools: `agentlab_arena_run_match`, `agentlab_arena_get_ratings`, `agentlab_arena_benchmark_swarm`, `agentlab_arena_export_ratings` (expanding the FastMCP suite to 41 tools).
- **Expanded Test Suite**:
  - Added 20 new unit and integration tests across `tests/test_arena.py`, `tests/test_cli_v9.py`, `tests/test_mcp_v9.py`, and `tests/test_v9_coverage_edges.py` (209 total tests, 100% pass rate, 92% code coverage).

## [0.8.0] - 2026-11-18

### Added
- **Cross-Agent Semantic Memory Store (`agent_lab.memory.CrossAgentMemoryStore`)**:
  - Thread-safe, multi-namespace cross-agent semantic memory store with vector indexing and memory partitioning.
  - Granular access control boundaries (`PRIVATE`, `SWARM`, `PUBLIC`) preventing inter-agent data leaks while enabling coordinated swarm context sharing.
  - Proactive and reactive memory eviction strategies (`DECAY`, `LRU`, `LIF`) to maintain strict partition capacity bounds.
  - Serialization engine supporting JSON backups (`export_json`) and incremental restores (`import_json`).
- **Pure-Python Deterministic Vector Math & Semantic Embedder (`SemanticTextEmbedder`)**:
  - Zero-dependency algebraic vector operations: `cosine_similarity`, `dot_product`, `euclidean_distance`, `normalize_vector`, `l2_norm`.
  - Multi-ngram feature hashing trick with sublinear term-frequency scaling ($1 + \ln(\text{tf})$) projecting arbitrary text into a fixed-dimensional unit hypersphere.
  - Cross-platform deterministic hashing via `hashlib.sha256`.
- **Hybrid Lexical & Dense Vector Retrieval (`CrossAgentMemoryStore.search`)**:
  - Harmonizes dense vector cosine similarity with token-overlap lexical scoring via tunable `hybrid_alpha`.
  - Continuous exponential temporal recency decay modeling ($e^{-\lambda \Delta t}$) based on configurable half-lives.
  - Composite multi-attribute ranking integrating similarity, lexical keywords, temporal recency, and innate importance.
- **Memory CLI Command Group (`agent-lab memory`)**:
  - `agent-lab memory store`: Stores episodic, semantic, or procedural memory items with tag indexing and importance weighting.
  - `agent-lab memory search`: Natural language hybrid vector/keyword query search with rich terminal table output and JSON export.
  - `agent-lab memory stats`: Summarizes total memory volume, namespace partitions, taxonomy counts, and embedding dimensions.
  - `agent-lab memory evict`: Prunes low-value or expired memories under `DECAY`, `LRU`, or `LIF` policies.
  - `agent-lab memory export` & `import`: Manages file-based JSON backups and restoration.
- **FastMCP Protocol Extensions**:
  - Added 4 new native tools: `agentlab_memory_store`, `agentlab_memory_search`, `agentlab_memory_get_stats`, `agentlab_memory_clear_or_evict` (expanding the FastMCP suite to 37 tools).
- **Expanded Test Suite**:
  - Added 17 new unit and integration tests across `tests/test_memory.py`, `tests/test_cli_v8.py`, `tests/test_mcp_v8.py`, and `tests/test_v8_coverage_edges.py` (189 total tests, 100% pass rate, 93% code coverage).

## [0.7.0] - 2026-11-18

### Added
- **Adaptive Dynamic Token Rate-Limiter & Sliding-Window Token Governor (`TokenGovernor`)**:
  - High-throughput rolling 60-second sliding-window request (RPM) and token (TPM) governor.
  - Priority-tiered capacity allowances (`CRITICAL` 1.20x burst, `HIGH` 1.00x, `NORMAL` 0.85x, `BACKGROUND` 0.65x).
  - Dynamic concurrency slot tracking, reservation, and safe release preventing provider saturation.
  - Granular per-agent token consumption budgets and quota enforcement (`AgentQuota`, `QuotaExceededError`).
- **3-State Provider Circuit Breakers (`ModelCircuitBreaker`)**:
  - Full state-machine transitions: `CLOSED` (healthy) -> `OPEN` (fail-fast) -> `HALF_OPEN` (canary testing).
  - Cooldown recovery with configurable randomized jitter mitigating thundering herd problems.
  - Latency SLA tracking: automatically trips circuit if rolling 5-call average latency exceeds SLA threshold.
  - Fail-fast trip triggers on consecutive failures, canary probe re-trips, and HTTP 429 rate limit errors.
- **Cross-Model Cascade Fallback Router (`AdaptiveModelRouter`)**:
  - Intelligent multi-model fallback cascade routing (`code-generation`, `reasoning-deliberation`, `fast-synthesis`).
  - Cryptographic and operational execution receipts (`RoutingExecutionReceipt`) tracking attempt histories, latency, token consumption, and accurate cost calculations.
  - Telemetry recording integration with `TelemetryCollector.record_routing_event`.
- **Router CLI Command Group (`agent-lab router`)**:
  - `agent-lab router status`: Displays rich terminal status of fleet circuit breakers, RPM/TPM utilization, and cascade configurations with JSON export option.
  - `agent-lab router route`: Dispatches prompts through adaptive fallback cascades with priority flags and simulated failover testing.
  - `agent-lab router reset-circuit`: Resets individual or all model circuit breakers to CLOSED.
  - `agent-lab router benchmark-resilience`: Simulates high-concurrency swarm workloads under synthetic provider faults to measure resilience rates and failover latency.
- **FastMCP Protocol Extensions**:
  - Added 4 new native tools: `agentlab_router_route`, `agentlab_router_get_fleet_status`, `agentlab_router_reset_circuit`, `agentlab_router_benchmark_resilience` (expanding FastMCP tool suite to 33 tools).
- **Expanded Test Suite**:
  - Added 29 new unit and integration tests across `tests/test_router.py`, `tests/test_cli_v7.py`, `tests/test_mcp_v7.py`, and `tests/test_v7_coverage_edges.py` (172 total tests, 100% pass rate, 93% code coverage).

## [0.6.0] - 2026-11-18

### Added
- **Cross-Agent Unified Benchmark Leaderboard (`agent_lab.leaderboard`)**:
  - `LeaderboardEngine`: Computes normalized multi-attribute composite scores (0–100) weighting Pass@1 accuracy (35%), tool selection trajectory F1 (25%), latency efficiency (15%), task token cost (15%), and safety/ZK proof compliance (10%).
  - Engineering Tier Classification: Automatically assigns competitive tiers (`S` >= 90.0, `A` >= 80.0, `B` >= 70.0, `C` >= 60.0, `D` < 60.0).
  - Wilson Score 95% Confidence Intervals: Statistically rigorous confidence bounds for agent pass rates (`pass_at_1_ci`).
  - Non-Dominated Pareto Frontier Analysis (`compute_pareto_frontier`): Identifies optimal model trade-offs minimizing task execution costs while maximizing composite benchmark scores.
- **Multi-Model Inference & Token Economics Simulator (`TokenEconomicsSimulator`)**:
  - `ModelPricing`: Built-in catalog covering frontier and open-weight models (`claude-3-5-sonnet`, `claude-3-5-haiku`, `gpt-4o`, `gpt-4o-mini`, `deepseek-coder-v2`, `qwen-2.5-coder-32b`, `llama-3.3-70b`, `gemini-1.5-pro`, `gemini-1.5-flash`).
  - Swarm Workload Simulator: Projects token consumption, daily/monthly operational expenses, and per-model traffic distribution across multi-agent swarms.
  - Hybrid Routing Downgrade Analysis: Quantifies quantitative monthly dollar and percentage savings achievable by intelligently routing simple tasks to low-cost or local models instead of 100% flagship models.
- **Interactive Dark-Theme HTML & SVG Dashboard (`LeaderboardDashboard`)**:
  - Standalone, responsive HTML5 dashboard with CSS glassmorphism and executive KPI summary cards.
  - Embedded SVG Horizontal Bar Chart: Visualizes composite scores across evaluated models with tier-based color gradients.
  - Embedded SVG Pareto Frontier Scatter Plot: Visualizes cost-versus-quality trade-offs with 2D coordinate gridlines and connected Pareto optimal frontier curves.
  - Markdown and JSON export capabilities for automated CI/CD and release publishing.
- **Telemetry Token & Latency Integration (`agent_lab.telemetry`)**:
  - Added `record_inference` to track prompt/completion tokens, latency in ms, and cumulative USD costs across models.
  - Added inference tracking to terminal dashboard and Markdown reports.
- **Leaderboard CLI Command Group (`agent-lab leaderboard`)**:
  - `agent-lab leaderboard generate`: Generates formatted rich terminal tables, JSON, or Markdown from benchmark results.
  - `agent-lab leaderboard render-html`: Exports standalone HTML dashboard with embedded SVG charts.
  - `agent-lab leaderboard simulate-cost`: Simulates multi-model swarm token consumption and cost optimization.
  - `agent-lab leaderboard export-json`: Exports structured JSON schema of ranked leaderboard entries.
- **FastMCP Leaderboard Tools**:
  - Added 4 new tools: `agentlab_leaderboard_generate`, `agentlab_leaderboard_pareto_analysis`, `agentlab_leaderboard_cost_simulator`, `agentlab_leaderboard_export_html` (expanding the FastMCP tool suite to 29 tools).
- **Expanded Test Suite**:
  - Added 30 new unit and integration tests across `tests/test_leaderboard.py`, `tests/test_cli_v6.py`, `tests/test_mcp_v6.py`, and `tests/test_v6_coverage_edges.py` (143 total tests, 100% pass rate, 93% code coverage).

## [0.5.0] - 2026-11-18

### Added
- **Zero-Knowledge Trajectory Execution & Deliberation Verifier (`agent_lab.zk`)**:
  - `ZKTrajectoryEngine`: Synthesizes and cryptographically audits succinct zero-knowledge execution receipts for agent deliberation trajectories and tool-use traces across RISC Zero (`risc0`) guest ELF and Succinct SP1 (`sp1`) zkVM architectures.
  - `ZKTrajectoryReceipt`: Verifiable zkVM execution receipt binding session ID, step count, tool call counts, action Merkle tree root, AST code safety verification flags, and deterministic cryptographic seal.
  - `ZKPolicyConstraints`: Security bounds enforced by the zkVM guest including maximum execution steps, allowed tool allowlists, forbidden tool denylists, mandatory AST syntax verification on code emissions, and minimum trajectory F1 score thresholds.
  - `ZKVerificationReport`: Granular cryptographic audit report validating journal digest integrity, program ID allowlist compliance, action Merkle root match, policy digest match, execution status code zero, and deterministic cryptographic seal signature binding.
- **ZK-Guarded Swarm Consensus Integration**:
  - `SwarmCoordinator`: Added `require_zk_proof` flag and `zk_constraints` policy bounds. Proposals submitted to BFT swarm rounds can require a valid, verified `ZKTrajectoryReceipt`. Rejects missing, tampered, or invalid zk receipts with atomic safety rollbacks.
  - `DistributedCoordinator` & `DistributedSwarmNode`: Network proposals serialized across distributed WebSocket swarms carry `zk_receipt` payloads for remote zero-knowledge consensus validation.
- **ZK CLI Command Group (`agent-lab zk`)**:
  - `agent-lab zk generate-receipt`: Generate succinct zkVM execution receipts for executed or synthetic benchmark trajectories with customizable policy constraints and target systems (`--system risc0|sp1`).
  - `agent-lab zk verify-receipt`: Cryptographically verify a zkVM trajectory receipt against policy constraints, program ID allowlists, and expected Merkle roots with detailed rich terminal audit tables.
  - `agent-lab zk inspect`: Inspect committed journal payloads, action Merkle roots, seals, and metadata.
- **FastMCP Zero-Knowledge Tools**:
  - Added 4 new tools: `agentlab_zk_generate_receipt`, `agentlab_zk_verify_receipt`, `agentlab_zk_audit_trajectory`, `agentlab_zk_policy_digest` (expanding the FastMCP tool suite to 25 tools).
- **Expanded Test Suite**:
  - Added 31 new unit and integration tests across `tests/test_zk.py`, `tests/test_zk_consensus.py`, `tests/test_cli_v5.py`, and `tests/test_mcp_v5.py` (113 total tests, 100% pass rate, 93% code coverage).

## [0.4.0] - 2026-11-18

### Added
- **Solana Anchor On-Chain Agent Registry (`AnchorAgentRegistryClient`)**: Decentralized identity, deterministic Program Derived Address (PDA) derivation (`["agent", pubkey]`), and Anchor discriminator calculations (`derive_anchor_discriminator`) for autonomous agents participating in governance and consensus.
- **Economic Staking & Collateral Requirements**: Minimum stake threshold validation (`DEFAULT_MIN_STAKE_LAMPORTS = 0.1 SOL`), stake deposits, withdrawals, and automatic lifecycle status transitions (`ACTIVE`, `INACTIVE`, `SLASHED`, `JAILED`).
- **Cryptographic Equivocation Slashing Engine**: Detects contradictory double-voting by rogue validators, calculates deterministic penalty collateral deductions (5000 bps / 50%), executes 50% collateral burn and 50% reporter bounty distribution, deducts reputation penalties (-2500 bps), and moves offending agents to `JAILED` status.
- **BFT Consensus & Swarm Registry Integration**: Integrated `SwarmCoordinator` with on-chain registry verification. Automatically filters out unregistered or under-staked nodes from voting weight, and triggers automated on-chain slashing submissions when conflicting votes occur.
- **Complete Anchor Program IDL (`ANCHOR_AGENT_REGISTRY_IDL`)**: Production-ready Anchor specification defining instructions (`initialize_registry`, `register_agent`, `deposit_stake`, `withdraw_stake`, `submit_slashing_proof`), accounts (`AgentRegistryState`, `AgentAccount`), and error codes (6000-6004).
- **Registry CLI Commands (`agent-lab registry`)**: Added `init`, `register`, `stake`, `info`, `list`, `slash-proof`, and `export-idl` subcommands with rich interactive terminal tables and panels.
- **FastMCP Registry Tools**: Added 7 new tools (`agentlab_registry_register_agent`, `agentlab_registry_get_agent`, `agentlab_registry_verify_stake`, `agentlab_registry_list_agents`, `agentlab_registry_deposit_stake`, `agentlab_registry_withdraw_stake`, `agentlab_registry_submit_slashing_proof`, `agentlab_registry_export_idl`).
- **Expanded Test Suite**: Added 20 new tests in `test_registry.py`, `test_cli_v4.py`, `test_mcp_v4.py`, and `test_v4_coverage_edges.py` (82 total tests, 100% pass rate, 93% code coverage).

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
