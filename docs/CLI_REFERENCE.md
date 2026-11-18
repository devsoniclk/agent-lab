# CLI Reference Guide

The `agent-lab` command-line utility provides tools for managing agent sessions, BFT consensus rounds, and code generation evaluation.

## Global Options

```bash
agent-lab --help
agent-lab --version
```

## Commands

### `agent-lab init`
Initialize an Agent-Lab configuration inside the current directory or a specified workspace.

```bash
agent-lab init [--workspace <path>]
```

### `agent-lab doctor`
Inspect local integrations (`rollback`, `solana-tx-lens`, `agent-witness`) and verify environment health.

```bash
agent-lab doctor
```

### `agent-lab session start`
Start a new transactional and attested session for an agent.

```bash
agent-lab session start --agent <name> --goal "<objective>"
```

### `agent-lab eval pass-at-k`
Calculate unbiased Pass@k metric using the Chen et al. (2021) formula.

```bash
agent-lab eval pass-at-k -n <total_samples> -c <correct_samples> -k <k_value>
```

### `agent-lab eval ast-check`
Verify syntax, calculate cyclomatic complexity, and check entrypoint existence for a Python file.

```bash
agent-lab eval ast-check <file_path> [--entry <function_name>]
```

### `agent-lab eval trajectory`
Evaluate multi-step agent tool trajectories for precision, recall, tool F1 score, loop detection, and step efficiency.

```bash
agent-lab eval trajectory --trajectory <trajectory_json> --expected "<tool1,tool2>" [--max-steps <int>]
```

### `agent-lab eval suite`
Execute an entire coding benchmark suite using canonical evaluation and Wilson score confidence intervals.

```bash
agent-lab eval suite --tasks <tasks_json>
```

### `agent-lab telemetry show`
Display real-time interactive terminal telemetry dashboard tracking rollbacks, consensus rounds, benchmark pass rates, and tool-use F1 scores.

```bash
agent-lab telemetry show [--path <telemetry_file>]
```

### `agent-lab telemetry export`
Export aggregated cross-agent telemetry metrics to Markdown or JSON.

```bash
agent-lab telemetry export [--format markdown|json] [--output <file_path>]
```

### `agent-lab eval trajectory`
Evaluate an agent tool execution trajectory JSON log for tool precision, recall, F1 score, loop detection, and sequence validity.

```bash
agent-lab eval trajectory --trajectory <trajectory.json> --expected "<tool1>,<tool2>" [--max-steps <int>]
```

### `agent-lab eval suite`
Run an entire automated coding benchmark suite with canonical verification and 95% Wilson score confidence interval calculation.

```bash
agent-lab eval suite --tasks <tasks.json>
```

### `agent-lab telemetry show`
Launch the real-time interactive terminal dashboard displaying cross-agent lifecycle metrics, rollbacks, consensus stats, and evaluation pass rates.

```bash
agent-lab telemetry show [--path <telemetry.json>]
```

### `agent-lab telemetry export`
Export cross-agent operational telemetry to Markdown or structured JSON.

```bash
agent-lab telemetry export [--format markdown|json] [--output <path>]
```

### `agent-lab registry init`
Initialize a Solana Anchor agent registry configuration file.

```bash
agent-lab registry init [--storage <path>] [--min-stake <lamports>] [--penalty-bps <bps>]
```

### `agent-lab registry register`
Register an autonomous agent validator with on-chain collateral stake.

```bash
agent-lab registry register --pubkey <pubkey> --name <name> [--stake <lamports>] [--metadata <uri>] [--storage <path>]
```

### `agent-lab registry stake`
Deposit additional SOL stake into an agent validator account.

```bash
agent-lab registry stake --pubkey <pubkey> --amount <lamports> [--storage <path>]
```

### `agent-lab registry info`
Query an agent validator's on-chain account, stake, reputation, and BFT consensus eligibility.

```bash
agent-lab registry info --pubkey <pubkey> [--storage <path>]
```

### `agent-lab registry list`
List all registered agents and their stake/reputation metrics in an interactive terminal table.

```bash
agent-lab registry list [--storage <path>]
```

### `agent-lab registry slash-proof`
Submit a cryptographic equivocation proof to slash offender collateral and jail Byzantine nodes.

```bash
agent-lab registry slash-proof --offender <pubkey> --reporter <pubkey> --proposal <id> --vote-a <decision> --vote-b <decision> [--storage <path>]
```

### `agent-lab registry export-idl`
Export the production-ready Anchor Program IDL specification for agent registry.

```bash
agent-lab registry export-idl [--out <path>]
```

### `agent-lab zk generate-receipt`
Synthesize a verifiable zero-knowledge virtual machine (zkVM) execution receipt for agent deliberation trajectories (supporting RISC Zero and Succinct SP1).

```bash
agent-lab zk generate-receipt [--trajectory-file <path>] [--session-id <id>] [--system risc0|sp1] [--max-steps <int>] [--require-ast] [--out <path>]
```

### `agent-lab zk verify-receipt`
Cryptographically audit and verify a zkVM trajectory receipt against policy constraints, program ID allowlists, and expected Merkle roots.

```bash
agent-lab zk verify-receipt -f <receipt.json> [--allow-program-id <id>] [--require-ast] [--expected-root <root_hash>]
```

### `agent-lab zk inspect`
Inspect the full committed journal payload, action Merkle tree root, seal, and metadata of a zkVM execution receipt.

```bash
agent-lab zk inspect -f <receipt.json>
```

### `agent-lab leaderboard generate`
Generate ranked cross-agent benchmark leaderboard with multi-attribute composite scores (0–100), Wilson 95% confidence intervals, and Pareto optimal tagging.

```bash
agent-lab leaderboard generate [--eval-file <path>] [--sample] [--output-format table|json|markdown] [--output-file <path>]
```

### `agent-lab leaderboard render-html`
Export a standalone, responsive dark-theme HTML report with embedded SVG horizontal bar charts (Composite Rankings) and 2D scatter plots (Cost vs Quality Pareto Frontier).

```bash
agent-lab leaderboard render-html [--eval-file <path>] [--output-file <path>] [--title "<text>"] [--simulate-economics/--no-simulate-economics]
```

### `agent-lab leaderboard simulate-cost`
Simulate multi-agent swarm token economics, per-model traffic distribution, and quantitative hybrid routing downgrade savings over a multi-day horizon.

```bash
agent-lab leaderboard simulate-cost [--tasks-per-day <int>] [--days <int>] [--avg-prompt-tokens <int>] [--avg-completion-tokens <int>]
```

### `agent-lab leaderboard export-json`
Export structured JSON schema of ranked leaderboard entries for CI/CD pipelines and external dashboards.

```bash
agent-lab leaderboard export-json [--eval-file <path>] -o <output.json>
```

### `agent-lab router status`
Display real-time circuit breaker states, sliding-window token utilization (RPM/TPM), active concurrency, and configured cross-model fallback cascades.

```bash
agent-lab router status [--json-out <path>]
```

### `agent-lab router route`
Route an LLM task through adaptive fallback cascade with rate-limiting, priority reservation, and automatic circuit breaker failover.

```bash
agent-lab router route [--prompt "<text>"] [--route <route_name>] [--priority CRITICAL|HIGH|NORMAL|BACKGROUND] [--agent <agent_id>] [--tokens <int>] [--simulate-fail-primary]
```

### `agent-lab router reset-circuit`
Reset tripped model circuit breakers back to CLOSED operational state.

```bash
agent-lab router reset-circuit [--model <model_id>] [--all]
```

### `agent-lab router benchmark-resilience`
Benchmark swarm routing resilience, zero-downtime failover rate, and end-to-end latency under synthetic HTTP 429 rate limit faults.

```bash
agent-lab router benchmark-resilience [--requests <int>] [--fault-rate <float>] [--route <route_name>]
```
