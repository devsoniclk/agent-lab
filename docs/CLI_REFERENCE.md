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
