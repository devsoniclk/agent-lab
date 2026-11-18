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
