# Contributing to agent-lab

We welcome contributions to `agent-lab`!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/devsoniclk/agent-lab.git
   cd agent-lab
   ```

2. Set up a virtual environment and install development dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the test suite:
   ```bash
   pytest tests/
   ```

## Pull Request Guidelines

- Ensure 100% of automated tests pass (`pytest`).
- Maintain type annotations and strict modular design.
- Include unit tests for any new features or bug fixes.
- Follow conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`).
