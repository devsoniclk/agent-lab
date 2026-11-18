"""
agent_lab.config: Unified configuration management for Agent-Lab workspaces.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AgentLabConfig:
    """Configuration schema for an Agent-Lab workspace."""

    workspace_root: str = "."
    storage_dir: str = ".agentlab"
    env: str = "development"
    solana_rpc: str = "https://api.mainnet-beta.solana.com"
    consensus_quorum_ratio: float = 0.67
    bft_timeout_ms: int = 5000
    eval_timeout_secs: int = 30
    eval_max_workers: int = 4
    enable_rollback: bool = True
    enable_witness: bool = True
    enable_solana_lens: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, workspace_root: Optional[str] = None) -> AgentLabConfig:
        """Load configuration from .agentlab/config.json or environment."""
        root = Path(workspace_root or os.environ.get("AGENT_LAB_WORKSPACE", ".")).resolve()
        storage = root / os.environ.get("AGENT_LAB_STORAGE_DIR", ".agentlab")
        config_path = storage / "config.json"

        data: Dict[str, Any] = {
            "workspace_root": str(root),
            "storage_dir": str(storage),
            "env": os.environ.get("AGENT_LAB_ENV", "development"),
            "solana_rpc": os.environ.get("AGENT_LAB_SOLANA_RPC", "https://api.mainnet-beta.solana.com"),
        }

        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    data.update(file_data)
            except Exception:
                pass

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, target_dir: Optional[str] = None) -> Path:
        """Persist configuration to .agentlab/config.json."""
        root = Path(self.workspace_root).resolve()
        if target_dir:
            dest_dir = Path(target_dir).resolve()
        else:
            dest_dir = (root / self.storage_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        config_path = dest_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

        return config_path
