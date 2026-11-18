import os
from pathlib import Path
from agent_lab.config import AgentLabConfig


def test_config_defaults(tmp_path: Path):
    cfg = AgentLabConfig(workspace_root=str(tmp_path))
    assert cfg.workspace_root == str(tmp_path)
    assert cfg.storage_dir == ".agentlab"
    assert cfg.consensus_quorum_ratio == 0.67


def test_config_save_and_load(tmp_path: Path):
    cfg = AgentLabConfig(
        workspace_root=str(tmp_path),
        storage_dir=str(tmp_path / ".agentlab"),
        env="testing",
        consensus_quorum_ratio=0.75,
    )
    saved_path = cfg.save()
    assert saved_path.is_file()

    loaded = AgentLabConfig.load(workspace_root=str(tmp_path))
    assert loaded.env == "testing"
    assert loaded.consensus_quorum_ratio == 0.75
