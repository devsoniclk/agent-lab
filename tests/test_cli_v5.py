import json
from pathlib import Path
from click.testing import CliRunner

from agent_lab.cli import main
from agent_lab.zk import ZKTrajectoryEngine


def test_cli_zk_help():
    runner = CliRunner()
    result = runner.invoke(main, ["zk", "--help"])
    assert result.exit_code == 0
    assert "Zero-Knowledge Trajectory Execution & Deliberation Verifier" in result.output
    assert "generate-receipt" in result.output
    assert "verify-receipt" in result.output
    assert "inspect" in result.output


def test_cli_zk_generate_receipt_default():
    runner = CliRunner()
    result = runner.invoke(main, ["zk", "generate-receipt", "--system", "risc0"])
    assert result.exit_code == 0
    assert "ZKVM Trajectory Execution Receipt" in result.output
    assert "RISC0" in result.output
    assert "Action Merkle Root" in result.output


def test_cli_zk_generate_and_verify_flow(tmp_path: Path):
    receipt_file = str(tmp_path / "receipt.json")
    trajectory_file = str(tmp_path / "trajectory.json")

    # Write custom trajectory
    steps = [
        {"step_index": 0, "type": "action", "tool": "search", "content": "finding"},
        {"step_index": 1, "type": "action", "tool": "code", "content": "def run():\n    return 42\n"},
    ]
    Path(trajectory_file).write_text(json.dumps(steps), encoding="utf-8")

    runner = CliRunner()

    # Generate receipt to file
    gen_res = runner.invoke(main, [
        "zk", "generate-receipt",
        "-f", trajectory_file,
        "--out", receipt_file,
        "--require-ast",
    ])
    assert gen_res.exit_code == 0
    assert Path(receipt_file).is_file()

    # Inspect receipt
    inspect_res = runner.invoke(main, ["zk", "inspect", "-f", receipt_file])
    assert inspect_res.exit_code == 0
    assert "ZKVM Receipt Details" in inspect_res.output
    assert "Committed Journal Payload" in inspect_res.output

    # Verify receipt
    verify_res = runner.invoke(main, [
        "zk", "verify-receipt",
        "-f", receipt_file,
        "--require-ast",
    ])
    assert verify_res.exit_code == 0
    assert "VERIFIED VALID" in verify_res.output


def test_cli_zk_verify_failure_tampered(tmp_path: Path):
    receipt_file = str(tmp_path / "tampered_receipt.json")
    steps = [{"tool": "test", "content": "ok"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)
    d = receipt.to_dict()
    d["seal_hex"] = "0x" + "1" * 80  # Tampered seal
    Path(receipt_file).write_text(json.dumps(d), encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(main, ["zk", "verify-receipt", "-f", receipt_file])
    assert res.exit_code == 1
    assert "VERIFICATION FAILED" in res.output


def test_cli_zk_missing_file_errors():
    runner = CliRunner()
    res1 = runner.invoke(main, ["zk", "generate-receipt", "-f", "nonexistent.json"])
    assert res1.exit_code == 1
    assert "File not found" in res1.output

    res2 = runner.invoke(main, ["zk", "verify-receipt", "-f", "nonexistent.json"])
    assert res2.exit_code == 1
    assert "File not found" in res2.output
