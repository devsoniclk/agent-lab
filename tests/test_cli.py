from click.testing import CliRunner
from pathlib import Path
from agent_lab.cli import main


def test_cli_init_and_doctor(tmp_path: Path):
    runner = CliRunner()

    # Test init
    result_init = runner.invoke(main, ["init", "--workspace", str(tmp_path)])
    assert result_init.exit_code == 0
    assert "Initialized Agent-Lab workspace" in result_init.output
    assert (tmp_path / ".agentlab" / "config.json").is_file()

    # Test doctor
    result_doc = runner.invoke(main, ["doctor"])
    assert result_doc.exit_code == 0
    assert "rollback" in result_doc.output
    assert "solana-tx-lens" in result_doc.output
    assert "agent-witness" in result_doc.output


def test_cli_session_start():
    runner = CliRunner()
    result = runner.invoke(main, ["session", "start", "--agent", "CliAgent", "--goal", "CLI Test"])
    assert result.exit_code == 0
    assert "Session ID" in result.output
    assert "CliAgent" in result.output


def test_cli_eval_tools(tmp_path: Path):
    runner = CliRunner()

    # Test eval pass-at-k
    result_eval = runner.invoke(main, ["eval", "pass-at-k", "-n", "10", "-c", "7", "-k", "1"])
    assert result_eval.exit_code == 0
    assert "Pass@1:" in result_eval.output

    # Test ast-check
    test_file = tmp_path / "valid.py"
    test_file.write_text("def compute(): return 100\n", encoding="utf-8")
    result_ast = runner.invoke(main, ["eval", "ast-check", str(test_file), "--entry", "compute"])
    assert result_ast.exit_code == 0
    assert "Code is syntactically valid" in result_ast.output
