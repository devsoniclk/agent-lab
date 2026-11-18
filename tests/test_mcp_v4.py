import json
import pytest
from agent_lab.mcp_server import (
    agentlab_registry_register_agent,
    agentlab_registry_get_agent,
    agentlab_registry_verify_stake,
    agentlab_registry_list_agents,
    agentlab_registry_deposit_stake,
    agentlab_registry_withdraw_stake,
    agentlab_registry_submit_slashing_proof,
    agentlab_registry_export_idl,
)


def test_mcp_registry_registration_and_queries():
    # 1. Register agent
    raw = agentlab_registry_register_agent(
        pubkey="mcp_test_agent_01",
        name="MCP_Validator_1",
        stake_lamports=120_000_000,
        metadata_uri="https://agentlab.io/agent01.json",
    )
    res = json.loads(raw)
    assert res["status"] == "success"
    assert res["agent"]["pubkey"] == "mcp_test_agent_01"

    # 2. Get agent
    raw_get = agentlab_registry_get_agent(pubkey="mcp_test_agent_01")
    res_get = json.loads(raw_get)
    assert res_get["status"] == "success"
    assert res_get["agent"]["name"] == "MCP_Validator_1"

    # 3. Verify stake
    raw_ver = agentlab_registry_verify_stake(pubkey="mcp_test_agent_01")
    res_ver = json.loads(raw_ver)
    assert res_ver["status"] == "success"
    assert res_ver["eligible"] is True

    # 4. List agents
    raw_list = agentlab_registry_list_agents()
    res_list = json.loads(raw_list)
    assert res_list["status"] == "success"
    assert res_list["count"] >= 1


def test_mcp_registry_stake_deposit_and_withdraw():
    # Deposit
    raw_dep = agentlab_registry_deposit_stake(
        pubkey="mcp_test_agent_01",
        amount_lamports=50_000_000,
    )
    res_dep = json.loads(raw_dep)
    assert res_dep["status"] == "success"
    assert res_dep["agent"]["stake_lamports"] == 170_000_000

    # Withdraw
    raw_with = agentlab_registry_withdraw_stake(
        pubkey="mcp_test_agent_01",
        amount_lamports=20_000_000,
    )
    res_with = json.loads(raw_with)
    assert res_with["status"] == "success"
    assert res_with["agent"]["stake_lamports"] == 150_000_000


def test_mcp_registry_slashing_and_idl():
    # Register bad agent
    agentlab_registry_register_agent(
        pubkey="mcp_bad_agent",
        name="MCP_Bad_Validator",
        stake_lamports=200_000_000,
    )

    # Slash
    raw_slash = agentlab_registry_submit_slashing_proof(
        offender_pubkey="mcp_bad_agent",
        reporter_pubkey="mcp_reporter_agent",
        proposal_id="prop_mcp_slash_01",
        vote_a_decision=True,
        vote_b_decision=False,
    )
    res_slash = json.loads(raw_slash)
    assert res_slash["status"] == "success"
    assert res_slash["receipt"]["offender_pubkey"] == "mcp_bad_agent"
    assert res_slash["receipt"]["slashed_lamports"] == 100_000_000

    # Export IDL
    raw_idl = agentlab_registry_export_idl()
    res_idl = json.loads(raw_idl)
    assert res_idl["status"] == "success"
    assert res_idl["idl"]["name"] == "agent_registry"
