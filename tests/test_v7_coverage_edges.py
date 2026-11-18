"""
Edge cases and coverage optimization tests for Agent-Lab v0.7.0 router module.
"""

from __future__ import annotations

import json
import pytest

from agent_lab.router import (
    AdaptiveModelRouter,
    AgentPriority,
    AgentQuota,
    CircuitBreakerPolicy,
    CircuitState,
    FallbackRoute,
    ModelCircuitBreaker,
    QuotaExceededError,
    RateLimitPolicy,
    RoutingAttempt,
    RoutingExecutionReceipt,
    TokenGovernor,
    RouterError,
    RateLimitExceededError,
    CircuitBreakerOpenError,
    CascadeExhaustedError,
)
from agent_lab.telemetry import TelemetryCollector


def test_router_exceptions_hierarchy():
    err = RateLimitExceededError("model-x", "rate exceeded", 5.5)
    assert isinstance(err, RouterError)
    assert err.retry_after_seconds == 5.5
    assert "model-x" in str(err)

    cb_err = CircuitBreakerOpenError("model-y", "tripped")
    assert isinstance(cb_err, RouterError)
    assert "model-y" in str(cb_err)

    q_err = QuotaExceededError("agent-z", 1000, 1500)
    assert isinstance(q_err, RouterError)
    assert q_err.budget == 1000
    assert q_err.consumed == 1500

    c_err = CascadeExhaustedError("primary-m", ["m1", "m2"], "all models down")
    assert isinstance(c_err, RouterError)


def test_routing_receipt_and_attempt_serialization():
    att = RoutingAttempt(
        attempt_index=1,
        model_id="gemini-1.5-flash",
        success=True,
        latency_ms=45.2,
        prompt_tokens=100,
        completion_tokens=250,
        cost_usd=0.00012,
    )
    d_att = att.to_dict()
    assert d_att["attempt_index"] == 1
    assert d_att["model_id"] == "gemini-1.5-flash"
    assert d_att["success"] is True

    receipt = RoutingExecutionReceipt(
        receipt_id="rcpt_001",
        task_id="tsk_001",
        agent_id="agent-edge",
        priority=AgentPriority.BACKGROUND,
        primary_model="gpt-4o",
        resolved_model="gemini-1.5-flash",
        fallback_occurred=True,
        status="fallback_success",
        attempts=[att],
        total_latency_ms=45.2,
        total_tokens=350,
        total_cost_usd=0.00012,
        output_preview="Hello world",
        timestamp="2026-11-18T12:00:00Z",
    )
    d_rec = receipt.to_dict()
    assert d_rec["receipt_id"] == "rcpt_001"
    assert d_rec["priority"] == "BACKGROUND"
    assert d_rec["fallback_occurred"] is True
    assert len(d_rec["attempts"]) == 1


def test_token_governor_unregistered_model():
    gov = TokenGovernor()
    # Acquiring for an unregistered model should succeed with default policy
    ok = gov.acquire(
        model_id="unregistered-model",
        priority=AgentPriority.NORMAL,
        estimated_tokens=50,
    )
    assert ok is True
    util = gov.get_utilization("unregistered-model")
    assert util["current_rpm"] == 1
    assert util["current_tpm"] == 50


def test_token_governor_tpm_limit_exceeded():
    gov = TokenGovernor()
    policy = RateLimitPolicy(
        model_id="tiny-tpm-model",
        requests_per_minute=100,
        tokens_per_minute=200,
        max_concurrency=10,
    )
    gov.register_policy(policy)

    # First request: 150 tokens (fits under 200 * 0.85 = 170 for NORMAL priority)
    ok1 = gov.acquire("tiny-tpm-model", priority=AgentPriority.NORMAL, estimated_tokens=150)
    assert ok1 is True

    # Second request: another 50 tokens exceeds 170
    avail, wait, reason = gov.check_available("tiny-tpm-model", priority=AgentPriority.NORMAL, estimated_tokens=50)
    assert avail is False
    assert "tpm_limit_reached" in reason
    assert wait > 0.0


def test_custom_fallback_route_registration():
    router = AdaptiveModelRouter()
    custom_route = FallbackRoute(
        route_name="custom-fast",
        primary_model="gemini-1.5-flash",
        fallback_models=["gpt-4o-mini", "claude-3-5-haiku"],
        description="Ultra fast lightweight fallback cascade",
    )
    router.register_route(custom_route)
    assert "custom-fast" in router.routes
    status = router.get_fleet_status()
    assert "custom-fast" in status["routes"]


def test_telemetry_integration_with_router():
    telem = TelemetryCollector()
    telem.record_routing_event(
        primary_model="claude-3-5-sonnet",
        resolved_model="deepseek-coder-v2",
        fallback_occurred=True,
        latency_ms=125.0,
        tokens=450,
        priority="HIGH",
        status="fallback_success",
    )
    assert telem.metrics.router_routed_calls == 1
    assert telem.metrics.router_fallback_calls == 1

    md = telem.to_markdown()
    assert "Model Router & Fallbacks" in md
    assert "1 routed" in md
