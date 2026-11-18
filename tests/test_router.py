"""
Unit and integration tests for agent_lab.router:
- TokenGovernor (sliding-window TPM/RPM, priority allowances, concurrency limits)
- ModelCircuitBreaker (CLOSED, OPEN, HALF_OPEN states, canary recovery, SLA tracking)
- AdaptiveModelRouter (cascade fallback failover, quota management, receipts)
"""

from __future__ import annotations

import time
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
    TokenGovernor,
    CascadeExhaustedError,
)


def test_agent_priority_properties():
    assert AgentPriority.CRITICAL.capacity_allowance == 1.20
    assert AgentPriority.HIGH.capacity_allowance == 1.00
    assert AgentPriority.NORMAL.capacity_allowance == 0.85
    assert AgentPriority.BACKGROUND.capacity_allowance == 0.65
    assert AgentPriority.CRITICAL.weight > AgentPriority.NORMAL.weight


def test_token_governor_registration_and_utilization():
    gov = TokenGovernor()
    policy = RateLimitPolicy(
        model_id="test-model",
        requests_per_minute=10,
        tokens_per_minute=1000,
        max_concurrency=2,
    )
    gov.register_policy(policy)
    util = gov.get_utilization("test-model")
    assert util["current_rpm"] == 0
    assert util["current_tpm"] == 0
    assert util["max_rpm"] == 10
    assert util["max_tpm"] == 1000
    assert util["active_concurrency"] == 0


def test_token_governor_sliding_window_consumption():
    gov = TokenGovernor()
    policy = RateLimitPolicy(
        model_id="fast-model",
        requests_per_minute=5,
        tokens_per_minute=1000,
        max_concurrency=20,
    )
    gov.register_policy(policy)

    # Acquire tokens
    for i in range(4):
        ok = gov.acquire(
            model_id="fast-model",
            priority=AgentPriority.HIGH,
            estimated_tokens=50,
        )
        assert ok is True

    util = gov.get_utilization("fast-model")
    assert util["current_rpm"] == 4
    assert util["current_tpm"] == 200

    # 5th request should succeed at HIGH priority (requests_per_minute=5 * 1.0)
    ok = gov.acquire(
        model_id="fast-model",
        priority=AgentPriority.HIGH,
        estimated_tokens=50,
    )
    assert ok is True

    # 6th request should fail
    avail, wait_time, reason = gov.check_available(
        model_id="fast-model",
        priority=AgentPriority.HIGH,
        estimated_tokens=50,
    )
    assert avail is False
    assert "rpm_limit_reached" in reason
    assert wait_time > 0.0


def test_token_governor_concurrency_tracking():
    gov = TokenGovernor()
    policy = RateLimitPolicy(
        model_id="conc-model",
        requests_per_minute=100,
        tokens_per_minute=10000,
        max_concurrency=2,
    )
    gov.register_policy(policy)

    # Occupy 2 concurrency slots
    ok1 = gov.acquire("conc-model", priority=AgentPriority.NORMAL, estimated_tokens=10)
    ok2 = gov.acquire("conc-model", priority=AgentPriority.NORMAL, estimated_tokens=10)
    assert ok1 and ok2
    assert gov.get_utilization("conc-model")["active_concurrency"] == 2

    # 3rd request should fail concurrency check
    avail, wait, reason = gov.check_available("conc-model", priority=AgentPriority.NORMAL, estimated_tokens=10)
    assert avail is False
    assert "max_concurrency_reached" in reason

    # Release one slot
    gov.release_concurrency("conc-model")
    assert gov.get_utilization("conc-model")["active_concurrency"] == 1

    # Now 4th request can acquire
    ok4 = gov.acquire("conc-model", priority=AgentPriority.NORMAL, estimated_tokens=10)
    assert ok4 is True


def test_model_circuit_breaker_transitions():
    policy = CircuitBreakerPolicy(
        failure_threshold=3,
        recovery_time_seconds=0.1,
        success_threshold=2,
        latency_sla_ms=500.0,
        jitter_ratio=0.0,
    )
    cb = ModelCircuitBreaker(model_id="flakey-model", policy=policy)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # Record 2 failures (threshold is 3)
    cb.record_failure("HTTP 500: Server Error", is_rate_limit=False)
    cb.record_failure("HTTP 502: Bad Gateway", is_rate_limit=False)
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 2

    # 3rd failure trips the breaker to OPEN
    cb.record_failure("HTTP 503: Service Unavailable", is_rate_limit=False)
    assert cb.state == CircuitState.OPEN
    assert cb.total_trips == 1
    assert cb.can_execute() is False

    # During cooldown, can_execute remains False
    time.sleep(0.02)
    assert cb.can_execute() is False

    # Wait for recovery_time_seconds to expire
    time.sleep(0.12)
    # can_execute transitions state to HALF_OPEN
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # 1st success in HALF_OPEN
    cb.record_success(latency_ms=100.0)
    assert cb.state == CircuitState.HALF_OPEN

    # 2nd success in HALF_OPEN recovers breaker back to CLOSED
    cb.record_success(latency_ms=120.0)
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0


def test_model_circuit_breaker_half_open_failure_re_trips():
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_time_seconds=0.05,
        success_threshold=2,
        jitter_ratio=0.0,
    )
    cb = ModelCircuitBreaker(model_id="unstable-model", policy=policy)
    cb.record_failure("Crash", is_rate_limit=False)
    assert cb.state == CircuitState.OPEN

    time.sleep(0.06)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # A failure in HALF_OPEN immediately re-trips to OPEN
    cb.record_failure("Canary request failed", is_rate_limit=False)
    assert cb.state == CircuitState.OPEN
    assert cb.total_trips == 2


def test_model_circuit_breaker_latency_sla_trip():
    policy = CircuitBreakerPolicy(
        failure_threshold=5,
        latency_sla_ms=100.0,
    )
    cb = ModelCircuitBreaker(model_id="slow-model", policy=policy)
    # 5 calls to establish average latency
    for _ in range(4):
        cb.record_success(latency_ms=50.0)
    assert cb.state == CircuitState.CLOSED

    # 5th call pushes average latency over 100ms
    cb.record_success(latency_ms=500.0)
    assert cb.state == CircuitState.OPEN
    assert "latency_sla_violated" in cb.trip_reason


def test_adaptive_router_direct_primary_hit():
    router = AdaptiveModelRouter()
    receipt = router.route_and_execute(
        prompt="Explain quantum superposition in 2 sentences.",
        route_name="code-generation",
        priority=AgentPriority.NORMAL,
        agent_id="test-agent",
        estimated_output_tokens=150,
    )
    assert receipt.status == "success"
    assert receipt.primary_model == "claude-3-5-sonnet"
    assert receipt.resolved_model == "claude-3-5-sonnet"
    assert receipt.fallback_occurred is False
    assert receipt.total_tokens > 0
    assert receipt.total_cost_usd > 0.0
    assert len(receipt.attempts) == 1
    assert receipt.attempts[0].success is True


def test_adaptive_router_cascade_failover():
    router = AdaptiveModelRouter()

    # Define invoker where primary model (claude-3-5-sonnet) fails with 429
    def mock_invoker(model_id: str, prompt: str):
        if model_id == "claude-3-5-sonnet":
            return False, "HTTP 429: Rate limit exceeded", 15, 0
        return True, f"Generated from fallback {model_id}", 20, 200

    receipt = router.route_and_execute(
        prompt="Write a quicksort implementation.",
        route_name="code-generation",
        priority=AgentPriority.HIGH,
        agent_id="agent-high",
        estimated_output_tokens=200,
        provider_invoker=mock_invoker,
    )

    assert receipt.status == "fallback_success"
    assert receipt.primary_model == "claude-3-5-sonnet"
    assert receipt.resolved_model == "deepseek-coder-v2"
    assert receipt.fallback_occurred is True
    assert len(receipt.attempts) == 2
    assert receipt.attempts[0].success is False
    assert receipt.attempts[1].success is True


def test_adaptive_router_cascade_exhaustion():
    router = AdaptiveModelRouter()

    # Every model fails
    def fail_all(model_id: str, prompt: str):
        return False, f"Provider outage on {model_id}", 10, 0

    with pytest.raises(CascadeExhaustedError) as exc_info:
        router.route_and_execute(
            prompt="Compute prime factors",
            route_name="code-generation",
            provider_invoker=fail_all,
        )
    assert "Cascade exhausted" in str(exc_info.value)


def test_adaptive_router_agent_quota_enforcement():
    router = AdaptiveModelRouter()
    # Set quota of 100 tokens
    router.governor.set_agent_quota("broke-agent", 100)

    # Calling with 500 estimated tokens exceeds quota
    with pytest.raises(QuotaExceededError):
        router.route_and_execute(
            prompt="Simple task",
            route_name="code-generation",
            agent_id="broke-agent",
            estimated_output_tokens=500,
        )


def test_adaptive_router_reset_circuits():
    router = AdaptiveModelRouter()
    cb = router.get_circuit_breaker("claude-3-5-sonnet")
    cb.trip("Manual test trip")
    assert cb.state == CircuitState.OPEN

    # Reset single
    ok = router.reset_circuit("claude-3-5-sonnet")
    assert ok is True
    assert cb.state == CircuitState.CLOSED

    # Trip multiple and reset all
    cb.trip("Trip 1")
    cb2 = router.get_circuit_breaker("gpt-4o")
    cb2.trip("Trip 2")
    assert cb.state == CircuitState.OPEN and cb2.state == CircuitState.OPEN

    router.reset_all_circuits()
    assert cb.state == CircuitState.CLOSED
    assert cb2.state == CircuitState.CLOSED
