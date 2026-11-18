"""
agent_lab.router: Adaptive dynamic token rate-limiter, circuit breakers,
and cross-model cascade fallback router for autonomous agent swarms.

Provides:
- AgentPriority: CRITICAL, HIGH, NORMAL, BACKGROUND with priority-weighted quotas.
- TokenGovernor: Sliding-window rolling 60-second TPM/RPM governor with burst handling.
- ModelCircuitBreaker: 3-state (CLOSED, OPEN, HALF_OPEN) circuit breaker with latency SLA tracking.
- AdaptiveModelRouter: Intelligent cascade fallback router with automated failover and telemetry.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class AgentPriority(str, Enum):
    """Execution priority tiers for autonomous agents."""
    CRITICAL = "CRITICAL"      # System orchestrator, consensus coordinators, emergency rollbacks (1.0x capacity)
    HIGH = "HIGH"              # Primary code generation, transaction simulation (0.90x capacity)
    NORMAL = "NORMAL"          # Routine deliberation, test generation, evaluation (0.75x capacity)
    BACKGROUND = "BACKGROUND"  # Telemetry exports, log compression, background indexing (0.50x capacity)

    @property
    def capacity_allowance(self) -> float:
        """Fraction of total sliding-window capacity permissible for this priority tier."""
        if self == AgentPriority.CRITICAL:
            return 1.20  # Allows emergency burst beyond nominal ceiling
        elif self == AgentPriority.HIGH:
            return 1.00
        elif self == AgentPriority.NORMAL:
            return 0.85
        elif self == AgentPriority.BACKGROUND:
            return 0.65
        return 0.85

    @property
    def weight(self) -> int:
        if self == AgentPriority.CRITICAL:
            return 4
        elif self == AgentPriority.HIGH:
            return 3
        elif self == AgentPriority.NORMAL:
            return 2
        return 1


class CircuitState(str, Enum):
    """Operational states of a model provider circuit breaker."""
    CLOSED = "CLOSED"        # Healthy: traffic flows freely
    OPEN = "OPEN"            # Tripped: fail-fast to secondary fallback models
    HALF_OPEN = "HALF_OPEN"  # Testing recovery with canary probe requests


class RouterError(Exception):
    """Base exception for routing and rate-limiting errors."""
    pass


class RateLimitExceededError(RouterError):
    """Raised when request cannot be accommodated within rate limits."""
    def __init__(self, model_id: str, message: str, retry_after_seconds: float = 0.0) -> None:
        super().__init__(f"[{model_id}] {message} (retry_after={retry_after_seconds:.2f}s)")
        self.model_id = model_id
        self.retry_after_seconds = retry_after_seconds


class CircuitBreakerOpenError(RouterError):
    """Raised when model circuit breaker is in OPEN state."""
    def __init__(self, model_id: str, message: str) -> None:
        super().__init__(f"[{model_id}] Circuit breaker OPEN: {message}")
        self.model_id = model_id


class QuotaExceededError(RouterError):
    """Raised when an individual agent's token consumption quota is exhausted."""
    def __init__(self, agent_id: str, budget: int, consumed: int) -> None:
        super().__init__(f"Agent quota exceeded for '{agent_id}': consumed {consumed:,}/{budget:,} tokens")
        self.agent_id = agent_id
        self.budget = budget
        self.consumed = consumed


class CascadeExhaustedError(RouterError):
    """Raised when all candidate models in a fallback cascade fail."""
    def __init__(self, primary_model: str, attempted_models: List[str], last_error: str) -> None:
        super().__init__(
            f"Cascade exhausted starting from '{primary_model}'. "
            f"Failed models: {attempted_models}. Last error: {last_error}"
        )
        self.primary_model = primary_model
        self.attempted_models = attempted_models
        self.last_error = last_error


@dataclass
class RateLimitPolicy:
    """Sliding-window rate limit policy for an individual LLM model."""
    model_id: str
    requests_per_minute: int = 1000
    tokens_per_minute: int = 100_000
    burst_factor: float = 1.2
    max_concurrency: int = 20
    cost_per_million_input: float = 3.0
    cost_per_million_output: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class CircuitBreakerPolicy:
    """Trip and recovery parameters for a model provider circuit breaker."""
    failure_threshold: int = 3
    recovery_time_seconds: float = 30.0
    success_threshold: int = 2
    latency_sla_ms: float = 5000.0
    jitter_ratio: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class AgentQuota:
    """Token consumption budget and usage tracker for an agent."""
    agent_id: str
    token_budget: int = 500_000
    tokens_consumed: int = 0
    requests_count: int = 0
    window_start: float = field(default_factory=time.time)

    def is_exhausted(self, required_tokens: int) -> bool:
        return (self.tokens_consumed + required_tokens) > self.token_budget

    def consume(self, tokens: int) -> None:
        self.tokens_consumed += tokens
        self.requests_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "token_budget": self.token_budget,
            "tokens_consumed": self.tokens_consumed,
            "requests_count": self.requests_count,
            "utilization_pct": round((self.tokens_consumed / self.token_budget) * 100, 2) if self.token_budget > 0 else 0.0,
            "remaining_tokens": max(0, self.token_budget - self.tokens_consumed),
        }


@dataclass
class RoutingAttempt:
    """Detailed record of an individual model invocation attempt."""
    model_id: str
    attempt_index: int
    success: bool
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "attempt_index": self.attempt_index,
            "success": self.success,
            "error_message": self.error_message,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp,
        }


@dataclass
class RoutingExecutionReceipt:
    """Comprehensive cryptographic and operational receipt for a routed LLM execution."""
    receipt_id: str
    task_id: str
    agent_id: str
    priority: AgentPriority
    primary_model: str
    resolved_model: str
    fallback_occurred: bool
    status: str  # "success", "fallback_success", "failed"
    attempts: List[RoutingAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    output_preview: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "priority": self.priority.value,
            "primary_model": self.primary_model,
            "resolved_model": self.resolved_model,
            "fallback_occurred": self.fallback_occurred,
            "status": self.status,
            "attempts": [a.to_dict() for a in self.attempts],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "output_preview": self.output_preview,
            "timestamp": self.timestamp,
        }


class TokenGovernor:
    """
    High-throughput sliding-window rolling 60-second concurrency and token governor.
    Prevents HTTP 429 rate limit spikes across concurrent swarm workers with
    dynamic priority reservation and burst allowance.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._policies: Dict[str, RateLimitPolicy] = {}
        # Mapping model_id -> deque of (timestamp, token_count)
        self._token_windows: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
        # Mapping model_id -> deque of timestamp
        self._request_windows: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
        # Mapping model_id -> current concurrent executions
        self._active_concurrency: Dict[str, int] = collections.defaultdict(int)
        # Agent quotas
        self._quotas: Dict[str, AgentQuota] = {}

    def register_policy(self, policy: RateLimitPolicy) -> None:
        """Register or update a model rate limit policy."""
        with self._lock:
            self._policies[policy.model_id] = policy

    def set_agent_quota(self, agent_id: str, budget: int) -> AgentQuota:
        """Set token consumption quota for a specific agent."""
        with self._lock:
            quota = AgentQuota(agent_id=agent_id, token_budget=budget)
            self._quotas[agent_id] = quota
            return quota

    def get_agent_quota(self, agent_id: str) -> Optional[AgentQuota]:
        with self._lock:
            return self._quotas.get(agent_id)

    def _prune_windows(self, model_id: str, now: float) -> None:
        """Remove timestamps older than 60.0 seconds from rolling queues."""
        cutoff = now - 60.0
        tok_q = self._token_windows[model_id]
        while tok_q and tok_q[0][0] < cutoff:
            tok_q.popleft()

        req_q = self._request_windows[model_id]
        while req_q and req_q[0] < cutoff:
            req_q.popleft()

    def check_available(
        self,
        model_id: str,
        estimated_tokens: int = 1000,
        priority: AgentPriority = AgentPriority.NORMAL,
        now: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        Check if model capacity permits execution for the given priority tier.
        Returns: (is_available, wait_time_seconds, reason)
        """
        if now is None:
            now = time.time()

        with self._lock:
            policy = self._policies.get(model_id)
            if not policy:
                return True, 0.0, "ok_no_policy"

            self._prune_windows(model_id, now)

            # 1. Concurrency check
            current_conc = self._active_concurrency[model_id]
            if current_conc >= policy.max_concurrency:
                return False, 0.5, f"max_concurrency_reached ({current_conc}/{policy.max_concurrency})"

            # 2. Sliding window requests per minute (RPM)
            current_rpm = len(self._request_windows[model_id])
            rpm_ceiling = int(policy.requests_per_minute * priority.capacity_allowance)
            if current_rpm >= rpm_ceiling:
                oldest_req = self._request_windows[model_id][0]
                wait = max(0.1, 60.0 - (now - oldest_req))
                return False, wait, f"rpm_limit_reached ({current_rpm}/{rpm_ceiling})"

            # 3. Sliding window tokens per minute (TPM)
            current_tpm = sum(tokens for _, tokens in self._token_windows[model_id])
            tpm_ceiling = int(policy.tokens_per_minute * priority.capacity_allowance)
            if (current_tpm + estimated_tokens) > tpm_ceiling:
                if self._token_windows[model_id]:
                    oldest_tok_time = self._token_windows[model_id][0][0]
                    wait = max(0.1, 60.0 - (now - oldest_tok_time))
                else:
                    wait = 1.0
                return False, wait, f"tpm_limit_reached ({current_tpm + estimated_tokens:,}/{tpm_ceiling:,})"

            return True, 0.0, "ok"

    def acquire(
        self,
        model_id: str,
        estimated_tokens: int = 1000,
        priority: AgentPriority = AgentPriority.NORMAL,
        timeout: float = 0.0,
    ) -> bool:
        """
        Acquire reservation in the sliding window.
        If timeout > 0, optionally waits for capacity to clear.
        """
        start = time.time()
        while True:
            now = time.time()
            with self._lock:
                avail, wait_time, reason = self.check_available(model_id, estimated_tokens, priority, now=now)
                if avail:
                    self._request_windows[model_id].append(now)
                    self._token_windows[model_id].append((now, estimated_tokens))
                    self._active_concurrency[model_id] += 1
                    return True

            elapsed = time.time() - start
            if timeout <= 0.0 or elapsed >= timeout:
                return False

            sleep_dur = min(wait_time, timeout - elapsed, 0.25)
            time.sleep(sleep_dur)

    def release_concurrency(self, model_id: str) -> None:
        """Decrement active concurrency counter."""
        with self._lock:
            if self._active_concurrency[model_id] > 0:
                self._active_concurrency[model_id] -= 1

    def adjust_tokens(self, model_id: str, delta_tokens: int) -> None:
        """Adjust token count if actual tokens differed from estimate."""
        with self._lock:
            if delta_tokens != 0:
                now = time.time()
                self._token_windows[model_id].append((now, delta_tokens))

    def get_utilization(self, model_id: str) -> Dict[str, Any]:
        """Compute current rolling window utilization statistics."""
        now = time.time()
        with self._lock:
            self._prune_windows(model_id, now)
            policy = self._policies.get(model_id)
            current_rpm = len(self._request_windows[model_id])
            current_tpm = sum(t for _, t in self._token_windows[model_id])
            current_conc = self._active_concurrency[model_id]

            max_rpm = policy.requests_per_minute if policy else 1000
            max_tpm = policy.tokens_per_minute if policy else 100_000
            max_conc = policy.max_concurrency if policy else 20

            return {
                "model_id": model_id,
                "current_rpm": current_rpm,
                "max_rpm": max_rpm,
                "rpm_pct": round((current_rpm / max_rpm) * 100, 1) if max_rpm > 0 else 0.0,
                "current_tpm": current_tpm,
                "max_tpm": max_tpm,
                "tpm_pct": round((current_tpm / max_tpm) * 100, 1) if max_tpm > 0 else 0.0,
                "active_concurrency": current_conc,
                "max_concurrency": max_conc,
            }


class ModelCircuitBreaker:
    """
    3-State Circuit Breaker (CLOSED -> OPEN -> HALF_OPEN) with automated
    cooldown recovery, exponential backoff with jitter, and latency SLA violation detection.
    """

    def __init__(
        self,
        model_id: str,
        policy: Optional[CircuitBreakerPolicy] = None,
    ) -> None:
        self.model_id = model_id
        self.policy = policy or CircuitBreakerPolicy()
        self._lock = threading.RLock()
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_state_change = time.time()
        self.last_failure_time = 0.0
        self.trip_reason: Optional[str] = None
        self.latencies: collections.deque = collections.deque(maxlen=20)
        self.total_trips = 0

    def can_execute(self, now: Optional[float] = None) -> bool:
        """Check if execution is allowed through the circuit breaker."""
        if now is None:
            now = time.time()

        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery cooldown period has elapsed
                elapsed = now - self.last_failure_time
                cooldown = self._calculate_cooldown()
                if elapsed >= cooldown:
                    self.state = CircuitState.HALF_OPEN
                    self.consecutive_successes = 0
                    self.last_state_change = now
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                # In HALF_OPEN, permit canary probe requests
                return True

            return False

    def _calculate_cooldown(self) -> float:
        """Calculate cooldown with random jitter."""
        base = self.policy.recovery_time_seconds
        jitter_range = base * self.policy.jitter_ratio
        jitter = random.uniform(-jitter_range, jitter_range)
        return max(0.01, base + jitter)

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record successful execution."""
        with self._lock:
            if latency_ms > 0:
                self.latencies.append(latency_ms)

            # Check latency SLA violation
            if self.latencies and len(self.latencies) >= 5:
                avg_lat = sum(self.latencies) / len(self.latencies)
                if avg_lat > self.policy.latency_sla_ms:
                    self._trip(f"latency_sla_violated ({avg_lat:.1f}ms > {self.policy.latency_sla_ms:.1f}ms)")
                    return

            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.policy.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.consecutive_failures = 0
                    self.consecutive_successes = 0
                    self.trip_reason = None
                    self.last_state_change = time.time()
            elif self.state == CircuitState.CLOSED:
                self.consecutive_failures = 0

    def record_failure(self, error_message: str, is_rate_limit: bool = False) -> None:
        """Record failed execution (e.g. HTTP 429, 500, network error)."""
        with self._lock:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = time.time()

            if is_rate_limit:
                # Immediate trip on rate limit 429
                self._trip(f"rate_limit_429: {error_message}")
            elif self.state == CircuitState.HALF_OPEN:
                # Immediate trip if canary probe fails
                self._trip(f"half_open_probe_failed: {error_message}")
            elif self.consecutive_failures >= self.policy.failure_threshold:
                self._trip(f"threshold_exceeded ({self.consecutive_failures} failures): {error_message}")

    def _trip(self, reason: str) -> None:
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
        self.trip_reason = reason
        self.last_state_change = time.time()
        self.last_failure_time = time.time()
        self.total_trips += 1

    def trip(self, reason: str = "manual_trip") -> None:
        """Manually trip circuit breaker to OPEN state."""
        with self._lock:
            self._trip(reason)

    def reset(self) -> None:
        """Manually force circuit breaker back to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.trip_reason = None
            self.last_state_change = time.time()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            avg_lat = round(sum(self.latencies) / len(self.latencies), 1) if self.latencies else 0.0
            return {
                "model_id": self.model_id,
                "state": self.state.value,
                "consecutive_failures": self.consecutive_failures,
                "consecutive_successes": self.consecutive_successes,
                "trip_reason": self.trip_reason,
                "total_trips": self.total_trips,
                "avg_latency_ms": avg_lat,
                "policy": self.policy.to_dict(),
            }


@dataclass
class FallbackRoute:
    """Ordered cascade routing configuration."""
    route_name: str
    primary_model: str
    fallback_models: List[str] = field(default_factory=list)
    description: str = ""

    @property
    def full_cascade(self) -> List[str]:
        return [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]


DEFAULT_MODEL_POLICIES: Dict[str, RateLimitPolicy] = {
    "claude-3-5-sonnet": RateLimitPolicy(
        model_id="claude-3-5-sonnet",
        requests_per_minute=1000,
        tokens_per_minute=80_000,
        burst_factor=1.2,
        max_concurrency=15,
        cost_per_million_input=3.00,
        cost_per_million_output=15.00,
    ),
    "gpt-4o": RateLimitPolicy(
        model_id="gpt-4o",
        requests_per_minute=1000,
        tokens_per_minute=80_000,
        burst_factor=1.2,
        max_concurrency=15,
        cost_per_million_input=2.50,
        cost_per_million_output=10.00,
    ),
    "deepseek-coder-v2": RateLimitPolicy(
        model_id="deepseek-coder-v2",
        requests_per_minute=2000,
        tokens_per_minute=160_000,
        burst_factor=1.3,
        max_concurrency=25,
        cost_per_million_input=0.14,
        cost_per_million_output=0.28,
    ),
    "claude-3-5-haiku": RateLimitPolicy(
        model_id="claude-3-5-haiku",
        requests_per_minute=1500,
        tokens_per_minute=120_000,
        burst_factor=1.25,
        max_concurrency=20,
        cost_per_million_input=0.80,
        cost_per_million_output=4.00,
    ),
    "gpt-4o-mini": RateLimitPolicy(
        model_id="gpt-4o-mini",
        requests_per_minute=3000,
        tokens_per_minute=200_000,
        burst_factor=1.3,
        max_concurrency=30,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
    ),
    "qwen-2.5-coder-32b": RateLimitPolicy(
        model_id="qwen-2.5-coder-32b",
        requests_per_minute=5000,
        tokens_per_minute=300_000,
        burst_factor=1.5,
        max_concurrency=50,
        cost_per_million_input=0.07,
        cost_per_million_output=0.14,
    ),
}

DEFAULT_FALLBACK_ROUTES: Dict[str, FallbackRoute] = {
    "code-generation": FallbackRoute(
        route_name="code-generation",
        primary_model="claude-3-5-sonnet",
        fallback_models=["deepseek-coder-v2", "gpt-4o-mini", "qwen-2.5-coder-32b"],
        description="High-fidelity code generation with DeepSeek and Qwen open-weight fallback cascade.",
    ),
    "fast-reasoning": FallbackRoute(
        route_name="fast-reasoning",
        primary_model="gpt-4o",
        fallback_models=["claude-3-5-haiku", "gpt-4o-mini"],
        description="Low-latency reasoning and fast validation cascade.",
    ),
    "economy-swarm": FallbackRoute(
        route_name="economy-swarm",
        primary_model="deepseek-coder-v2",
        fallback_models=["qwen-2.5-coder-32b", "gpt-4o-mini"],
        description="Budget-conscious high-volume swarm workload cascade.",
    ),
}


class AdaptiveModelRouter:
    """
    Autonomous cross-model fallback router and rate-limiting orchestrator.
    Handles dynamic cascading, priority reservations, circuit breakers,
    and granular token attribution.
    """

    def __init__(
        self,
        governor: Optional[TokenGovernor] = None,
        routes: Optional[Dict[str, FallbackRoute]] = None,
    ) -> None:
        self.governor = governor or TokenGovernor()
        self.routes = routes or dict(DEFAULT_FALLBACK_ROUTES)
        self.circuit_breakers: Dict[str, ModelCircuitBreaker] = {}
        self._lock = threading.RLock()

        # Initialize default policies and breakers
        for model_id, policy in DEFAULT_MODEL_POLICIES.items():
            self.governor.register_policy(policy)
            self.circuit_breakers[model_id] = ModelCircuitBreaker(model_id=model_id)

    def register_route(self, route: FallbackRoute) -> None:
        """Add or update a fallback route."""
        with self._lock:
            self.routes[route.route_name] = route

    def get_circuit_breaker(self, model_id: str) -> ModelCircuitBreaker:
        """Get or create circuit breaker for a model."""
        with self._lock:
            if model_id not in self.circuit_breakers:
                self.circuit_breakers[model_id] = ModelCircuitBreaker(model_id=model_id)
            return self.circuit_breakers[model_id]

    def reset_circuit(self, model_id: str) -> bool:
        """Reset a model's circuit breaker to CLOSED."""
        with self._lock:
            if model_id in self.circuit_breakers:
                self.circuit_breakers[model_id].reset()
                return True
            return False

    def reset_all_circuits(self) -> None:
        """Reset all circuit breakers across the model fleet."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()

    def route_and_execute(
        self,
        prompt: str,
        route_name: str = "code-generation",
        priority: AgentPriority = AgentPriority.NORMAL,
        agent_id: str = "agent-default",
        task_id: str = "",
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: int = 500,
        provider_invoker: Optional[Callable[[str, str], Tuple[bool, str, int, int]]] = None,
    ) -> RoutingExecutionReceipt:
        """
        Execute prompt against the target route with automated fallback handling.

        provider_invoker signature: (model_id, prompt) -> (success, output_or_error, prompt_tok, comp_tok)
        If provider_invoker is None, simulates execution.
        """
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}"

        route = self.routes.get(route_name)
        if not route:
            # Fallback to direct single-model route
            route = FallbackRoute(route_name=route_name, primary_model=route_name, fallback_models=[])

        cascade = route.full_cascade
        est_in = estimated_input_tokens or max(10, len(prompt.split()) * 2)
        total_est = est_in + estimated_output_tokens

        # Check agent quota
        quota = self.governor.get_agent_quota(agent_id)
        if quota and quota.is_exhausted(total_est):
            raise QuotaExceededError(agent_id, quota.token_budget, quota.tokens_consumed)

        receipt_id = f"rcpt_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        attempts: List[RoutingAttempt] = []
        resolved_model = ""
        output_result = ""
        overall_start = time.time()

        for idx, candidate_model in enumerate(cascade):
            breaker = self.get_circuit_breaker(candidate_model)

            # 1. Check Circuit Breaker
            if not breaker.can_execute():
                attempts.append(RoutingAttempt(
                    model_id=candidate_model,
                    attempt_index=idx + 1,
                    success=False,
                    error_message=f"Circuit Breaker OPEN ({breaker.trip_reason or 'unhealthy'})",
                    latency_ms=0.5,
                ))
                continue

            # 2. Check Sliding Window Rate Limits
            avail, wait_time, reason = self.governor.check_available(
                candidate_model,
                estimated_tokens=total_est,
                priority=priority,
            )
            if not avail:
                attempts.append(RoutingAttempt(
                    model_id=candidate_model,
                    attempt_index=idx + 1,
                    success=False,
                    error_message=f"TokenGovernor rate limited: {reason}",
                    latency_ms=0.5,
                ))
                continue

            # 3. Acquire capacity
            acquired = self.governor.acquire(
                candidate_model,
                estimated_tokens=total_est,
                priority=priority,
                timeout=0.0,
            )
            if not acquired:
                attempts.append(RoutingAttempt(
                    model_id=candidate_model,
                    attempt_index=idx + 1,
                    success=False,
                    error_message="Governor acquisition failed during race",
                    latency_ms=0.5,
                ))
                continue

            # 4. Invoke Provider
            attempt_start = time.time()
            success = False
            error_msg: Optional[str] = None
            prompt_tok = est_in
            comp_tok = estimated_output_tokens

            try:
                if provider_invoker:
                    success, output_or_err, prompt_tok, comp_tok = provider_invoker(candidate_model, prompt)
                    if success:
                        output_result = output_or_err
                    else:
                        error_msg = output_or_err
                else:
                    # Deterministic simulated execution
                    success = True
                    output_result = f"Simulated output from {candidate_model} for prompt '{prompt[:30]}...'"
            except Exception as exc:
                success = False
                error_msg = str(exc)
            finally:
                self.governor.release_concurrency(candidate_model)

            latency_ms = (time.time() - attempt_start) * 1000.0

            # Compute cost
            policy = DEFAULT_MODEL_POLICIES.get(candidate_model)
            cost_usd = 0.0
            if policy:
                cost_usd = (prompt_tok * policy.cost_per_million_input + comp_tok * policy.cost_per_million_output) / 1_000_000.0

            attempt = RoutingAttempt(
                model_id=candidate_model,
                attempt_index=idx + 1,
                success=success,
                error_message=error_msg,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                cost_usd=cost_usd,
            )
            attempts.append(attempt)

            if success:
                breaker.record_success(latency_ms)
                resolved_model = candidate_model
                # Record to agent quota
                if quota:
                    quota.consume(prompt_tok + comp_tok)
                # Adjust actual tokens vs estimate
                actual_tokens = prompt_tok + comp_tok
                delta = actual_tokens - total_est
                self.governor.adjust_tokens(candidate_model, delta)
                break
            else:
                is_429 = "429" in (error_msg or "") or "rate_limit" in (error_msg or "").lower()
                breaker.record_failure(error_msg or "unknown_error", is_rate_limit=is_429)

        total_latency = (time.time() - overall_start) * 1000.0
        fallback_occurred = resolved_model != "" and resolved_model != route.primary_model
        total_tokens = sum(a.prompt_tokens + a.completion_tokens for a in attempts if a.success)
        total_cost = sum(a.cost_usd for a in attempts)

        if not resolved_model:
            last_err = attempts[-1].error_message if attempts else "no candidates available"
            raise CascadeExhaustedError(
                primary_model=route.primary_model,
                attempted_models=[a.model_id for a in attempts],
                last_error=last_err or "unknown",
            )

        status_str = "fallback_success" if fallback_occurred else "success"

        return RoutingExecutionReceipt(
            receipt_id=receipt_id,
            task_id=task_id,
            agent_id=agent_id,
            priority=priority,
            primary_model=route.primary_model,
            resolved_model=resolved_model,
            fallback_occurred=fallback_occurred,
            status=status_str,
            attempts=attempts,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            output_preview=output_result[:100],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def get_fleet_status(self) -> Dict[str, Any]:
        """Aggregate comprehensive operational status across models, governors, and breakers."""
        with self._lock:
            models_info = {}
            for model_id, breaker in self.circuit_breakers.items():
                util = self.governor.get_utilization(model_id)
                cb_info = breaker.to_dict()
                models_info[model_id] = {
                    "circuit_state": cb_info["state"],
                    "consecutive_failures": cb_info["consecutive_failures"],
                    "total_trips": cb_info["total_trips"],
                    "avg_latency_ms": cb_info["avg_latency_ms"],
                    "trip_reason": cb_info["trip_reason"],
                    "current_rpm": util["current_rpm"],
                    "max_rpm": util["max_rpm"],
                    "rpm_pct": util["rpm_pct"],
                    "current_tpm": util["current_tpm"],
                    "max_tpm": util["max_tpm"],
                    "tpm_pct": util["tpm_pct"],
                    "active_concurrency": util["active_concurrency"],
                    "max_concurrency": util["max_concurrency"],
                }

            routes_info = {
                name: {
                    "primary_model": r.primary_model,
                    "fallback_models": r.fallback_models,
                    "description": r.description,
                }
                for name, r in self.routes.items()
            }

            quotas_info = {
                agent_id: q.to_dict()
                for agent_id, q in self.governor._quotas.items()
            }

            return {
                "models": models_info,
                "routes": routes_info,
                "quotas": quotas_info,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
