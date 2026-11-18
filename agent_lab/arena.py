"""
agent_lab.arena: Multi-Agent Swarm Arena, Head-to-Head Tournament Engine,
and Collaborative Multi-Agent Workflow Benchmark Harness.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agent_lab.eval import ASTVerifier, BenchmarkTask, StatisticalInterval, TrajectoryEvaluator


class MatchResult(str, enum.Enum):
    """Outcome of a head-to-head match between two agents."""
    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"


@dataclass
class AgentMatchScore:
    """Individual performance score for an agent in a tournament match."""
    agent_id: str
    tasks_passed: int
    total_tasks: int
    pass_rate: float
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    ast_score: float = 1.0
    trajectory_score: float = 1.0
    composite_score: float = 0.0

    def compute_composite(
        self,
        weight_pass: float = 0.50,
        weight_ast: float = 0.20,
        weight_trajectory: float = 0.15,
        weight_efficiency: float = 0.15,
    ) -> float:
        """
        Calculate unified composite score (0.0 to 1.0) balancing correctness,
        syntactic validity, trajectory rule compliance, and token efficiency.
        """
        # Efficiency normalized against a 10,000 token baseline
        token_penalty = min(1.0, max(0.0, self.total_tokens / 10000.0))
        efficiency_score = max(0.0, 1.0 - (0.5 * token_penalty))

        composite = (
            (weight_pass * self.pass_rate)
            + (weight_ast * self.ast_score)
            + (weight_trajectory * self.trajectory_score)
            + (weight_efficiency * efficiency_score)
        )
        self.composite_score = round(max(0.0, min(1.0, composite)), 4)
        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentMatchScore:
        return cls(
            agent_id=data["agent_id"],
            tasks_passed=data["tasks_passed"],
            total_tasks=data["total_tasks"],
            pass_rate=data["pass_rate"],
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            total_tokens=data.get("total_tokens", 0),
            ast_score=data.get("ast_score", 1.0),
            trajectory_score=data.get("trajectory_score", 1.0),
            composite_score=data.get("composite_score", 0.0),
        )


@dataclass
class MatchOutcome:
    """Recorded outcome of a pairwise evaluation matchup."""
    match_id: str
    agent_a: str
    agent_b: str
    score_a: AgentMatchScore
    score_b: AgentMatchScore
    result_a: MatchResult
    elo_change_a: float
    elo_change_b: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["result_a"] = self.result_a.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MatchOutcome:
        return cls(
            match_id=data["match_id"],
            agent_a=data["agent_a"],
            agent_b=data["agent_b"],
            score_a=AgentMatchScore.from_dict(data["score_a"]),
            score_b=AgentMatchScore.from_dict(data["score_b"]),
            result_a=MatchResult(data["result_a"]),
            elo_change_a=float(data["elo_change_a"]),
            elo_change_b=float(data["elo_change_b"]),
            timestamp=data.get("timestamp", ""),
        )


class EloRatingSystem:
    """
    Deterministic Elo rating system with dynamic K-factors based on experience.
    """

    def __init__(
        self,
        default_rating: float = 1200.0,
        k_provisional: float = 32.0,
        k_established: float = 16.0,
        provisional_threshold: int = 30,
    ) -> None:
        self.default_rating = default_rating
        self.k_provisional = k_provisional
        self.k_established = k_established
        self.provisional_threshold = provisional_threshold

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """
        Standard logistic win expectation for agent A facing agent B:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        exponent = (rating_b - rating_a) / 400.0
        # Prevent floating point overflow
        if exponent > 50.0:
            return 0.0
        elif exponent < -50.0:
            return 1.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))

    def get_k_factor(self, match_count: int) -> float:
        """Dynamic K-factor: higher for provisional agents to settle quickly."""
        if match_count < self.provisional_threshold:
            return self.k_provisional
        return self.k_established

    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        actual_score_a: float,
        matches_a: int = 0,
        matches_b: int = 0,
    ) -> Tuple[float, float, float, float]:
        """
        Compute updated ratings and delta adjustments.
        actual_score_a: 1.0 for WIN, 0.5 for DRAW, 0.0 for LOSS.
        Returns: (new_rating_a, new_rating_b, delta_a, delta_b)
        """
        exp_a = self.expected_score(rating_a, rating_b)
        exp_b = 1.0 - exp_a
        actual_score_b = 1.0 - actual_score_a

        k_a = self.get_k_factor(matches_a)
        k_b = self.get_k_factor(matches_b)

        delta_a = round(k_a * (actual_score_a - exp_a), 2)
        delta_b = round(k_b * (actual_score_b - exp_b), 2)

        new_a = round(rating_a + delta_a, 2)
        new_b = round(rating_b + delta_b, 2)

        return new_a, new_b, delta_a, delta_b


class BradleyTerryEstimator:
    """
    Bradley-Terry Luce probabilistic choice model estimator for multi-agent rankings.
    Solves for latent skill parameters pi_i such that P(i beats j) = pi_i / (pi_i + pi_j).
    """

    @staticmethod
    def estimate_skills(
        wins_matrix: Dict[str, Dict[str, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Iterative MM (Minorization-Maximization) algorithm for Bradley-Terry model.
        wins_matrix[i][j] = number of times agent i defeated agent j.
        Returns normalized skill parameters summing to 1.0.
        """
        agents = sorted(wins_matrix.keys())
        n = len(agents)
        if n == 0:
            return {}
        if n == 1:
            return {agents[0]: 1.0}

        # Initial uniform skills
        pi: Dict[str, float] = {a: 1.0 / n for a in agents}

        # Total wins per agent
        w: Dict[str, float] = {a: sum(wins_matrix[a].get(b, 0.0) for b in agents if b != a) for a in agents}

        for _ in range(max_iterations):
            new_pi: Dict[str, float] = {}
            for a in agents:
                denom = 0.0
                for b in agents:
                    if b == a:
                        continue
                    n_ab = wins_matrix[a].get(b, 0.0) + wins_matrix.get(b, {}).get(a, 0.0)
                    if n_ab > 0:
                        denom += n_ab / (pi[a] + pi[b])
                if denom > 0:
                    new_pi[a] = w[a] / denom
                else:
                    new_pi[a] = pi[a]

            # Normalize to sum to 1.0
            total_pi = sum(new_pi.values())
            if total_pi > 0:
                new_pi = {a: v / total_pi for a, v in new_pi.items()}

            # Check convergence
            delta = sum(abs(new_pi[a] - pi[a]) for a in agents)
            pi = new_pi
            if delta < tolerance:
                break

        return {a: round(v, 4) for a, v in pi.items()}


@dataclass
class ArenaLeaderboard:
    """
    Persistent state and rankings tracker for the agent arena.
    """
    elo_system: EloRatingSystem = field(default_factory=EloRatingSystem)
    ratings: Dict[str, float] = field(default_factory=dict)
    matches_count: Dict[str, int] = field(default_factory=dict)
    wins_count: Dict[str, int] = field(default_factory=dict)
    losses_count: Dict[str, int] = field(default_factory=dict)
    draws_count: Dict[str, int] = field(default_factory=dict)
    match_history: List[MatchOutcome] = field(default_factory=list)

    def register_agent(self, agent_id: str, initial_rating: Optional[float] = None) -> None:
        """Register an agent with default or specified rating."""
        if agent_id not in self.ratings:
            self.ratings[agent_id] = initial_rating or self.elo_system.default_rating
            self.matches_count[agent_id] = 0
            self.wins_count[agent_id] = 0
            self.losses_count[agent_id] = 0
            self.draws_count[agent_id] = 0

    def record_match(
        self,
        agent_a: str,
        agent_b: str,
        score_a: AgentMatchScore,
        score_b: AgentMatchScore,
        match_id: Optional[str] = None,
    ) -> MatchOutcome:
        """
        Record and score a head-to-head match, updating Elo ratings and win/loss records.
        """
        self.register_agent(agent_a)
        self.register_agent(agent_b)

        # Determine winner based on composite score
        comp_a = score_a.compute_composite()
        comp_b = score_b.compute_composite()

        epsilon = 0.005  # threshold for draw
        if abs(comp_a - comp_b) <= epsilon:
            result_a = MatchResult.DRAW
            actual_a = 0.5
            self.draws_count[agent_a] += 1
            self.draws_count[agent_b] += 1
        elif comp_a > comp_b:
            result_a = MatchResult.WIN
            actual_a = 1.0
            self.wins_count[agent_a] += 1
            self.losses_count[agent_b] += 1
        else:
            result_a = MatchResult.LOSS
            actual_a = 0.0
            self.losses_count[agent_a] += 1
            self.wins_count[agent_b] += 1

        new_a, new_b, delta_a, delta_b = self.elo_system.update_ratings(
            rating_a=self.ratings[agent_a],
            rating_b=self.ratings[agent_b],
            actual_score_a=actual_a,
            matches_a=self.matches_count[agent_a],
            matches_b=self.matches_count[agent_b],
        )

        self.ratings[agent_a] = new_a
        self.ratings[agent_b] = new_b
        self.matches_count[agent_a] += 1
        self.matches_count[agent_b] += 1

        outcome = MatchOutcome(
            match_id=match_id or f"match-{uuid.uuid4().hex[:8]}",
            agent_a=agent_a,
            agent_b=agent_b,
            score_a=score_a,
            score_b=score_b,
            result_a=result_a,
            elo_change_a=delta_a,
            elo_change_b=delta_b,
        )
        self.match_history.append(outcome)
        return outcome

    def get_rankings(self) -> List[Dict[str, Any]]:
        """
        Return sorted leaderboard table with win rates and Wilson confidence intervals.
        """
        rankings = []
        for agent_id, rating in sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True):
            matches = self.matches_count[agent_id]
            wins = self.wins_count[agent_id]
            losses = self.losses_count[agent_id]
            draws = self.draws_count[agent_id]
            win_rate = round(wins / matches, 4) if matches > 0 else 0.0
            ci_lower, ci_upper = StatisticalInterval.wilson_score_interval(wins, matches) if matches > 0 else (0.0, 0.0)

            rankings.append({
                "agent_id": agent_id,
                "rating": rating,
                "matches": matches,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "win_rate_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            })
        return rankings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratings": self.ratings,
            "matches_count": self.matches_count,
            "wins_count": self.wins_count,
            "losses_count": self.losses_count,
            "draws_count": self.draws_count,
            "match_history": [m.to_dict() for m in self.match_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArenaLeaderboard:
        lb = cls()
        lb.ratings = {k: float(v) for k, v in data.get("ratings", {}).items()}
        lb.matches_count = {k: int(v) for k, v in data.get("matches_count", {}).items()}
        lb.wins_count = {k: int(v) for k, v in data.get("wins_count", {}).items()}
        lb.losses_count = {k: int(v) for k, v in data.get("losses_count", {}).items()}
        lb.draws_count = {k: int(v) for k, v in data.get("draws_count", {}).items()}
        lb.match_history = [MatchOutcome.from_dict(m) for m in data.get("match_history", [])]
        return lb

    def export_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def import_json(cls, path: Union[str, Path]) -> ArenaLeaderboard:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SwarmWorkflowStage:
    """Individual execution stage within a multi-agent DAG workflow."""
    stage_id: str
    role: str
    assigned_agent: str
    input_artifacts: List[str] = field(default_factory=list)
    output_artifacts: List[str] = field(default_factory=list)
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    execution_time_ms: float = 0.0
    tokens_consumed: int = 0
    error: Optional[str] = None


@dataclass
class SwarmWorkflowReport:
    """Audit report for an end-to-end multi-agent collaborative benchmark workflow."""
    workflow_id: str
    workflow_name: str
    participating_agents: List[str]
    stages: List[SwarmWorkflowStage]
    overall_success: bool
    total_execution_time_ms: float
    total_tokens_consumed: int
    consensus_rounds: int = 1
    occ_conflicts_resolved: int = 0
    coordination_efficiency: float = 1.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SwarmWorkflowHarness:
    """
    Benchmarking harness for executing and validating complex multi-agent collaborative workflows.
    Models multi-stage DAGs: Specification -> Implementation -> Verification -> Swarm Consensus.
    """

    def __init__(self) -> None:
        self.execution_history: List[SwarmWorkflowReport] = []

    @staticmethod
    def create_standard_dev_workflow(
        workflow_id: str,
        planner: str = "agent-planner",
        coder: str = "agent-coder",
        reviewer: str = "agent-reviewer",
        auditor: str = "agent-auditor",
    ) -> List[SwarmWorkflowStage]:
        """Generate a standard 4-stage software engineering swarm pipeline."""
        return [
            SwarmWorkflowStage(
                stage_id="stage-1-spec",
                role="PLANNER",
                assigned_agent=planner,
                input_artifacts=["user_requirements.md"],
                output_artifacts=["architecture_spec.json"],
            ),
            SwarmWorkflowStage(
                stage_id="stage-2-codegen",
                role="CODER",
                assigned_agent=coder,
                input_artifacts=["architecture_spec.json"],
                output_artifacts=["solution_module.py"],
            ),
            SwarmWorkflowStage(
                stage_id="stage-3-review",
                role="REVIEWER",
                assigned_agent=reviewer,
                input_artifacts=["solution_module.py"],
                output_artifacts=["ast_review_report.json"],
            ),
            SwarmWorkflowStage(
                stage_id="stage-4-consensus",
                role="AUDITOR",
                assigned_agent=auditor,
                input_artifacts=["solution_module.py", "ast_review_report.json"],
                output_artifacts=["consensus_commit_receipt.json"],
            ),
        ]

    def execute_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        stages: List[SwarmWorkflowStage],
        stage_executor: Optional[Callable[[SwarmWorkflowStage], bool]] = None,
    ) -> SwarmWorkflowReport:
        """
        Execute the stages sequentially, measuring latency, token usage, and coordination.
        """
        start_time = time.time()
        participating = list(dict.fromkeys(s.assigned_agent for s in stages))
        overall_success = True
        total_tokens = 0

        for stage in stages:
            stage_start = time.time()
            stage.status = "IN_PROGRESS"

            # Execute stage
            success = True
            if stage_executor:
                try:
                    success = stage_executor(stage)
                except Exception as e:
                    success = False
                    stage.error = str(e)
            else:
                # Default mock execution
                stage.tokens_consumed = 450
                stage.execution_time_ms = 120.0

            stage.execution_time_ms = round((time.time() - stage_start) * 1000.0, 2)
            total_tokens += stage.tokens_consumed

            if success:
                stage.status = "COMPLETED"
            else:
                stage.status = "FAILED"
                overall_success = False
                break

        total_time_ms = round((time.time() - start_time) * 1000.0, 2)

        # Efficiency calculation: stage success ratio discounted by execution time
        stage_completion = sum(1 for s in stages if s.status == "COMPLETED") / len(stages) if stages else 0.0
        efficiency = round(max(0.1, stage_completion * (1.0 - min(0.5, total_tokens / 50000.0))), 3)

        report = SwarmWorkflowReport(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            participating_agents=participating,
            stages=stages,
            overall_success=overall_success,
            total_execution_time_ms=total_time_ms,
            total_tokens_consumed=total_tokens,
            consensus_rounds=1 if overall_success else 2,
            occ_conflicts_resolved=0,
            coordination_efficiency=efficiency,
        )
        self.execution_history.append(report)
        return report
