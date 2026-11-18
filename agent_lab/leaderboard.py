"""
agent_lab.leaderboard: Cross-Agent Unified Benchmark Dashboard,
Multi-Model Token Economics & Comparative Leaderboard Engine.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent_lab.eval import StatisticalInterval


@dataclass
class ModelPricing:
    """Pricing and hardware SLA characteristics for an LLM model."""
    model_id: str
    provider: str
    input_cost_per_million: float  # USD per 1M prompt tokens
    output_cost_per_million: float  # USD per 1M completion tokens
    context_window: int = 128000
    latency_sla_ms: float = 1500.0

    def compute_task_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute execution cost in USD for a single task."""
        input_cost = (prompt_tokens / 1_000_000.0) * self.input_cost_per_million
        output_cost = (completion_tokens / 1_000_000.0) * self.output_cost_per_million
        return round(input_cost + output_cost, 6)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default model catalog covering frontier and open-weight models
DEFAULT_MODEL_PRICING_CATALOG: Dict[str, ModelPricing] = {
    "claude-3-5-sonnet": ModelPricing(
        model_id="claude-3-5-sonnet",
        provider="anthropic",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        context_window=200000,
        latency_sla_ms=1200.0,
    ),
    "claude-3-5-haiku": ModelPricing(
        model_id="claude-3-5-haiku",
        provider="anthropic",
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
        context_window=200000,
        latency_sla_ms=550.0,
    ),
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        provider="openai",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
        context_window=128000,
        latency_sla_ms=1100.0,
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        provider="openai",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        context_window=128000,
        latency_sla_ms=480.0,
    ),
    "deepseek-coder-v2": ModelPricing(
        model_id="deepseek-coder-v2",
        provider="deepseek",
        input_cost_per_million=0.14,
        output_cost_per_million=0.28,
        context_window=128000,
        latency_sla_ms=850.0,
    ),
    "qwen-2.5-coder-32b": ModelPricing(
        model_id="qwen-2.5-coder-32b",
        provider="alibaba",
        input_cost_per_million=0.20,
        output_cost_per_million=0.20,
        context_window=128000,
        latency_sla_ms=750.0,
    ),
    "llama-3.3-70b": ModelPricing(
        model_id="llama-3.3-70b",
        provider="meta",
        input_cost_per_million=0.25,
        output_cost_per_million=0.50,
        context_window=128000,
        latency_sla_ms=800.0,
    ),
    "gemini-1.5-pro": ModelPricing(
        model_id="gemini-1.5-pro",
        provider="google",
        input_cost_per_million=1.25,
        output_cost_per_million=5.00,
        context_window=2000000,
        latency_sla_ms=1400.0,
    ),
    "gemini-1.5-flash": ModelPricing(
        model_id="gemini-1.5-flash",
        provider="google",
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        context_window=1000000,
        latency_sla_ms=420.0,
    ),
}


@dataclass
class AgentEvaluationRecord:
    """Benchmark test results for a specific agent and backing model configuration."""
    agent_id: str
    model_id: str
    benchmark_suite: str
    total_tasks: int
    passed_tasks: int
    pass_at_1: float
    pass_at_5: Optional[float] = None
    tool_f1_score: float = 0.0
    mean_latency_ms: float = 1000.0
    p95_latency_ms: float = 1800.0
    avg_prompt_tokens: int = 1500
    avg_completion_tokens: int = 500
    safety_violations: int = 0
    zk_verified_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_task_cost(self, pricing: Optional[ModelPricing] = None) -> float:
        """Compute estimated USD cost per task using pricing catalog or fallback."""
        model_pricing = pricing or DEFAULT_MODEL_PRICING_CATALOG.get(
            self.model_id,
            ModelPricing(self.model_id, "generic", 1.0, 3.0),
        )
        return model_pricing.compute_task_cost(
            self.avg_prompt_tokens,
            self.avg_completion_tokens,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentEvaluationRecord:
        return cls(
            agent_id=data["agent_id"],
            model_id=data["model_id"],
            benchmark_suite=data.get("benchmark_suite", "general-agent-eval"),
            total_tasks=int(data.get("total_tasks", 1)),
            passed_tasks=int(data.get("passed_tasks", 0)),
            pass_at_1=float(data.get("pass_at_1", 0.0)),
            pass_at_5=float(data["pass_at_5"]) if data.get("pass_at_5") is not None else None,
            tool_f1_score=float(data.get("tool_f1_score", 0.0)),
            mean_latency_ms=float(data.get("mean_latency_ms", 1000.0)),
            p95_latency_ms=float(data.get("p95_latency_ms", 1800.0)),
            avg_prompt_tokens=int(data.get("avg_prompt_tokens", 1500)),
            avg_completion_tokens=int(data.get("avg_completion_tokens", 500)),
            safety_violations=int(data.get("safety_violations", 0)),
            zk_verified_rate=float(data.get("zk_verified_rate", 1.0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeaderboardEntry:
    """Ranked leaderboard row with composite score and Pareto optimal tagging."""
    rank: int
    agent_id: str
    model_id: str
    tier: str  # S, A, B, C, D
    composite_score: float  # 0.0 - 100.0
    pass_at_1: float
    pass_at_1_ci: Tuple[float, float]
    tool_f1: float
    mean_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_tasks_usd: float
    safety_score: float
    zk_rate: float
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LeaderboardEngine:
    """Calculates weighted composite scores, rankings, and Pareto efficiency frontiers."""

    DEFAULT_WEIGHTS = {
        "accuracy": 0.35,
        "tool_f1": 0.25,
        "latency": 0.15,
        "cost": 0.15,
        "safety": 0.10,
    }

    @classmethod
    def calculate_composite_score(
        cls,
        eval_record: AgentEvaluationRecord,
        cost_per_1k: float,
        weights: Optional[Dict[str, float]] = None,
        max_latency_ref_ms: float = 3000.0,
        max_cost_ref_usd: float = 25.0,
    ) -> float:
        """
        Calculate normalized composite score (0-100) using weighted multi-attribute formula:
        - Accuracy (Pass@1): [0, 1] -> 0-100
        - Tool Trajectory F1: [0, 1] -> 0-100
        - Latency Efficiency: 1.0 - (latency / max_ref), clamped [0, 1]
        - Cost Efficiency: 1.0 - (cost / max_ref), clamped [0, 1]
        - Safety & ZK: (1.0 - min(violations * 0.1, 1.0)) * zk_verified_rate
        """
        w = weights or cls.DEFAULT_WEIGHTS
        total_w = sum(w.values()) or 1.0

        # Component 1: Accuracy
        acc_score = max(0.0, min(1.0, eval_record.pass_at_1)) * 100.0

        # Component 2: Tool selection F1
        f1_score = max(0.0, min(1.0, eval_record.tool_f1_score)) * 100.0

        # Component 3: Latency efficiency
        lat_ratio = eval_record.mean_latency_ms / max(max_latency_ref_ms, 1.0)
        lat_eff = max(0.0, min(1.0, 1.0 - lat_ratio)) * 100.0

        # Component 4: Cost efficiency
        cost_ratio = cost_per_1k / max(max_cost_ref_usd, 0.01)
        cost_eff = max(0.0, min(1.0, 1.0 - cost_ratio)) * 100.0

        # Component 5: Safety and ZK compliance
        safety_ratio = max(0.0, 1.0 - (eval_record.safety_violations * 0.15))
        safety_eff = safety_ratio * max(0.0, min(1.0, eval_record.zk_verified_rate)) * 100.0

        raw_score = (
            w.get("accuracy", 0.35) * acc_score
            + w.get("tool_f1", 0.25) * f1_score
            + w.get("latency", 0.15) * lat_eff
            + w.get("cost", 0.15) * cost_eff
            + w.get("safety", 0.10) * safety_eff
        ) / total_w

        return round(max(0.0, min(100.0, raw_score)), 2)

    @classmethod
    def assign_tier(cls, composite_score: float) -> str:
        """Assign engineering tier based on composite score."""
        if composite_score >= 90.0:
            return "S"
        elif composite_score >= 80.0:
            return "A"
        elif composite_score >= 70.0:
            return "B"
        elif composite_score >= 60.0:
            return "C"
        return "D"

    @classmethod
    def compute_pareto_frontier(
        cls,
        entries: List[LeaderboardEntry],
    ) -> List[LeaderboardEntry]:
        """
        Identify Pareto optimal entries minimizing cost while maximizing composite score.
        An entry A dominates B if A.cost <= B.cost AND A.score >= B.score,
        with at least one strict inequality.
        """
        for i, candidate in enumerate(entries):
            dominated = False
            for j, rival in enumerate(entries):
                if i == j:
                    continue
                # rival dominates candidate if rival has lower or equal cost and higher or equal score
                rival_cheaper_or_equal = rival.cost_per_1k_tasks_usd <= candidate.cost_per_1k_tasks_usd
                rival_better_or_equal = rival.composite_score >= candidate.composite_score
                rival_strictly_better = (
                    rival.cost_per_1k_tasks_usd < candidate.cost_per_1k_tasks_usd
                    or rival.composite_score > candidate.composite_score
                )
                if rival_cheaper_or_equal and rival_better_or_equal and rival_strictly_better:
                    dominated = True
                    break
            candidate.is_pareto_optimal = not dominated
        return entries

    @classmethod
    def generate_leaderboard(
        cls,
        evaluations: List[AgentEvaluationRecord],
        weights: Optional[Dict[str, float]] = None,
        pricing_catalog: Optional[Dict[str, ModelPricing]] = None,
    ) -> List[LeaderboardEntry]:
        """
        Generate ranked, tier-assigned, and Pareto-tagged leaderboard from evaluation records.
        """
        if not evaluations:
            return []

        catalog = pricing_catalog or DEFAULT_MODEL_PRICING_CATALOG

        # Determine reference normalization maximums
        costs_1k: List[float] = []
        for rec in evaluations:
            single_cost = rec.compute_task_cost(catalog.get(rec.model_id))
            costs_1k.append(single_cost * 1000.0)

        max_cost_ref = max(costs_1k) * 1.2 if costs_1k and max(costs_1k) > 0 else 25.0
        max_lat_ref = max([r.mean_latency_ms for r in evaluations]) * 1.25 if evaluations else 3000.0

        raw_entries: List[Tuple[float, AgentEvaluationRecord, float]] = []
        for rec, cost_1k in zip(evaluations, costs_1k):
            comp_score = cls.calculate_composite_score(
                rec,
                cost_1k,
                weights=weights,
                max_latency_ref_ms=max_lat_ref,
                max_cost_ref_usd=max_cost_ref,
            )
            raw_entries.append((comp_score, rec, cost_1k))

        # Sort descending by composite score (and break ties with accuracy then cost)
        raw_entries.sort(key=lambda x: (x[0], x[1].pass_at_1, -x[2]), reverse=True)

        entries: List[LeaderboardEntry] = []
        for rank, (score, rec, cost_1k) in enumerate(raw_entries, start=1):
            ci = StatisticalInterval.wilson_score_interval(
                rec.passed_tasks,
                rec.total_tasks,
                confidence=0.95,
            )
            safety_pct = max(0.0, 100.0 - (rec.safety_violations * 15.0))
            tier = cls.assign_tier(score)
            entry = LeaderboardEntry(
                rank=rank,
                agent_id=rec.agent_id,
                model_id=rec.model_id,
                tier=tier,
                composite_score=score,
                pass_at_1=round(rec.pass_at_1, 4),
                pass_at_1_ci=(round(ci[0], 4), round(ci[1], 4)),
                tool_f1=round(rec.tool_f1_score, 4),
                mean_latency_ms=round(rec.mean_latency_ms, 1),
                p95_latency_ms=round(rec.p95_latency_ms, 1),
                cost_per_1k_tasks_usd=round(cost_1k, 4),
                safety_score=round(safety_pct, 1),
                zk_rate=round(rec.zk_verified_rate, 4),
                is_pareto_optimal=False,
            )
            entries.append(entry)

        # Compute Pareto optimality flags
        cls.compute_pareto_frontier(entries)
        return entries


@dataclass
class SwarmEconomicsReport:
    """Projected multi-model token economics and cost analysis across an agent swarm."""
    total_tasks_per_day: int
    simulation_days: int
    total_prompt_tokens: int
    total_completion_tokens: int
    daily_cost_usd: float
    monthly_cost_usd: float
    model_breakdown: Dict[str, Dict[str, Any]]
    downgrade_savings_analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenEconomicsSimulator:
    """Simulates swarm-scale token economics, routing trade-offs, and cost projections."""

    @classmethod
    def simulate_workload(
        cls,
        model_allocations: Dict[str, int],  # model_id -> task count percentage (0-100)
        tasks_per_day: int = 10000,
        days: int = 30,
        avg_prompt_tokens: int = 1500,
        avg_completion_tokens: int = 500,
        pricing_catalog: Optional[Dict[str, ModelPricing]] = None,
    ) -> SwarmEconomicsReport:
        catalog = pricing_catalog or DEFAULT_MODEL_PRICING_CATALOG
        total_weight = sum(model_allocations.values()) or 1

        daily_prompt_tokens = 0
        daily_completion_tokens = 0
        daily_total_cost = 0.0
        breakdown: Dict[str, Dict[str, Any]] = {}

        for model_id, weight in model_allocations.items():
            pricing = catalog.get(
                model_id,
                ModelPricing(model_id, "generic", 1.0, 3.0),
            )
            model_task_count = int((weight / total_weight) * tasks_per_day)
            p_tokens = model_task_count * avg_prompt_tokens
            c_tokens = model_task_count * avg_completion_tokens
            task_cost = pricing.compute_task_cost(avg_prompt_tokens, avg_completion_tokens)
            model_daily_cost = model_task_count * task_cost

            daily_prompt_tokens += p_tokens
            daily_completion_tokens += c_tokens
            daily_total_cost += model_daily_cost

            breakdown[model_id] = {
                "tasks_per_day": model_task_count,
                "share_pct": round((weight / total_weight) * 100.0, 1),
                "cost_per_task_usd": task_cost,
                "daily_cost_usd": round(model_daily_cost, 2),
                "monthly_cost_usd": round(model_daily_cost * days, 2),
                "provider": pricing.provider,
                "latency_sla_ms": pricing.latency_sla_ms,
            }

        # Downgrade analysis: compare 100% frontier flagship vs current allocation
        flagship_pricing = catalog.get("claude-3-5-sonnet", catalog.get("gpt-4o"))
        if flagship_pricing:
            all_flagship_cost = tasks_per_day * flagship_pricing.compute_task_cost(
                avg_prompt_tokens,
                avg_completion_tokens,
            )
            monthly_flagship_cost = all_flagship_cost * days
            monthly_current = daily_total_cost * days
            savings_usd = round(max(0.0, monthly_flagship_cost - monthly_current), 2)
            savings_pct = round((savings_usd / max(monthly_flagship_cost, 0.01)) * 100.0, 1)
        else:
            savings_usd = 0.0
            savings_pct = 0.0

        downgrade_analysis = {
            "all_flagship_monthly_cost_usd": round(flagship_pricing.compute_task_cost(avg_prompt_tokens, avg_completion_tokens) * tasks_per_day * days, 2) if flagship_pricing else 0.0,
            "hybrid_routing_monthly_cost_usd": round(daily_total_cost * days, 2),
            "monthly_savings_usd": savings_usd,
            "savings_percentage": savings_pct,
        }

        return SwarmEconomicsReport(
            total_tasks_per_day=tasks_per_day,
            simulation_days=days,
            total_prompt_tokens=daily_prompt_tokens * days,
            total_completion_tokens=daily_completion_tokens * days,
            daily_cost_usd=round(daily_total_cost, 2),
            monthly_cost_usd=round(daily_total_cost * days, 2),
            model_breakdown=breakdown,
            downgrade_savings_analysis=downgrade_analysis,
        )


class LeaderboardDashboard:
    """Generates dark-theme responsive HTML reports with embedded SVG charts."""

    @classmethod
    def generate_svg_bar_chart(
        cls,
        entries: List[LeaderboardEntry],
        width: int = 720,
        height: int = 340,
    ) -> str:
        """Render standalone SVG horizontal bar chart for composite scores."""
        if not entries:
            return ""

        top_entries = entries[:8]
        bar_height = 24
        gap = 14
        start_y = 50
        max_bar_width = width - 260

        tier_colors = {
            "S": "#10b981",  # emerald
            "A": "#06b6d4",  # cyan
            "B": "#6366f1",  # indigo
            "C": "#f59e0b",  # amber
            "D": "#ef4444",  # red
        }

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" class="chart-svg">',
            '<defs>',
            '  <linearGradient id="gridGrad" x1="0" y1="0" x2="1" y2="0">',
            '    <stop offset="0%" stop-color="#334155" stop-opacity="0.2"/>',
            '    <stop offset="100%" stop-color="#334155" stop-opacity="0.8"/>',
            '  </linearGradient>',
            '</defs>',
            f'<rect width="{width}" height="{height}" fill="#0f172a" rx="8"/>',
            '<text x="20" y="30" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="14" font-weight="600">Model Composite Benchmark Scores (0–100)</text>',
        ]

        # Draw grid ticks
        for tick in [25, 50, 75, 100]:
            x_pos = 180 + int((tick / 100.0) * max_bar_width)
            svg_parts.append(
                f'<line x1="{x_pos}" y1="42" x2="{x_pos}" y2="{height - 20}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{x_pos}" y="{height - 8}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="10" text-anchor="middle">{tick}</text>'
            )

        for idx, item in enumerate(top_entries):
            y = start_y + idx * (bar_height + gap)
            bar_w = max(4, int((item.composite_score / 100.0) * max_bar_width))
            color = tier_colors.get(item.tier, "#06b6d4")
            label = f"{item.agent_id} ({item.model_id})"
            if len(label) > 22:
                label = label[:20] + ".."
            label_esc = html.escape(label)

            svg_parts.append(
                f'<text x="170" y="{y + 16}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="11" text-anchor="end">{label_esc}</text>'
            )
            svg_parts.append(
                f'<rect x="180" y="{y}" width="{bar_w}" height="{bar_height}" fill="{color}" rx="4" opacity="0.9"/>'
            )
            pareto_mark = " ★" if item.is_pareto_optimal else ""
            score_text = f"{item.composite_score:.1f}{pareto_mark}"
            svg_parts.append(
                f'<text x="{180 + bar_w + 8}" y="{y + 16}" fill="{color}" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">{score_text}</text>'
            )

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)

    @classmethod
    def generate_svg_scatter_plot(
        cls,
        entries: List[LeaderboardEntry],
        width: int = 720,
        height: int = 340,
    ) -> str:
        """Render standalone SVG Pareto frontier scatter plot (Cost vs Composite Score)."""
        if not entries:
            return ""

        plot_x = 70
        plot_y = 40
        plot_w = width - 110
        plot_h = height - 90

        max_cost = max([e.cost_per_1k_tasks_usd for e in entries]) * 1.15 or 10.0
        min_score = max(0.0, min([e.composite_score for e in entries]) - 10.0)
        max_score = 100.0
        score_span = max(max_score - min_score, 1.0)

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" class="chart-svg">',
            f'<rect width="{width}" height="{height}" fill="#0f172a" rx="8"/>',
            '<text x="20" y="25" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="14" font-weight="600">Pareto Frontier: Cost per 1k Tasks vs Composite Quality</text>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#475569" stroke-width="1.5"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#475569" stroke-width="1.5"/>',
            f'<text x="{plot_x + plot_w / 2}" y="{height - 10}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="11" text-anchor="middle">Execution Cost per 1,000 Tasks ($ USD)</text>',
            f'<text x="20" y="{plot_y + plot_h / 2}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="11" text-anchor="middle" transform="rotate(-90 20 {plot_y + plot_h / 2})">Score</text>',
        ]

        # Draw gridlines
        for i in range(5):
            frac = i / 4.0
            x_val = frac * max_cost
            cx = plot_x + frac * plot_w
            svg_parts.append(
                f'<line x1="{cx}" y1="{plot_y}" x2="{cx}" y2="{plot_y + plot_h}" stroke="#1e293b" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{cx}" y="{plot_y + plot_h + 16}" fill="#64748b" font-family="system-ui, sans-serif" font-size="10" text-anchor="middle">${x_val:.1f}</text>'
            )

            y_val = min_score + (1.0 - frac) * score_span
            cy = plot_y + frac * plot_h
            svg_parts.append(
                f'<line x1="{plot_x}" y1="{cy}" x2="{plot_x + plot_w}" y2="{cy}" stroke="#1e293b" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{plot_x - 8}" y="{cy + 4}" fill="#64748b" font-family="system-ui, sans-serif" font-size="10" text-anchor="end">{y_val:.0f}</text>'
            )

        # Plot Pareto frontier polyline connecting Pareto optimal points sorted by cost
        pareto_pts = [e for e in entries if e.is_pareto_optimal]
        pareto_pts.sort(key=lambda e: e.cost_per_1k_tasks_usd)
        if len(pareto_pts) > 1:
            line_pts: List[str] = []
            for p in pareto_pts:
                px = plot_x + (p.cost_per_1k_tasks_usd / max_cost) * plot_w
                py = plot_y + plot_h - ((p.composite_score - min_score) / score_span) * plot_h
                line_pts.append(f"{px:.1f},{py:.1f}")
            svg_parts.append(
                f'<polyline points="{" ".join(line_pts)}" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="4,4"/>'
            )

        # Plot data points
        for item in entries:
            px = plot_x + (item.cost_per_1k_tasks_usd / max_cost) * plot_w
            py = plot_y + plot_h - ((item.composite_score - min_score) / score_span) * plot_h
            color = "#10b981" if item.is_pareto_optimal else "#38bdf8"
            r = 6 if item.is_pareto_optimal else 4
            label = html.escape(f"{item.model_id} ({item.composite_score:.1f})")

            svg_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            svg_parts.append(
                f'<text x="{px + 8:.1f}" y="{py + 3:.1f}" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="9">{label}</text>'
            )

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)

    @classmethod
    def export_html_report(
        cls,
        entries: List[LeaderboardEntry],
        economics: Optional[SwarmEconomicsReport] = None,
        title: str = "Agent-Lab Cross-Agent Unified Benchmark Leaderboard",
    ) -> str:
        """Render complete, standalone HTML5 report with responsive dark styling and SVG graphs."""
        chart1_svg = cls.generate_svg_bar_chart(entries)
        chart2_svg = cls.generate_svg_scatter_plot(entries)

        # Build table rows
        rows: List[str] = []
        for e in entries:
            tier_class = f"badge-{e.tier.lower()}"
            pareto_badge = '<span class="badge-pareto">★ Pareto</span>' if e.is_pareto_optimal else ''
            ci_str = f"[{e.pass_at_1_ci[0]*100:.1f}%, {e.pass_at_1_ci[1]*100:.1f}%]"
            rows.append(f"""
            <tr>
                <td class="bold">#{e.rank}</td>
                <td><span class="badge {tier_class}">{e.tier}</span></td>
                <td><strong>{html.escape(e.agent_id)}</strong><br/><small class="text-muted">{html.escape(e.model_id)}</small></td>
                <td class="score-cell">{e.composite_score:.1f} {pareto_badge}</td>
                <td>{e.pass_at_1 * 100:.1f}% <small class="text-muted">{ci_str}</small></td>
                <td>{e.tool_f1:.4f}</td>
                <td>{e.mean_latency_ms:.0f} ms <small class="text-muted">(p95: {e.p95_latency_ms:.0f})</small></td>
                <td>${e.cost_per_1k_tasks_usd:.4f}</td>
                <td>{e.safety_score:.1f}%</td>
                <td>{e.zk_rate * 100:.1f}%</td>
            </tr>
            """)

        table_rows_html = "\n".join(rows)

        # Summary KPIs
        total_models = len(entries)
        top_model = entries[0].model_id if entries else "None"
        pareto_count = len([e for e in entries if e.is_pareto_optimal])
        best_cost_entry = min(entries, key=lambda x: x.cost_per_1k_tasks_usd) if entries else None
        cheapest_model = f"{best_cost_entry.model_id} (${best_cost_entry.cost_per_1k_tasks_usd:.4f})" if best_cost_entry else "None"

        # Economics breakdown HTML if provided
        econ_html = ""
        if economics:
            econ_html = f"""
            <div class="card" style="margin-top: 24px;">
                <h3>Multi-Model Token Economics & Hybrid Routing Simulation</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Simulated Daily Tasks</div>
                        <div class="metric-value">{economics.total_tasks_per_day:,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Projected Daily Cost</div>
                        <div class="metric-value">${economics.daily_cost_usd:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Projected Monthly Cost</div>
                        <div class="metric-value">${economics.monthly_cost_usd:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Hybrid Routing Monthly Savings</div>
                        <div class="metric-value text-emerald">${economics.downgrade_savings_analysis.get('monthly_savings_usd', 0):,.2f} ({economics.downgrade_savings_analysis.get('savings_percentage', 0):.1f}%)</div>
                    </div>
                </div>
            </div>
            """

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #0f172a;
            --surface-card: #1e293b;
            --border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-emerald: #10b981;
            --accent-cyan: #06b6d4;
            --accent-indigo: #6366f1;
        }}
        body {{
            margin: 0;
            padding: 32px 24px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 28px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 20px;
        }}
        .metric-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .text-emerald {{ color: var(--accent-emerald) !important; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        h3 {{
            margin: 0 0 16px 0;
            font-size: 16px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: #1e293b;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: rgba(51, 65, 85, 0.3);
        }}
        .score-cell {{
            font-weight: 700;
            font-size: 14px;
            color: #38bdf8;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
        }}
        .badge-s {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
        .badge-a {{ background: rgba(6, 182, 212, 0.2); color: #06b6d4; border: 1px solid #06b6d4; }}
        .badge-b {{ background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid #6366f1; }}
        .badge-c {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }}
        .badge-d {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }}
        .badge-pareto {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            margin-left: 4px;
        }}
        .text-muted {{ color: var(--text-secondary); }}
        .bold {{ font-weight: 700; }}
        footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
            padding-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{html.escape(title)}</h1>
            <div class="subtitle">Autonomous AI Agent Benchmark Suite • Multi-Model Inference & Token Economics</div>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Evaluated Models</div>
                <div class="metric-value">{total_models}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Top Performer</div>
                <div class="metric-value text-emerald">{html.escape(top_model)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pareto Optimal Frontier</div>
                <div class="metric-value">{pareto_count} Models</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Most Cost-Effective</div>
                <div class="metric-value">{html.escape(cheapest_model)}</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="card">
                <h3>Composite Benchmark Rankings</h3>
                {chart1_svg}
            </div>
            <div class="card">
                <h3>Cost vs Quality Pareto Frontier</h3>
                {chart2_svg}
            </div>
        </div>

        <div class="card">
            <h3>Unified Agent Evaluation Matrix</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Tier</th>
                        <th>Agent / Model</th>
                        <th>Composite Score</th>
                        <th>Pass@1 (95% CI)</th>
                        <th>Tool F1</th>
                        <th>Latency</th>
                        <th>Cost / 1k Tasks</th>
                        <th>Safety</th>
                        <th>ZK Proof</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>

        {econ_html}

        <footer>
            Agent-Lab v0.6.0 Autonomous Benchmark & Verification Engine • Generated {generated_at}
        </footer>
    </div>
</body>
</html>
"""

    @classmethod
    def export_markdown(cls, entries: List[LeaderboardEntry]) -> str:
        """Render clean Markdown table for READMEs and reports."""
        lines = [
            "# Agent-Lab Unified Benchmark Leaderboard",
            "",
            "| Rank | Tier | Agent / Model | Score | Pass@1 | Tool F1 | Latency | Cost / 1k Tasks | Pareto |",
            "|:----:|:----:|:-------------|:-----:|:------:|:-------:|:-------:|:---------------:|:------:|",
        ]
        for e in entries:
            pareto_str = "★ Yes" if e.is_pareto_optimal else "—"
            lines.append(
                f"| #{e.rank} | **{e.tier}** | {e.agent_id} (`{e.model_id}`) | **{e.composite_score:.1f}** | {e.pass_at_1*100:.1f}% | {e.tool_f1:.4f} | {e.mean_latency_ms:.0f}ms | ${e.cost_per_1k_tasks_usd:.4f} | {pareto_str} |"
            )
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def export_json(cls, entries: List[LeaderboardEntry]) -> str:
        """Export serialized JSON list of leaderboard entries."""
        return json.dumps([e.to_dict() for e in entries], indent=2)
