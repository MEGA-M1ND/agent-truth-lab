"""Metric computation over evaluated episodes.

The headline metric is the **false success rate**: an arm reported SUCCESS
while the mission's expected_state was actually violated. Its counterpart,
the false failure rate, keeps the experiment honest — an arm that reports
FAILURE for everything would score a perfect false-success rate.

Denominator convention: false-success and false-failure rates are computed
over episodes where ground truth is *determinate*. Episodes the verifier
could not decide are excluded from those rates and reported separately as an
indeterminate rate, so an unverifiable episode is never silently scored as a
pass or a fail.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from agent_truth_lab.verification.arms import EpisodeEvaluation, Outcome
from agent_truth_lab.verification.recovery import RecoveryOutcome


def _rate(numerator: int, denominator: int) -> float:
    """Rate in [0, 1]; an empty denominator yields 0.0 rather than an error."""
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class ArmMetrics:
    """One arm's scorecard over a set of episodes."""

    arm: str
    episodes: int
    determinate: int
    false_successes: int
    false_failures: int
    correct: int
    indeterminate: int

    @property
    def false_success_rate(self) -> float:
        return _rate(self.false_successes, self.determinate)

    @property
    def false_failure_rate(self) -> float:
        return _rate(self.false_failures, self.determinate)

    @property
    def indeterminate_rate(self) -> float:
        return _rate(self.indeterminate, self.episodes)

    @property
    def accuracy(self) -> float:
        return _rate(self.correct, self.determinate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "episodes": self.episodes,
            "determinate": self.determinate,
            "false_successes": self.false_successes,
            "false_failures": self.false_failures,
            "correct": self.correct,
            "indeterminate": self.indeterminate,
            "false_success_rate": self.false_success_rate,
            "false_failure_rate": self.false_failure_rate,
            "indeterminate_rate": self.indeterminate_rate,
            "accuracy": self.accuracy,
        }


def compute_arm_metrics(
    evaluations: list[EpisodeEvaluation], arm: str
) -> ArmMetrics:
    """Score one arm across the given episodes."""
    false_successes = false_failures = correct = indeterminate = determinate = 0
    for evaluation in evaluations:
        report = evaluation.reports.get(arm)
        if report is None:
            continue
        # Arm D changes the state it reports on, so it is scored against the
        # post-recovery state; every other arm against the episode's own.
        truth = evaluation.ground_truth_for(arm)
        if report.outcome is Outcome.INDETERMINATE:
            indeterminate += 1
        if truth is None:
            continue
        determinate += 1
        if report.outcome is Outcome.SUCCESS and not truth:
            false_successes += 1
        elif report.outcome is Outcome.FAILURE and truth:
            false_failures += 1
        elif (report.outcome is Outcome.SUCCESS and truth) or (
            report.outcome is Outcome.FAILURE and not truth
        ):
            correct += 1
    return ArmMetrics(
        arm=arm,
        episodes=len(evaluations),
        determinate=determinate,
        false_successes=false_successes,
        false_failures=false_failures,
        correct=correct,
        indeterminate=indeterminate,
    )


@dataclass(frozen=True)
class StateMetrics:
    """What actually happened to the environment, independent of any arm."""

    episodes: int
    ground_truth_violated: int
    partial_completions: int
    duplicate_side_effects: int
    collateral_damage: int
    crashed: int
    # Episodes the verifier passed even though the episode mutated rows outside
    # the mission's declared frame. This is Arm C's measured blind spot: ground
    # truth is scoped to the mission spec, so damage the spec never mentions
    # cannot fail it. Reported rather than folded into the verdict.
    verified_despite_collateral: int = 0

    @property
    def violation_rate(self) -> float:
        return _rate(self.ground_truth_violated, self.episodes)

    @property
    def partial_completion_rate(self) -> float:
        return _rate(self.partial_completions, self.episodes)

    @property
    def duplicate_side_effect_rate(self) -> float:
        return _rate(self.duplicate_side_effects, self.episodes)

    @property
    def collateral_damage_rate(self) -> float:
        return _rate(self.collateral_damage, self.episodes)

    @property
    def verifier_blind_spot_rate(self) -> float:
        """Share of episodes the verifier passed despite out-of-frame damage."""
        return _rate(self.verified_despite_collateral, self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "ground_truth_violated": self.ground_truth_violated,
            "violation_rate": self.violation_rate,
            "partial_completions": self.partial_completions,
            "partial_completion_rate": self.partial_completion_rate,
            "duplicate_side_effects": self.duplicate_side_effects,
            "duplicate_side_effect_rate": self.duplicate_side_effect_rate,
            "collateral_damage": self.collateral_damage,
            "collateral_damage_rate": self.collateral_damage_rate,
            "verified_despite_collateral": self.verified_despite_collateral,
            "verifier_blind_spot_rate": self.verifier_blind_spot_rate,
            "crashed": self.crashed,
        }


def compute_state_metrics(evaluations: list[EpisodeEvaluation]) -> StateMetrics:
    """Summarize the real-state outcomes across episodes."""
    return StateMetrics(
        episodes=len(evaluations),
        ground_truth_violated=sum(
            1 for e in evaluations if e.ground_truth_satisfied is False
        ),
        partial_completions=sum(1 for e in evaluations if e.verification.is_partial),
        duplicate_side_effects=sum(
            1 for e in evaluations if e.verification.has_duplicate_side_effect
        ),
        collateral_damage=sum(
            1 for e in evaluations if e.verification.unexpected_changes
        ),
        crashed=sum(1 for e in evaluations if e.crashed),
        verified_despite_collateral=sum(
            1
            for e in evaluations
            if e.ground_truth_satisfied is True and e.verification.unexpected_changes
        ),
    )


@dataclass(frozen=True)
class CostMetrics:
    """What the measurement itself cost, per episode."""

    episodes: int
    total_input_tokens: int
    total_output_tokens: int
    mean_episode_latency: float
    mean_verification_latency: float
    mean_verification_db_reads: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "mean_episode_latency": self.mean_episode_latency,
            "mean_verification_latency": self.mean_verification_latency,
            "mean_verification_db_reads": self.mean_verification_db_reads,
        }


def compute_cost_metrics(evaluations: list[EpisodeEvaluation]) -> CostMetrics:
    """Aggregate token, latency, and DB-read costs. Verification uses no tokens."""
    if not evaluations:
        return CostMetrics(0, 0, 0, 0.0, 0.0, 0.0)
    return CostMetrics(
        episodes=len(evaluations),
        total_input_tokens=sum(e.usage_input_tokens for e in evaluations),
        total_output_tokens=sum(e.usage_output_tokens for e in evaluations),
        mean_episode_latency=statistics.fmean(
            e.episode_latency_seconds for e in evaluations
        ),
        mean_verification_latency=statistics.fmean(
            e.verification.latency_seconds for e in evaluations
        ),
        mean_verification_db_reads=statistics.fmean(
            e.verification.db_reads for e in evaluations
        ),
    )


@dataclass(frozen=True)
class RecoveryMetrics:
    """Arm D only: how much of the damage the playbook actually repaired.

    Denominator for the recovery and escalation rates is the set of episodes
    where recovery was *needed* (Arm C found a violation) — recovering an
    already-correct episode is not an achievement.
    """

    episodes: int
    needed: int
    recovered: int
    escalated: int
    rolled_back: int
    damaged: int
    irreversible: int

    @property
    def auto_recovery_rate(self) -> float:
        return _rate(self.recovered, self.needed)

    @property
    def escalation_rate(self) -> float:
        return _rate(self.escalated, self.needed)

    @property
    def recovery_induced_damage_rate(self) -> float:
        """Must be 0.0 — recovery is required never to worsen state."""
        return _rate(self.damaged, self.needed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "needed": self.needed,
            "recovered": self.recovered,
            "escalated": self.escalated,
            "rolled_back": self.rolled_back,
            "damaged": self.damaged,
            "irreversible": self.irreversible,
            "auto_recovery_rate": self.auto_recovery_rate,
            "escalation_rate": self.escalation_rate,
            "recovery_induced_damage_rate": self.recovery_induced_damage_rate,
        }


def compute_recovery_metrics(evaluations: list[EpisodeEvaluation]) -> RecoveryMetrics:
    """Score Arm D's playbook across episodes that carried a recovery attempt."""
    with_recovery = [e for e in evaluations if e.recovery is not None]
    needed = [
        e for e in with_recovery
        if e.recovery.outcome is not RecoveryOutcome.NOT_NEEDED
    ]
    return RecoveryMetrics(
        episodes=len(with_recovery),
        needed=len(needed),
        recovered=sum(1 for e in needed if e.recovery.recovered),
        escalated=sum(1 for e in needed if e.recovery.escalated),
        rolled_back=sum(1 for e in needed if e.recovery.rolled_back),
        damaged=sum(1 for e in needed if e.recovery.caused_damage),
        irreversible=sum(
            1 for e in needed if any(not d.reversible for d in e.recovery.diagnoses)
        ),
    )


@dataclass
class RunMetrics:
    """Everything measured for one seed's full mission set."""

    seed: int
    arms: dict[str, ArmMetrics] = field(default_factory=dict)
    by_mode: dict[str, dict[str, ArmMetrics]] = field(default_factory=dict)
    state: StateMetrics | None = None
    cost: CostMetrics | None = None
    recovery: RecoveryMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "arms": {arm: m.to_dict() for arm, m in self.arms.items()},
            "by_mode": {
                mode: {arm: m.to_dict() for arm, m in arms.items()}
                for mode, arms in self.by_mode.items()
            },
            "state": self.state.to_dict() if self.state else None,
            "cost": self.cost.to_dict() if self.cost else None,
            "recovery": self.recovery.to_dict() if self.recovery else None,
        }

    def headline_rows(self) -> list[dict[str, Any]]:
        """Flat rows for the aggregate CSV."""
        return [
            {
                "seed": self.seed,
                "arm": arm,
                "episodes": m.episodes,
                "false_success_rate": m.false_success_rate,
                "false_failure_rate": m.false_failure_rate,
                "indeterminate_rate": m.indeterminate_rate,
                "accuracy": m.accuracy,
            }
            for arm, m in sorted(self.arms.items())
        ]


def compute_run_metrics(
    evaluations: list[EpisodeEvaluation],
    seed: int,
    arms: tuple[str, ...] = ("A", "B", "C", "D"),
) -> RunMetrics:
    """Score every arm overall and per failure mode for one seed."""
    modes = sorted({e.failure_mode for e in evaluations})
    return RunMetrics(
        seed=seed,
        arms={arm: compute_arm_metrics(evaluations, arm) for arm in arms},
        by_mode={
            mode: {
                arm: compute_arm_metrics(
                    [e for e in evaluations if e.failure_mode == mode], arm
                )
                for arm in arms
            }
            for mode in modes
        },
        state=compute_state_metrics(evaluations),
        cost=compute_cost_metrics(evaluations),
        recovery=compute_recovery_metrics(evaluations),
    )


@dataclass(frozen=True)
class Aggregate:
    """Mean and range of one metric across seeds — the statistical-hygiene view."""

    mean: float
    minimum: float
    maximum: float
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "values": list(self.values),
        }


def aggregate(values: list[float]) -> Aggregate:
    """Mean ± range over per-seed values."""
    if not values:
        return Aggregate(0.0, 0.0, 0.0, ())
    return Aggregate(
        mean=statistics.fmean(values),
        minimum=min(values),
        maximum=max(values),
        values=tuple(values),
    )


def aggregate_runs(runs: list[RunMetrics]) -> dict[str, Any]:
    """Combine per-seed runs into mean ± range per arm, overall and per mode."""
    if not runs:
        return {"seeds": [], "arms": {}, "by_mode": {}}
    arm_names = sorted(runs[0].arms)
    modes = sorted({mode for run in runs for mode in run.by_mode})
    return {
        "seeds": [run.seed for run in runs],
        "arms": {
            arm: {
                "false_success_rate": aggregate(
                    [run.arms[arm].false_success_rate for run in runs if arm in run.arms]
                ).to_dict(),
                "false_failure_rate": aggregate(
                    [run.arms[arm].false_failure_rate for run in runs if arm in run.arms]
                ).to_dict(),
                "indeterminate_rate": aggregate(
                    [run.arms[arm].indeterminate_rate for run in runs if arm in run.arms]
                ).to_dict(),
            }
            for arm in arm_names
        },
        "by_mode": {
            mode: {
                arm: aggregate(
                    [
                        run.by_mode[mode][arm].false_success_rate
                        for run in runs
                        if mode in run.by_mode and arm in run.by_mode[mode]
                    ]
                ).to_dict()
                for arm in arm_names
            }
            for mode in modes
        },
    }
