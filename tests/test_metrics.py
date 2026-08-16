"""M4 tests: metric computation against hand-constructed episodes.

Every metric is checked against episodes whose ground truth and arm verdicts
are set by hand, so a metric bug cannot hide behind a plausible-looking rate.
"""

from __future__ import annotations

import pytest

from agent_truth_lab.agent.missions import Assertion
from agent_truth_lab.harness import metrics
from agent_truth_lab.verification.arms import ArmReport, EpisodeEvaluation, Outcome
from agent_truth_lab.verification.verifier import (
    AssertionResult,
    StateChange,
    Verdict,
    VerificationResult,
)

DUMMY = Assertion("orders", {"id": 1}, {"status": "refunded"})


def make_evaluation(
    *,
    truth: bool | None,
    a: Outcome = Outcome.SUCCESS,
    b: Outcome = Outcome.SUCCESS,
    c: Outcome | None = None,
    mode: str = "clean",
    mission_id: str = "m",
    seed: int = 42,
    partial: bool = False,
    duplicate: bool = False,
    collateral: bool = False,
    crashed: bool = False,
    tokens: tuple[int, int] = (100, 20),
    episode_latency: float = 1.0,
    verify_latency: float = 0.01,
    db_reads: int = 12,
) -> EpisodeEvaluation:
    """Build an evaluation with the exact ground truth and verdicts we want."""
    verdict = {
        True: Verdict.VERIFIED,
        False: Verdict.FAILED,
        None: Verdict.INDETERMINATE,
    }[truth]
    # is_partial is derived from assertion results: some satisfied, some not.
    if partial:
        assertion_results = [
            AssertionResult(DUMMY, True, "ok"),
            AssertionResult(DUMMY, False, "nope"),
        ]
    else:
        satisfied = truth is True
        assertion_results = [AssertionResult(DUMMY, satisfied, "ok")]
    result = VerificationResult(
        verdict=verdict,
        assertion_results=assertion_results,
        violations=[],
        changes=[StateChange("orders", 99, "modified", "x", in_frame=not collateral)],
        db_reads=db_reads,
        latency_seconds=verify_latency,
    )
    if duplicate:
        result.assertion_results.append(
            AssertionResult(DUMMY, False, "dup", expected_count=1, actual_count=2)
        )
    if c is None:
        c = {
            Verdict.VERIFIED: Outcome.SUCCESS,
            Verdict.FAILED: Outcome.FAILURE,
            Verdict.INDETERMINATE: Outcome.INDETERMINATE,
        }[verdict]
    return EpisodeEvaluation(
        mission_id=mission_id,
        seed=seed,
        failure_mode=mode,
        verification=result,
        reports={
            "A": ArmReport("A", a, "test"),
            "B": ArmReport("B", b, "test"),
            "C": ArmReport("C", c, "test"),
        },
        usage_input_tokens=tokens[0],
        usage_output_tokens=tokens[1],
        episode_latency_seconds=episode_latency,
        crashed=crashed,
    )


# ---------------------------------------------------------------------------
# arm metrics
# ---------------------------------------------------------------------------


def test_false_success_rate():
    evaluations = [
        make_evaluation(truth=False, a=Outcome.SUCCESS),  # false success
        make_evaluation(truth=False, a=Outcome.SUCCESS),  # false success
        make_evaluation(truth=False, a=Outcome.FAILURE),  # correct
        make_evaluation(truth=True, a=Outcome.SUCCESS),   # correct
    ]
    m = metrics.compute_arm_metrics(evaluations, "A")
    assert m.false_successes == 2
    assert m.determinate == 4
    assert m.false_success_rate == 0.5
    assert m.correct == 2 and m.accuracy == 0.5


def test_false_failure_rate():
    evaluations = [
        make_evaluation(truth=True, a=Outcome.FAILURE),  # false failure
        make_evaluation(truth=True, a=Outcome.SUCCESS),  # correct
    ]
    m = metrics.compute_arm_metrics(evaluations, "A")
    assert m.false_failures == 1 and m.false_failure_rate == 0.5
    assert m.false_success_rate == 0.0


def test_pessimistic_arm_scores_zero_false_success_but_high_false_failure():
    """The honesty check: always-FAILURE is not a good verifier."""
    evaluations = [make_evaluation(truth=True, a=Outcome.FAILURE) for _ in range(4)]
    m = metrics.compute_arm_metrics(evaluations, "A")
    assert m.false_success_rate == 0.0
    assert m.false_failure_rate == 1.0
    assert m.accuracy == 0.0


def test_indeterminate_ground_truth_excluded_from_denominator():
    evaluations = [
        make_evaluation(truth=None, a=Outcome.SUCCESS),
        make_evaluation(truth=False, a=Outcome.SUCCESS),
    ]
    m = metrics.compute_arm_metrics(evaluations, "A")
    assert m.episodes == 2
    assert m.determinate == 1
    assert m.false_success_rate == 1.0  # 1 of the 1 decidable episode


def test_arm_indeterminate_counted_separately():
    evaluations = [
        make_evaluation(truth=False, a=Outcome.INDETERMINATE),
        make_evaluation(truth=False, a=Outcome.SUCCESS),
    ]
    m = metrics.compute_arm_metrics(evaluations, "A")
    assert m.indeterminate == 1 and m.indeterminate_rate == 0.5
    assert m.false_successes == 1
    assert m.correct == 0  # an indeterminate report is neither right nor wrong


def test_empty_evaluations_are_safe():
    m = metrics.compute_arm_metrics([], "A")
    assert m.false_success_rate == 0.0 and m.accuracy == 0.0
    assert metrics.compute_cost_metrics([]).episodes == 0


def test_missing_arm_is_skipped():
    evaluation = make_evaluation(truth=False)
    del evaluation.reports["C"]
    m = metrics.compute_arm_metrics([evaluation], "C")
    assert m.determinate == 0 and m.false_successes == 0


# ---------------------------------------------------------------------------
# state metrics
# ---------------------------------------------------------------------------


def test_state_metrics():
    evaluations = [
        make_evaluation(truth=False, partial=True),
        make_evaluation(truth=False, duplicate=True),
        make_evaluation(truth=False, collateral=True),
        make_evaluation(truth=True, crashed=True),
    ]
    s = metrics.compute_state_metrics(evaluations)
    assert s.episodes == 4
    assert s.ground_truth_violated == 3 and s.violation_rate == 0.75
    assert s.partial_completions == 1 and s.partial_completion_rate == 0.25
    assert s.duplicate_side_effects == 1
    assert s.collateral_damage == 1
    assert s.crashed == 1


def test_verifier_blind_spot_counts_passing_episodes_with_damage():
    """Damage outside the mission frame does not fail the verdict — measure it."""
    evaluations = [
        make_evaluation(truth=True, collateral=True),    # passed, but damaged
        make_evaluation(truth=True, collateral=False),   # clean pass
        make_evaluation(truth=False, collateral=True),   # already failing
    ]
    s = metrics.compute_state_metrics(evaluations)
    assert s.collateral_damage == 2
    assert s.verified_despite_collateral == 1
    assert s.verifier_blind_spot_rate == pytest.approx(1 / 3)
    assert s.to_dict()["verified_despite_collateral"] == 1


def test_cost_metrics():
    evaluations = [
        make_evaluation(truth=True, tokens=(100, 10), episode_latency=2.0, db_reads=10),
        make_evaluation(truth=True, tokens=(300, 30), episode_latency=4.0, db_reads=20),
    ]
    c = metrics.compute_cost_metrics(evaluations)
    assert c.total_input_tokens == 400 and c.total_output_tokens == 40
    assert c.mean_episode_latency == 3.0
    assert c.mean_verification_db_reads == 15.0


# ---------------------------------------------------------------------------
# run metrics and per-mode breakdown
# ---------------------------------------------------------------------------


def test_run_metrics_per_mode():
    evaluations = [
        make_evaluation(truth=True, a=Outcome.SUCCESS, mode="clean"),
        make_evaluation(truth=True, a=Outcome.SUCCESS, mode="clean"),
        make_evaluation(truth=False, a=Outcome.SUCCESS, mode="silent_noop"),
        make_evaluation(truth=False, a=Outcome.SUCCESS, mode="silent_noop"),
    ]
    run = metrics.compute_run_metrics(evaluations, seed=42)

    assert run.arms["A"].false_success_rate == 0.5
    assert run.by_mode["clean"]["A"].false_success_rate == 0.0
    assert run.by_mode["silent_noop"]["A"].false_success_rate == 1.0
    assert run.by_mode["silent_noop"]["C"].false_success_rate == 0.0
    assert set(run.by_mode) == {"clean", "silent_noop"}


def test_arm_c_false_success_is_zero_by_construction():
    """Arm C is the ground-truth evaluator, so it cannot report a false success."""
    evaluations = [
        make_evaluation(truth=False, mode="silent_noop"),
        make_evaluation(truth=True, mode="clean"),
        make_evaluation(truth=None, mode="stale_read"),
    ]
    run = metrics.compute_run_metrics(evaluations, seed=1)
    assert run.arms["C"].false_success_rate == 0.0
    assert run.arms["C"].false_failure_rate == 0.0
    assert run.arms["C"].indeterminate_rate == pytest.approx(1 / 3)


def test_headline_rows():
    run = metrics.compute_run_metrics(
        [make_evaluation(truth=False, a=Outcome.SUCCESS)], seed=7
    )
    rows = run.headline_rows()
    assert [r["arm"] for r in rows] == ["A", "B", "C", "D"]
    assert rows[0]["seed"] == 7 and rows[0]["false_success_rate"] == 1.0


def test_run_metrics_serializes():
    run = metrics.compute_run_metrics([make_evaluation(truth=False)], seed=3)
    payload = run.to_dict()
    assert payload["seed"] == 3
    assert payload["arms"]["A"]["false_success_rate"] == 1.0
    assert payload["state"]["episodes"] == 1


# ---------------------------------------------------------------------------
# multi-seed aggregation
# ---------------------------------------------------------------------------


def test_aggregate_mean_and_range():
    agg = metrics.aggregate([0.2, 0.4, 0.6])
    assert agg.mean == pytest.approx(0.4)
    assert (agg.minimum, agg.maximum) == (0.2, 0.6)
    assert metrics.aggregate([]).mean == 0.0


def test_aggregate_runs_across_seeds():
    runs = [
        metrics.compute_run_metrics(
            [
                make_evaluation(truth=False, a=Outcome.SUCCESS, mode="silent_noop"),
                make_evaluation(truth=True, a=Outcome.SUCCESS, mode="clean"),
            ],
            seed=42,
        ),
        metrics.compute_run_metrics(
            [
                make_evaluation(truth=False, a=Outcome.FAILURE, mode="silent_noop"),
                make_evaluation(truth=True, a=Outcome.SUCCESS, mode="clean"),
            ],
            seed=43,
        ),
    ]
    combined = metrics.aggregate_runs(runs)

    assert combined["seeds"] == [42, 43]
    fsr = combined["arms"]["A"]["false_success_rate"]
    assert fsr["values"] == [0.5, 0.0]
    assert fsr["mean"] == pytest.approx(0.25)
    assert (fsr["min"], fsr["max"]) == (0.0, 0.5)
    assert combined["by_mode"]["silent_noop"]["A"]["mean"] == pytest.approx(0.5)


def test_aggregate_runs_empty():
    assert metrics.aggregate_runs([])["arms"] == {}
