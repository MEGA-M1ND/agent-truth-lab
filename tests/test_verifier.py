"""M4 tests: the deterministic verifier and the outcome parsing for arms A-C.

Every scenario is built by executing real (clean or injected) tool calls
against a seeded environment, dumping the result, and handing that snapshot
to the verifier — so these tests exercise the same path the harness uses.
"""

from __future__ import annotations

import sqlite3

import pytest

from agent_truth_lab.agent import missions
from agent_truth_lab.agent.missions import Assertion
from agent_truth_lab.environment import db, tools
from agent_truth_lab.injection.injector import Injector
from agent_truth_lab.verification import arms
from agent_truth_lab.verification.arms import Outcome, arm_a, arm_b, arm_c, parse_outcome
from agent_truth_lab.verification.verifier import Verdict, verify

SEED = 42

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def seeded() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, SEED)
    return conn


@pytest.fixture(scope="module")
def mission_set():
    conn = seeded()
    try:
        return missions.build_missions(conn)
    finally:
        conn.close()


def get_mission(mission_set, mission_id):
    return next(m for m in mission_set if m.mission_id == mission_id)


def refund_mission_params(mission):
    """(order_id, amount, customer_id, refund_id) for a refund-archetype mission."""
    refunds = next(a for a in mission.assertions if a.table == "refunds")
    emails = next(a for a in mission.assertions if a.table == "emails_sent")
    order_id = refunds.where["order_id"]
    amount = refunds.expect["amount_paise"]
    customer_id = emails.where["customer_id"]
    refund_id = int(emails.where["related_entity"].split(":")[1])
    return order_id, amount, customer_id, refund_id


def run_calls(mission, calls):
    """Execute (tool_name, args) pairs under the mission's injection plan."""
    conn = seeded()
    clock = db.SimClock()
    injector = Injector(conn, clock, mission.injection)
    results = [injector.call(name, args) for name, args in calls]
    dump = db.dump(conn)
    conn.close()
    return dump, results


def clean_refund_calls(mission):
    order_id, amount, customer_id, refund_id = refund_mission_params(mission)
    return [
        ("issue_refund", {"order_id": order_id, "amount_paise": amount, "reason": "rtn"}),
        (
            "send_customer_email",
            {
                "customer_id": customer_id,
                "template": "refund_completed",
                "related_entity": f"refund:{refund_id}",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_correct_execution_verifies(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    dump, _ = run_calls(mission, clean_refund_calls(mission))

    result = verify(mission, dump, SEED)

    assert result.verdict is Verdict.VERIFIED
    assert all(r.satisfied for r in result.assertion_results)
    assert result.violations == []
    assert result.unexpected_changes == []  # nothing outside the mission's frame
    assert not result.is_partial
    assert result.db_reads > 0


def test_untouched_environment_fails(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    conn = seeded()
    dump = db.dump(conn)
    conn.close()

    result = verify(mission, dump, SEED)

    assert result.verdict is Verdict.FAILED
    assert result.satisfied_assertions == 0
    assert not result.is_partial  # nothing done at all is not a partial completion
    assert result.changes == []


def test_f1_silent_noop_is_caught(mission_set):
    """The tool claimed success; the verifier sees an untouched database."""
    mission = get_mission(mission_set, "m16_refund_f1_noop")
    dump, results = run_calls(mission, clean_refund_calls(mission))

    assert results[0].ok and results[0].http_status == 200  # the lie
    result = verify(mission, dump, SEED)
    assert result.verdict is Verdict.FAILED
    refunds = next(
        r for r in result.assertion_results if r.assertion.table == "refunds"
    )
    assert refunds.actual_count == 0 and "found 0" in refunds.detail


def test_f4_partial_completion_flagged_as_partial(mission_set):
    mission = get_mission(mission_set, "m28_refund_f4_partial")
    dump, results = run_calls(mission, clean_refund_calls(mission))

    assert results[0].ok  # claimed a complete refund
    result = verify(mission, dump, SEED)

    assert result.verdict is Verdict.FAILED
    assert result.is_partial  # refund row + email landed; status and balance did not
    satisfied = {r.assertion.table for r in result.assertion_results if r.satisfied}
    unsatisfied = {r.assertion.table for r in result.assertion_results if not r.satisfied}
    assert "refunds" in satisfied and "emails_sent" in satisfied
    assert {"orders", "customers"} <= unsatisfied


def test_f3_duplicate_detected(mission_set):
    mission = get_mission(mission_set, "m24_refund_f3_dup")
    order_id, amount, _, _ = refund_mission_params(mission)
    args = {"order_id": order_id, "amount_paise": amount, "reason": "rtn"}
    dump, results = run_calls(mission, [("issue_refund", args), ("issue_refund", args)])

    assert not results[0].ok and results[0].http_status == 504  # timeout
    assert results[1].ok  # retry "succeeded"

    result = verify(mission, dump, SEED)
    assert result.verdict is Verdict.FAILED
    assert result.has_duplicate_side_effect
    assert any(v.rule == "single_refund_per_order" for v in result.violations)


def test_f2_wrong_target_reports_collateral_damage(mission_set):
    mission = get_mission(mission_set, "m20_refund_f2_wrong")
    order_id, amount, _, _ = refund_mission_params(mission)
    dump, results = run_calls(
        mission,
        [("issue_refund", {"order_id": order_id, "amount_paise": amount, "reason": "r"})],
    )

    assert results[0].ok and results[0].data["order_id"] == order_id  # the claim
    result = verify(mission, dump, SEED)

    assert result.verdict is Verdict.FAILED
    damaged = result.unexpected_changes
    assert damaged, "wrong_target damage must be reported as out-of-frame changes"
    assert any(c.table == "refunds" and c.kind == "added" for c in damaged)
    assert any(c.table == "orders" and c.kind == "modified" for c in damaged)
    assert all(c.row_id != order_id for c in damaged if c.table == "orders")


def test_invariant_breach_fails_even_when_assertions_hold(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    conn = seeded()
    clock = db.SimClock()
    injector = Injector(conn, clock, {})
    for name, args in clean_refund_calls(mission):
        injector.call(name, args)
    # A refund_completed email pointing at a refund that does not exist.
    tools.send_customer_email(conn, clock, 1, "refund_completed", "refund:999999")
    dump = db.dump(conn)
    conn.close()

    result = verify(mission, dump, SEED)

    assert all(r.satisfied for r in result.assertion_results)
    assert result.verdict is Verdict.FAILED
    assert any(v.rule == "refund_email_references_refund" for v in result.violations)


def test_declined_retry_mission_verifies(mission_set):
    """A correctly-handled decline is a success: count-0 assertions must work."""
    mission = get_mission(mission_set, "m10_retry_declined_clean")
    sub_id = next(
        a for a in mission.assertions if a.table == "subscriptions"
    ).where["id"]
    dump, results = run_calls(
        mission, [("retry_subscription_charge", {"subscription_id": sub_id})]
    )

    assert not results[0].ok and results[0].http_status == 402
    result = verify(mission, dump, SEED)
    assert result.verdict is Verdict.VERIFIED


# ---------------------------------------------------------------------------
# indeterminate paths
# ---------------------------------------------------------------------------


def test_unreadable_snapshot_is_indeterminate(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    result = verify(mission, "this is not sql;", SEED)
    assert result.verdict is Verdict.INDETERMINATE
    assert "unreadable" in result.error


def test_unknown_table_is_indeterminate():
    broken = missions.Mission(
        mission_id="broken",
        instruction="x",
        assertions=(Assertion("wallets", {"id": 1}, {"count": 1}),),
    )
    conn = seeded()
    dump = db.dump(conn)
    conn.close()

    result = verify(broken, dump, SEED)
    assert result.verdict is Verdict.INDETERMINATE
    assert "unknown table" in result.error


def test_unknown_column_is_indeterminate():
    broken = missions.Mission(
        mission_id="broken",
        instruction="x",
        assertions=(Assertion("orders", {"nonexistent": 1}, {"count": 1}),),
    )
    conn = seeded()
    dump = db.dump(conn)
    conn.close()

    result = verify(broken, dump, SEED)
    assert result.verdict is Verdict.INDETERMINATE
    assert "no column" in result.error


def test_verification_is_deterministic_and_independent(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    dump, _ = run_calls(mission, clean_refund_calls(mission))

    first = verify(mission, dump, SEED)
    second = verify(mission, dump, SEED)

    assert first.to_dict() == second.to_dict() or (
        first.verdict == second.verdict
        and first.db_reads == second.db_reads
        and [r.detail for r in first.assertion_results]
        == [r.detail for r in second.assertion_results]
    )


# ---------------------------------------------------------------------------
# Arm A — outcome parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "method"),
    [
        ("TASK_COMPLETE: refunded.", Outcome.SUCCESS, "marker"),
        ("TASK_FAILED: could not refund.", Outcome.FAILURE, "marker"),
        ("Long preamble.\n\nTASK_COMPLETE: done.", Outcome.SUCCESS, "marker"),
        ("TASK_FAILED: first\nactually TASK_COMPLETE: recovered", Outcome.SUCCESS, "marker"),
        ("The refund was completed successfully.", Outcome.SUCCESS, "heuristic"),
        ("I was unable to complete the refund.", Outcome.FAILURE, "heuristic"),
        ("Hmm.", Outcome.INDETERMINATE, "unparseable"),
        (None, Outcome.INDETERMINATE, "no_final_message"),
        ("   ", Outcome.INDETERMINATE, "no_final_message"),
    ],
)
def test_parse_outcome(text, expected, method):
    outcome, parse_method = parse_outcome(text)
    assert (outcome, parse_method) == (expected, method)


def test_arm_a_on_crashed_episode():
    record = {"final_text": None, "stop_reason": "crashed", "tool_calls": []}
    report = arm_a(record)
    assert report.outcome is Outcome.INDETERMINATE
    assert report.parse_method == "no_final_message"


# ---------------------------------------------------------------------------
# Arm B — tool responses
# ---------------------------------------------------------------------------


def test_arm_b_all_ok_is_success():
    record = {
        "tool_calls": [
            {"tool_name": "issue_refund", "result": {"ok": True, "http_status": 200}},
            {"tool_name": "send_customer_email", "result": {"ok": True, "http_status": 200}},
        ]
    }
    assert arm_b(record).outcome is Outcome.SUCCESS


def test_arm_b_any_failure_is_failure():
    record = {
        "tool_calls": [
            {"tool_name": "issue_refund", "result": {"ok": True, "http_status": 200}},
            {"tool_name": "issue_refund", "result": {"ok": False, "http_status": 504}},
        ]
    }
    report = arm_b(record)
    assert report.outcome is Outcome.FAILURE and "504" in report.rationale


def test_arm_b_no_tool_calls_is_failure():
    assert arm_b({"tool_calls": [], "stop_reason": "crashed"}).outcome is Outcome.FAILURE


def test_arm_b_ignores_read_tool_calls():
    """A diagnostic read that legitimately 404s must not fail the mission."""
    record = {
        "tool_calls": [
            {"tool_name": "issue_refund", "result": {"ok": True, "http_status": 200}},
            {"tool_name": "get_settlement", "result": {"ok": False, "http_status": 404}},
        ]
    }
    report = arm_b(record)
    assert report.outcome is Outcome.SUCCESS


def test_arm_b_only_read_calls_is_failure():
    record = {
        "tool_calls": [
            {"tool_name": "get_order", "result": {"ok": True, "http_status": 200}},
        ]
    }
    report = arm_b(record)
    assert report.outcome is Outcome.FAILURE
    assert "no write tool call" in report.rationale


# ---------------------------------------------------------------------------
# Arm C + end-to-end episode evaluation
# ---------------------------------------------------------------------------


def test_arm_c_maps_verdicts(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    dump, _ = run_calls(mission, clean_refund_calls(mission))
    assert arm_c(verify(mission, dump, SEED)).outcome is Outcome.SUCCESS

    conn = seeded()
    untouched = db.dump(conn)
    conn.close()
    assert arm_c(verify(mission, untouched, SEED)).outcome is Outcome.FAILURE


def test_evaluate_episode_disagreement_is_a_false_success(mission_set):
    """The headline phenomenon: A and B say success, C says the state is wrong."""
    mission = get_mission(mission_set, "m16_refund_f1_noop")
    dump, _ = run_calls(mission, clean_refund_calls(mission))
    record = {
        "seed": SEED,
        "db_dump": dump,
        "final_text": "TASK_COMPLETE: refund issued and customer emailed.",
        "stop_reason": "end_turn",
        "crashed": False,
        "tool_calls": [
            {"tool_name": "issue_refund", "args": {}, "result": {"ok": True, "http_status": 200}},
            {
                "tool_name": "send_customer_email",
                "args": {},
                "result": {"ok": True, "http_status": 200},
            },
        ],
        "usage_input_tokens": 100,
        "usage_output_tokens": 20,
        "latency_seconds": 1.0,
    }

    evaluation = arms.evaluate_episode(record, mission)

    assert evaluation.reports["A"].outcome is Outcome.SUCCESS
    assert evaluation.reports["B"].outcome is Outcome.SUCCESS
    assert evaluation.reports["C"].outcome is Outcome.FAILURE
    assert evaluation.ground_truth_satisfied is False
    assert evaluation.failure_mode == "silent_noop"


def test_failure_mode_label(mission_set):
    assert arms.failure_mode_of(get_mission(mission_set, "m01_refund_full_clean")) == "clean"
    assert (
        arms.failure_mode_of(get_mission(mission_set, "m24_refund_f3_dup"))
        == "timeout_then_duplicate"
    )
