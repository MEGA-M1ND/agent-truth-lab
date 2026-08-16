"""M5 tests: the Arm D recovery playbook.

The most important test in this file is
`test_recovery_never_worsens_state_across_all_missions` — recovery runs
against every mission in the set and must never leave the environment worse
than it found it, for any failure mode.
"""

from __future__ import annotations

import sqlite3

import pytest

from agent_truth_lab.agent import missions
from agent_truth_lab.agent.missions import Assertion, Mission
from agent_truth_lab.environment import db, tools
from agent_truth_lab.harness import metrics
from agent_truth_lab.injection.injector import EpisodeCrash, Injector
from agent_truth_lab.verification import arms, recovery
from agent_truth_lab.verification.arms import Outcome
from agent_truth_lab.verification.recovery import (
    DivergenceKind,
    RecoveryOutcome,
    diagnose,
    idempotent_refund,
    is_worse,
    recover,
)
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


def get_mission(mission_set, mission_id) -> Mission:
    return next(m for m in mission_set if m.mission_id == mission_id)


def refund_params(mission):
    refunds = next(a for a in mission.assertions if a.table == "refunds")
    emails = next(a for a in mission.assertions if a.table == "emails_sent")
    return (
        refunds.where["order_id"],
        refunds.expect["amount_paise"],
        emails.where["customer_id"],
        int(emails.where["related_entity"].split(":")[1]),
    )


def run_calls(mission, calls, retry_timeouts: bool = True):
    """Execute tool calls under the mission's injection plan; return the dump.

    Mirrors the agent's system prompt by retrying once on a 504 — that is what
    turns F3's masked write into an observable duplicate, so the property test
    below actually exercises the duplicate path.
    """
    conn = seeded()
    clock = db.SimClock()
    injector = Injector(conn, clock, mission.injection)
    for name, args in calls:
        try:
            result = injector.call(name, args)
            if retry_timeouts and result.http_status == 504:
                injector.call(name, args)
        except EpisodeCrash:
            break
    dump = db.dump(conn)
    conn.close()
    return dump


def refund_and_email(mission):
    order_id, amount, customer_id, refund_id = refund_params(mission)
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


def plausible_calls(mission):
    """The tool calls a competent agent would make for any mission archetype."""
    tables = {a.table for a in mission.assertions}
    if "refunds" in tables and "emails_sent" in tables:
        calls = refund_and_email(mission)
        if "cancelled" in mission.instruction:
            # The return-then-refund archetype cancels first; without this the
            # update_order_status injection would never fire.
            orders = next(a for a in mission.assertions if a.table == "orders")
            calls.insert(
                0,
                (
                    "update_order_status",
                    {"order_id": orders.where["id"], "new_status": "cancelled"},
                ),
            )
        return calls
    if "refunds" in tables:
        refunds = next(a for a in mission.assertions if a.table == "refunds")
        return [
            (
                "issue_refund",
                {
                    "order_id": refunds.where["order_id"],
                    "amount_paise": refunds.expect["amount_paise"],
                    "reason": "policy",
                },
            )
        ]
    if "charges" in tables:
        charges = next(a for a in mission.assertions if a.table == "charges")
        calls = [
            ("retry_subscription_charge", {"subscription_id": charges.where["subscription_id"]})
        ]
        emails = next((a for a in mission.assertions if a.table == "emails_sent"), None)
        if emails is not None and emails.expect.get("count") == 1:
            calls.append(
                (
                    "send_customer_email",
                    {
                        "customer_id": emails.where["customer_id"],
                        "template": emails.where["template"],
                        "related_entity": emails.where["related_entity"],
                    },
                )
            )
        return calls
    if "settlements" in tables:
        settlements = next(a for a in mission.assertions if a.table == "settlements")
        return [
            (
                "create_settlement",
                {
                    "merchant_day": settlements.where["merchant_day"],
                    "mark_processed": settlements.expect.get("status") == "processed",
                },
            )
        ]
    orders = next(a for a in mission.assertions if a.table == "orders")
    emails = next((a for a in mission.assertions if a.table == "emails_sent"), None)
    calls = [
        (
            "update_order_status",
            {"order_id": orders.where["id"], "new_status": orders.expect["status"]},
        )
    ]
    if emails is not None:
        calls.append(
            (
                "send_customer_email",
                {
                    "customer_id": emails.where["customer_id"],
                    "template": emails.where["template"],
                    "related_entity": emails.where["related_entity"],
                },
            )
        )
    return calls


# ---------------------------------------------------------------------------
# THE core guarantee
# ---------------------------------------------------------------------------


def test_recovery_never_worsens_state_across_all_missions(mission_set):
    """Recovery must never leave the environment worse off — for any mode."""
    for mission in mission_set:
        dump = run_calls(mission, plausible_calls(mission))
        result = recover(mission, dump, SEED)

        if result.outcome is RecoveryOutcome.NOT_NEEDED:
            continue
        pre, post = result.pre_verification, result.post_verification
        assert not is_worse(post, pre), (
            f"{mission.mission_id}: recovery worsened state "
            f"({pre.satisfied_assertions}->{post.satisfied_assertions} assertions, "
            f"{len(pre.violations)}->{len(post.violations)} violations)"
        )
        assert not result.caused_damage, f"{mission.mission_id} reported damage"
        # The dump it hands back must always be loadable and re-verifiable.
        assert verify(mission, result.db_dump, SEED).verdict is not Verdict.INDETERMINATE


def test_rollback_restores_original_state_when_repair_would_worsen(mission_set):
    """A repair that makes things worse is discarded wholesale."""
    mission = get_mission(mission_set, "m01_refund_full_clean")
    dump = run_calls(mission, [])  # nothing done: recovery is needed

    # A repair plan guaranteed to damage: delete rows instead of adding them.
    def sabotage(conn, _clock, _result, _mission):
        conn.execute("DELETE FROM orders WHERE id IN (SELECT id FROM orders LIMIT 5)")
        conn.commit()
        return [recovery.RecoveryAction("sabotage", "orders", None, "removed rows")]

    original = recovery._repair_refunds
    recovery._repair_refunds = sabotage
    try:
        result = recover(mission, dump, SEED)
    finally:
        recovery._repair_refunds = original

    assert result.rolled_back
    assert result.outcome is RecoveryOutcome.ESCALATED
    assert result.db_dump == dump  # byte-for-byte the original snapshot
    assert not result.caused_damage
    assert "rolled back: True" in result.incident


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------


def test_diagnose_missing_effect(mission_set):
    mission = get_mission(mission_set, "m16_refund_f1_noop")
    dump = run_calls(mission, refund_and_email(mission))
    diagnoses = diagnose(verify(mission, dump, SEED))
    assert any(d.kind is DivergenceKind.MISSING_EFFECT for d in diagnoses)


def test_diagnose_duplicate(mission_set):
    mission = get_mission(mission_set, "m24_refund_f3_dup")
    order_id, amount, _, _ = refund_params(mission)
    args = {"order_id": order_id, "amount_paise": amount, "reason": "rtn"}
    dump = run_calls(mission, [("issue_refund", args), ("issue_refund", args)])
    diagnoses = diagnose(verify(mission, dump, SEED))
    assert any(d.kind is DivergenceKind.DUPLICATE for d in diagnoses)


def test_diagnose_wrong_target(mission_set):
    mission = get_mission(mission_set, "m20_refund_f2_wrong")
    dump = run_calls(mission, refund_and_email(mission))
    diagnoses = diagnose(verify(mission, dump, SEED))
    assert any(d.kind is DivergenceKind.WRONG_TARGET for d in diagnoses)


def test_diagnose_partial(mission_set):
    mission = get_mission(mission_set, "m28_refund_f4_partial")
    dump = run_calls(mission, refund_and_email(mission))
    diagnoses = diagnose(verify(mission, dump, SEED))
    assert any(d.kind is DivergenceKind.PARTIAL for d in diagnoses)


def test_diagnose_marks_stray_email_irreversible(mission_set):
    """A missing email can be sent; a misdirected one cannot be recalled."""
    mission = get_mission(mission_set, "m23_refund_email_f2_wrong")
    dump = run_calls(mission, refund_and_email(mission))
    diagnoses = diagnose(verify(mission, dump, SEED))

    stray = [
        d for d in diagnoses
        if d.table == "emails_sent" and d.kind is DivergenceKind.WRONG_TARGET
    ]
    missing = [
        d for d in diagnoses
        if d.table == "emails_sent" and d.kind is DivergenceKind.MISSING_EFFECT
    ]
    assert stray and all(not d.reversible for d in stray)
    assert missing and all(d.reversible for d in missing)


# ---------------------------------------------------------------------------
# repair paths
# ---------------------------------------------------------------------------


def test_recovers_silent_noop(mission_set):
    """F1: the tool lied and did nothing — recovery re-applies the whole mission."""
    mission = get_mission(mission_set, "m16_refund_f1_noop")
    dump = run_calls(mission, refund_and_email(mission))
    assert verify(mission, dump, SEED).verdict is Verdict.FAILED

    result = recover(mission, dump, SEED)

    assert result.recovered
    assert result.post_verification.verdict is Verdict.VERIFIED
    assert any(a.action == "retry_refund_idempotent" for a in result.actions)


def test_recovers_partial_completion(mission_set):
    """F4: refund row exists, status and balance do not — finish the job."""
    mission = get_mission(mission_set, "m28_refund_f4_partial")
    dump = run_calls(mission, refund_and_email(mission))

    result = recover(mission, dump, SEED)

    assert result.recovered
    assert any(a.action == "correct_state_value" and a.table == "orders" for a in result.actions)
    post = recovery_snapshot(result)
    order_id, _, customer_id, _ = refund_params(mission)
    assert post.execute(
        "SELECT status FROM orders WHERE id = ?", (order_id,)
    ).fetchone()["status"] == "refunded"
    post.close()


def recovery_snapshot(result) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(result.db_dump)
    return conn


def test_recovers_duplicate_by_compensating(mission_set):
    """F3: the duplicate refund is reversed and the double credit clawed back."""
    mission = get_mission(mission_set, "m24_refund_f3_dup")
    order_id, amount, customer_id, _ = refund_params(mission)
    args = {"order_id": order_id, "amount_paise": amount, "reason": "rtn"}
    dump = run_calls(mission, [("issue_refund", args), ("issue_refund", args)])

    result = recover(mission, dump, SEED)

    assert any(a.action == "compensate_duplicate_refund" for a in result.actions)
    post = recovery_snapshot(result)
    refunds = post.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order_id,)
    ).fetchone()["n"]
    assert refunds == 1  # the duplicate is gone
    baseline = seeded()
    expected = (
        baseline.execute(
            "SELECT balance_paise FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()["balance_paise"]
        + amount
    )
    assert post.execute(
        "SELECT balance_paise FROM customers WHERE id = ?", (customer_id,)
    ).fetchone()["balance_paise"] == expected
    baseline.close()
    post.close()


def test_recovers_wrong_target_by_reversing_stray_write(mission_set):
    """F2: undo the damage on the wrong order, then apply to the right one."""
    mission = get_mission(mission_set, "m20_refund_f2_wrong")
    dump = run_calls(mission, refund_and_email(mission))
    pre = verify(mission, dump, SEED)
    damaged_orders = {c.row_id for c in pre.unexpected_changes if c.table == "orders"}
    assert damaged_orders

    result = recover(mission, dump, SEED)

    assert any(a.action in ("reverse_stray_write", "restore_baseline_row")
               for a in result.actions)
    post = recovery_snapshot(result)
    baseline = seeded()
    for order_id in damaged_orders:
        restored = post.execute(
            "SELECT status FROM orders WHERE id = ?", (order_id,)
        ).fetchone()["status"]
        original = baseline.execute(
            "SELECT status FROM orders WHERE id = ?", (order_id,)
        ).fetchone()["status"]
        assert restored == original, "collateral damage was not reversed"
    baseline.close()
    post.close()


def test_recovers_invariant_violation_in_settlement(mission_set):
    """F6: the settlement forgot to subtract refunds — recompute it."""
    mission = get_mission(mission_set, "m35_settlement_f6_inv")
    dump = run_calls(mission, plausible_calls(mission))
    assert verify(mission, dump, SEED).verdict is Verdict.FAILED

    result = recover(mission, dump, SEED)

    assert result.recovered
    assert any(a.action == "recompute_settlement" for a in result.actions)
    assert result.post_verification.violations == []


def test_recovers_crashed_episode(mission_set):
    """F7: the process died after the write; recovery finishes the mission."""
    mission = get_mission(mission_set, "m38_refund_f7_crash")
    dump = run_calls(mission, refund_and_email(mission))  # crash aborts after step 1

    result = recover(mission, dump, SEED)

    assert result.recovered
    assert result.post_verification.verdict is Verdict.VERIFIED


def test_escalates_irreversible_email_damage(mission_set):
    """A misdirected email cannot be recalled — that must escalate, not pretend."""
    mission = get_mission(mission_set, "m23_refund_email_f2_wrong")
    dump = run_calls(mission, refund_and_email(mission))

    result = recover(mission, dump, SEED)

    assert result.escalated
    assert result.incident is not None
    assert "IRREVERSIBLE" in result.incident
    assert "REQUIRES HUMAN REVIEW" in result.incident
    assert any(a.action == "escalate_irreversible" for a in result.actions)


def test_no_recovery_needed_when_state_is_correct(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    dump = run_calls(mission, refund_and_email(mission))

    result = recover(mission, dump, SEED)

    assert result.outcome is RecoveryOutcome.NOT_NEEDED
    assert result.actions == []
    assert result.db_dump == dump


def test_indeterminate_state_escalates(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    result = recover(mission, "not valid sql;", SEED)
    assert result.escalated and result.incident is not None


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_idempotent_refund_applies_once():
    conn = seeded()
    clock = db.SimClock()
    order = conn.execute(
        "SELECT * FROM orders WHERE status = 'delivered' ORDER BY id LIMIT 1"
    ).fetchone()
    balance_before = conn.execute(
        "SELECT balance_paise FROM customers WHERE id = ?", (order["customer_id"],)
    ).fetchone()["balance_paise"]

    first = idempotent_refund(
        conn, clock, order["id"], order["amount_paise"], "recovery", "key-1"
    )
    second = idempotent_refund(
        conn, clock, order["id"], order["amount_paise"], "recovery", "key-1"
    )

    assert first[0] is True and second[0] is False
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order["id"],)
    ).fetchone()["n"] == 1
    assert conn.execute(
        "SELECT balance_paise FROM customers WHERE id = ?", (order["customer_id"],)
    ).fetchone()["balance_paise"] == balance_before + order["amount_paise"]
    conn.close()


# ---------------------------------------------------------------------------
# is_worse comparator
# ---------------------------------------------------------------------------


def test_is_worse_detects_regressions(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    good = verify(mission, run_calls(mission, refund_and_email(mission)), SEED)
    bad = verify(mission, run_calls(mission, []), SEED)

    assert is_worse(bad, good)
    assert not is_worse(good, bad)
    assert not is_worse(good, good)


def test_is_worse_flags_new_invariant_violations(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    conn = seeded()
    clock = db.SimClock()
    injector = Injector(conn, clock, {})
    for name, args in refund_and_email(mission):
        injector.call(name, args)
    clean = verify(mission, db.dump(conn), SEED)
    tools.send_customer_email(conn, clock, 1, "refund_completed", "refund:999999")
    broken = verify(mission, db.dump(conn), SEED)
    conn.close()

    assert is_worse(broken, clean)


# ---------------------------------------------------------------------------
# Arm D wiring and metrics
# ---------------------------------------------------------------------------


def make_record(mission, dump, final_text="TASK_COMPLETE: done."):
    return {
        "seed": SEED,
        "db_dump": dump,
        "final_text": final_text,
        "stop_reason": "end_turn",
        "crashed": False,
        "tool_calls": [
            {"tool_name": "issue_refund", "args": {}, "result": {"ok": True, "http_status": 200}}
        ],
        "usage_input_tokens": 100,
        "usage_output_tokens": 20,
        "latency_seconds": 1.0,
    }


def test_arm_d_reports_success_after_recovery(mission_set):
    mission = get_mission(mission_set, "m16_refund_f1_noop")
    dump = run_calls(mission, refund_and_email(mission))
    evaluation = arms.evaluate_episode(make_record(mission, dump), mission)

    assert evaluation.reports["C"].outcome is Outcome.FAILURE
    assert evaluation.reports["D"].outcome is Outcome.SUCCESS
    assert "auto-recovered" in evaluation.reports["D"].rationale
    # Arm C's own measurement is untouched by the repair.
    assert evaluation.ground_truth_satisfied is False
    assert evaluation.ground_truth_for("D") is True
    assert evaluation.ground_truth_for("C") is False


def test_arm_d_reports_failure_on_escalation(mission_set):
    mission = get_mission(mission_set, "m23_refund_email_f2_wrong")
    dump = run_calls(mission, refund_and_email(mission))
    evaluation = arms.evaluate_episode(make_record(mission, dump), mission)

    assert evaluation.reports["D"].outcome is Outcome.FAILURE
    assert "escalated" in evaluation.reports["D"].rationale


def test_evaluate_episode_can_skip_recovery(mission_set):
    mission = get_mission(mission_set, "m01_refund_full_clean")
    dump = run_calls(mission, refund_and_email(mission))
    evaluation = arms.evaluate_episode(
        make_record(mission, dump), mission, include_recovery=False
    )
    assert "D" not in evaluation.reports and evaluation.recovery is None


def test_recovery_metrics(mission_set):
    evaluations = []
    for mission_id in (
        "m01_refund_full_clean",       # NOT_NEEDED
        "m16_refund_f1_noop",          # RECOVERED
        "m23_refund_email_f2_wrong",   # ESCALATED (irreversible email)
    ):
        mission = get_mission(mission_set, mission_id)
        dump = run_calls(mission, refund_and_email(mission))
        evaluations.append(arms.evaluate_episode(make_record(mission, dump), mission))

    m = metrics.compute_recovery_metrics(evaluations)

    assert m.episodes == 3
    assert m.needed == 2
    assert m.recovered == 1 and m.escalated == 1
    assert m.auto_recovery_rate == 0.5
    assert m.escalation_rate == 0.5
    assert m.recovery_induced_damage_rate == 0.0
    assert m.irreversible >= 1


def test_arm_d_included_in_run_metrics(mission_set):
    mission = get_mission(mission_set, "m16_refund_f1_noop")
    dump = run_calls(mission, refund_and_email(mission))
    evaluation = arms.evaluate_episode(make_record(mission, dump), mission)

    run = metrics.compute_run_metrics([evaluation], seed=SEED)

    assert set(run.arms) == {"A", "B", "C", "D"}
    assert run.arms["A"].false_success_rate == 1.0  # agent claimed success, state wrong
    assert run.arms["D"].false_success_rate == 0.0  # D fixed it, then verified
    assert run.recovery.recovered == 1
    assert run.to_dict()["recovery"]["auto_recovery_rate"] == 1.0


def test_incident_report_is_structured_and_deterministic(mission_set):
    mission = get_mission(mission_set, "m23_refund_email_f2_wrong")
    dump = run_calls(mission, refund_and_email(mission))

    first = recover(mission, dump, SEED)
    second = recover(mission, dump, SEED)

    assert first.incident == second.incident
    assert first.incident.startswith("INCIDENT: mission m23_refund_email_f2_wrong")
    for section in ("divergences:", "actions attempted:", "REQUIRES HUMAN REVIEW"):
        assert section in first.incident


def test_unrepairable_assertion_escalates_without_synthesizing_rows():
    """Recovery refuses to invent a row it cannot safely construct."""
    mission = Mission(
        mission_id="synthetic",
        instruction="x",
        assertions=(Assertion("subscriptions", {"id": 99999}, {"status": "active"}),),
    )
    conn = seeded()
    dump = db.dump(conn)
    conn.close()

    result = recover(mission, dump, SEED)

    assert result.escalated
    assert any(a.action == "escalate_unrepairable" and not a.applied for a in result.actions)
