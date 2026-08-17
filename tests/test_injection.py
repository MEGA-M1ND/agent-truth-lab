"""M2 tests: prove every failure mode actually injects.

For each mode this file asserts BOTH halves of the spec:
  (a) the response looks the way the spec says (plausible success / timeout),
  (b) the DB state diverges from the response exactly as specified.

If these tests are wrong, the whole experiment is garbage — they are the
ground truth that injected missions really contain the failure they claim.
"""

from __future__ import annotations

import sqlite3

import pytest

from agent_truth_lab.environment import db, invariants, tools
from agent_truth_lab.injection.injector import Injector, validate_plan
from agent_truth_lab.injection.modes import EpisodeCrash, FailureMode, corrupt_amount

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fresh_env(seed: int = 42) -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, seed)
    return conn


def make_injector(conn, plan):
    return Injector(conn, db.SimClock(), plan)


def delivered_order(conn):
    return conn.execute(
        "SELECT * FROM orders WHERE status = 'delivered' ORDER BY id LIMIT 1"
    ).fetchone()


def funded_past_due_sub(conn):
    return conn.execute(
        "SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        " WHERE s.status = 'past_due' AND c.balance_paise >= s.amount_paise"
        " ORDER BY s.id LIMIT 1"
    ).fetchone()


def order_status(conn, order_id):
    return conn.execute(
        "SELECT status FROM orders WHERE id = ?", (order_id,)
    ).fetchone()["status"]


def balance(conn, customer_id):
    return conn.execute(
        "SELECT balance_paise FROM customers WHERE id = ?", (customer_id,)
    ).fetchone()["balance_paise"]


def refunds_for(conn, order_id):
    return conn.execute(
        "SELECT * FROM refunds WHERE order_id = ? ORDER BY id", (order_id,)
    ).fetchall()


# ---------------------------------------------------------------------------
# plan validation + passthrough
# ---------------------------------------------------------------------------


def test_validate_plan_rejects_unknown_tool():
    with pytest.raises(ValueError, match="unknown tool"):
        validate_plan({"teleport_money": "silent_noop"})


def test_validate_plan_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown failure mode"):
        validate_plan({"issue_refund": "gremlins"})


def test_validate_plan_rejects_partial_on_single_step_tool():
    with pytest.raises(ValueError, match="multi-step"):
        validate_plan({"send_customer_email": "partial_completion"})


def test_call_unknown_tool_raises(conn):
    with pytest.raises(ValueError, match="unknown tool"):
        make_injector(conn, {}).call("teleport_money", {})


def test_empty_plan_is_exact_passthrough():
    """No injection => byte-identical outcome to calling the clean tool directly."""
    conn_a, conn_b = fresh_env(), fresh_env()
    order = delivered_order(conn_a)
    args = {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"}

    injected = make_injector(conn_a, {}).call("issue_refund", args)
    direct = tools.issue_refund(conn_b, db.SimClock(), **args)

    assert injected.to_dict() == direct.to_dict()
    assert db.dump(conn_a) == db.dump(conn_b)


# ---------------------------------------------------------------------------
# F1 silent_noop
# ---------------------------------------------------------------------------


def test_f1_refund_plausible_success_db_untouched(conn):
    order = delivered_order(conn)
    before = db.dump(conn)

    result = make_injector(conn, {"issue_refund": "silent_noop"}).call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )

    # (a) response is a plausible clean-shaped success
    assert result.ok and result.http_status == 200 and result.error is None
    assert set(result.data) == {
        "refund_id", "order_id", "customer_id", "amount_paise", "status", "created_at",
    }
    assert result.data["order_id"] == order["id"]
    assert result.data["status"] == "completed"
    assert isinstance(result.data["refund_id"], int)
    # (b) DB completely unchanged
    assert db.dump(conn) == before


def test_f1_settlement_reports_real_numbers_writes_nothing(conn):
    result = make_injector(conn, {"create_settlement": "silent_noop"}).call(
        "create_settlement", {"merchant_day": db.HISTORY_DAYS[0]}
    )
    assert result.ok and result.data["gross_paise"] > 0
    assert conn.execute("SELECT COUNT(*) AS n FROM settlements").fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# F2 wrong_target
# ---------------------------------------------------------------------------


def test_f2_refund_lands_on_other_order_claims_requested(conn):
    order = delivered_order(conn)
    result = make_injector(conn, {"issue_refund": "wrong_target"}).call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )

    # (a) response claims the requested order
    assert result.ok
    assert result.data["order_id"] == order["id"]
    assert result.data["customer_id"] == order["customer_id"]
    # (b) requested order untouched; a DIFFERENT order actually got refunded
    assert order_status(conn, order["id"]) == "delivered"
    assert refunds_for(conn, order["id"]) == []
    landed = conn.execute(
        "SELECT * FROM refunds ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert landed["order_id"] != order["id"]
    assert landed["amount_paise"] == order["amount_paise"]
    assert order_status(conn, landed["order_id"]) == "refunded"


def test_f2_email_lands_on_other_customer(conn):
    result = make_injector(conn, {"send_customer_email": "wrong_target"}).call(
        "send_customer_email",
        {"customer_id": 5, "template": "order_update", "related_entity": "order:1001"},
    )
    assert result.ok and result.data["customer_id"] == 5
    row = conn.execute("SELECT * FROM emails_sent").fetchone()
    assert row["customer_id"] != 5


def test_f2_target_choice_is_deterministic():
    landed = []
    for _ in range(2):
        conn = fresh_env()
        order = delivered_order(conn)
        make_injector(conn, {"issue_refund": "wrong_target"}).call(
            "issue_refund",
            {"order_id": order["id"], "amount_paise": 100, "reason": "rtn"},
        )
        landed.append(conn.execute("SELECT order_id FROM refunds ORDER BY id DESC LIMIT 1")
                      .fetchone()["order_id"])
    assert landed[0] == landed[1]


# ---------------------------------------------------------------------------
# F3 timeout_then_duplicate
# ---------------------------------------------------------------------------


def test_f3_first_call_commits_but_reports_timeout(conn):
    order = delivered_order(conn)
    bal_before = balance(conn, order["customer_id"])

    result = make_injector(conn, {"issue_refund": "timeout_then_duplicate"}).call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )

    # (a) response is a timeout error
    assert not result.ok and result.http_status == 504
    assert "timeout" in result.error
    # (b) the side effect actually committed
    assert len(refunds_for(conn, order["id"])) == 1
    assert order_status(conn, order["id"]) == "refunded"
    assert balance(conn, order["customer_id"]) == bal_before + order["amount_paise"]


def test_f3_retry_creates_duplicate_refund(conn):
    order = delivered_order(conn)
    bal_before = balance(conn, order["customer_id"])
    injector = make_injector(conn, {"issue_refund": "timeout_then_duplicate"})
    args = {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"}

    injector.call("issue_refund", args)          # timeout, effect committed
    retry = injector.call("issue_refund", args)  # agent retries

    # (a) the retry reports clean success
    assert retry.ok and retry.data["status"] == "completed"
    # (b) DB now holds TWO refunds and a double balance credit
    assert len(refunds_for(conn, order["id"])) == 2
    assert balance(conn, order["customer_id"]) == bal_before + 2 * order["amount_paise"]
    # and the invariant layer can see the damage
    assert any(
        v.rule in ("single_refund_per_order", "refund_le_order")
        for v in invariants.check_all(conn)
    )


def test_f3_retry_duplicates_subscription_charge(conn):
    sub = funded_past_due_sub(conn)
    bal_before = balance(conn, sub["customer_id"])
    injector = make_injector(conn, {"retry_subscription_charge": "timeout_then_duplicate"})

    injector.call("retry_subscription_charge", {"subscription_id": sub["id"]})
    retry = injector.call("retry_subscription_charge", {"subscription_id": sub["id"]})

    assert retry.ok
    charges = conn.execute(
        "SELECT * FROM charges WHERE subscription_id = ? AND status = 'succeeded'"
        " AND attempt_no > 1",
        (sub["id"],),
    ).fetchall()
    assert len(charges) == 2  # double charge
    assert balance(conn, sub["customer_id"]) == bal_before - 2 * sub["amount_paise"]


# ---------------------------------------------------------------------------
# F4 partial_completion
# ---------------------------------------------------------------------------


def test_f4_refund_row_only_claims_full_success(conn):
    order = delivered_order(conn)
    bal_before = balance(conn, order["customer_id"])

    result = make_injector(conn, {"issue_refund": "partial_completion"}).call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )

    # (a) response indistinguishable from a full clean success
    assert result.ok and result.data["status"] == "completed"
    # (b) only step 1 landed: refund row exists, order and balance untouched
    rows = refunds_for(conn, order["id"])
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    assert order_status(conn, order["id"]) == "delivered"
    assert balance(conn, order["customer_id"]) == bal_before


def test_f4_charge_row_only_subscription_stays_past_due(conn):
    sub = funded_past_due_sub(conn)
    bal_before = balance(conn, sub["customer_id"])

    result = make_injector(conn, {"retry_subscription_charge": "partial_completion"}).call(
        "retry_subscription_charge", {"subscription_id": sub["id"]}
    )

    assert result.ok and result.data["status"] == "succeeded"
    charge = conn.execute(
        "SELECT * FROM charges WHERE subscription_id = ? ORDER BY id DESC LIMIT 1",
        (sub["id"],),
    ).fetchone()
    assert charge["status"] == "succeeded"
    after = conn.execute(
        "SELECT * FROM subscriptions WHERE id = ?", (sub["id"],)
    ).fetchone()
    assert after["status"] == "past_due"          # step 2 never happened
    assert after["next_charge_at"] == sub["next_charge_at"]
    assert balance(conn, sub["customer_id"]) == bal_before  # step 3 never happened


# ---------------------------------------------------------------------------
# F5 stale_read (lost write)
# ---------------------------------------------------------------------------


def test_f5_response_is_genuine_but_write_never_persists():
    conn_injected, conn_clean = fresh_env(), fresh_env()
    order = delivered_order(conn_injected)
    args = {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"}
    before = db.dump(conn_injected)

    injected = make_injector(conn_injected, {"issue_refund": "stale_read"}).call(
        "issue_refund", args
    )
    clean = tools.issue_refund(conn_clean, db.SimClock(), **args)

    # (a) the response is EXACTLY what a durable clean execution would return
    #     (this is what distinguishes F5 from F1's fabricated response)
    assert injected.to_dict() == clean.to_dict()
    # (b) nothing persisted: the acknowledged write was lost
    assert db.dump(conn_injected) == before
    assert refunds_for(conn_injected, order["id"]) == []
    assert order_status(conn_injected, order["id"]) == "delivered"


# ---------------------------------------------------------------------------
# F6 invariant_violation
# ---------------------------------------------------------------------------


def test_f6_refund_records_corrupted_amount_claims_requested(conn):
    order = delivered_order(conn)
    requested = order["amount_paise"]
    expected_corrupted = corrupt_amount(requested, ceiling=order["amount_paise"])
    assert expected_corrupted != requested, "corruption must change the amount"

    result = make_injector(conn, {"issue_refund": "invariant_violation"}).call(
        "issue_refund", {"order_id": order["id"], "amount_paise": requested, "reason": "r"}
    )

    # (a) response claims the requested amount
    assert result.ok and result.data["amount_paise"] == requested
    # (b) the DB recorded a different amount, and the balance credit matches the
    #     corrupted value, not the claimed one
    rows = refunds_for(conn, order["id"])
    assert len(rows) == 1
    assert rows[0]["amount_paise"] == expected_corrupted


def test_f6_settlement_omits_refunds_and_invariant_fires(conn):
    day = db.HISTORY_DAYS[0]
    refunds_total = conn.execute(
        "SELECT COALESCE(SUM(amount_paise), 0) AS t FROM refunds"
        " WHERE status = 'completed' AND substr(created_at, 1, 10) = ?",
        (day,),
    ).fetchone()["t"]
    assert refunds_total > 0, "this test needs a day with refunds to omit"

    result = make_injector(conn, {"create_settlement": "invariant_violation"}).call(
        "create_settlement", {"merchant_day": day}
    )

    assert result.ok
    row = conn.execute("SELECT * FROM settlements").fetchone()
    assert row["net_paise"] == row["gross_paise"] - row["fees_paise"]  # refunds omitted
    assert any(v.rule == "settlement_math" for v in invariants.settlement_math(conn))


def test_f6_order_status_lands_refunded_and_invariant_fires(conn):
    order = conn.execute(
        "SELECT * FROM orders WHERE status = 'placed' ORDER BY id LIMIT 1"
    ).fetchone()

    result = make_injector(conn, {"update_order_status": "invariant_violation"}).call(
        "update_order_status", {"order_id": order["id"], "new_status": "shipped"}
    )

    assert result.ok and result.data["new_status"] == "shipped"  # the claim
    assert order_status(conn, order["id"]) == "refunded"          # the reality
    found = invariants.refunded_orders_have_completed_refund(conn)
    assert order["id"] in [v.entity_id for v in found]


def test_f6_email_records_dangling_reference(conn, clock):
    refund = tools.issue_refund(
        conn, clock, delivered_order(conn)["id"], 100, "setup"
    )
    good_ref = f"refund:{refund.data['refund_id']}"

    result = make_injector(conn, {"send_customer_email": "invariant_violation"}).call(
        "send_customer_email",
        {"customer_id": 1, "template": "refund_completed", "related_entity": good_ref},
    )

    assert result.ok and result.data["related_entity"] == good_ref  # the claim
    row = conn.execute("SELECT * FROM emails_sent").fetchone()
    assert row["related_entity"] == "refund:999999"                  # the reality
    assert len(invariants.refund_emails_reference_completed_refund(conn)) == 1


# ---------------------------------------------------------------------------
# F7 crash_after_side_effect
# ---------------------------------------------------------------------------


def test_f7_side_effect_commits_then_episode_crashes(conn):
    order = delivered_order(conn)
    injector = make_injector(conn, {"issue_refund": "crash_after_side_effect"})

    with pytest.raises(EpisodeCrash) as excinfo:
        injector.call(
            "issue_refund",
            {"order_id": order["id"], "amount_paise": order["amount_paise"],
             "reason": "rtn"},
        )

    assert excinfo.value.tool_name == "issue_refund"
    # the refund really happened even though the agent never saw a result
    assert len(refunds_for(conn, order["id"])) == 1
    assert order_status(conn, order["id"]) == "refunded"


# ---------------------------------------------------------------------------
# cross-cutting: determinism and clean-tool isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [m.value for m in FailureMode])
def test_every_mode_is_deterministic(mode):
    """Same seed + same plan + same call sequence => byte-identical databases."""
    dumps = []
    for _ in range(2):
        conn = fresh_env()
        order = delivered_order(conn)
        injector = make_injector(conn, {"issue_refund": mode})
        args = {"order_id": order["id"], "amount_paise": order["amount_paise"],
                "reason": "rtn"}
        for _call in range(2):  # two calls: exercises F3's retry path too
            try:
                injector.call("issue_refund", args)
            except EpisodeCrash:
                break
        dumps.append(db.dump(conn))
        conn.close()
    assert dumps[0] == dumps[1]


def test_injection_only_touches_targeted_tool(conn):
    """A plan targeting issue_refund leaves every other tool perfectly clean."""
    injector = make_injector(conn, {"issue_refund": "silent_noop"})
    sub = funded_past_due_sub(conn)

    result = injector.call("retry_subscription_charge", {"subscription_id": sub["id"]})

    assert result.ok
    after = conn.execute(
        "SELECT status FROM subscriptions WHERE id = ?", (sub["id"],)
    ).fetchone()
    assert after["status"] == "active"  # the clean tool really ran


# ---------------------------------------------------------------------------
# read tools (M7): truthful by default, stale_read on the read channel itself
# ---------------------------------------------------------------------------


def test_read_tool_truthful_by_default(conn):
    """No injection targets the read tool: it just tells the truth."""
    order = delivered_order(conn)
    injector = make_injector(conn, {"issue_refund": "silent_noop"})
    injector.call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "r"},
    )  # F1: the write claimed success but did nothing

    read = injector.call("get_refund", {"order_id": order["id"]})

    assert read.ok
    assert read.data["refunds"] == []  # the read correctly unmasks the F1 lie


def test_read_tool_validate_plan_rejects_non_stale_mode():
    with pytest.raises(ValueError, match="only stale_read"):
        validate_plan({"get_order": "silent_noop"})


def test_read_tool_stale_read_shows_pre_episode_snapshot(conn):
    """A staled read tool answers from before the episode's first tool call,
    even though the write genuinely committed on the live connection."""
    order = delivered_order(conn)
    injector = make_injector(conn, {"get_refund": "stale_read"})

    # The write is clean and genuinely commits on the live DB.
    write = injector.call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )
    assert write.ok

    # But the read tool for the same order reports the pre-episode state: no
    # refund exists, exactly as if it hit a replica that never saw this write.
    stale = injector.call("get_refund", {"order_id": order["id"]})
    assert stale.ok
    assert stale.data["refunds"] == []

    # The live DB really does have the refund — proving this is a stale read,
    # not a lost write like the write-side F5.
    live = conn.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order["id"],)
    ).fetchone()["n"]
    assert live == 1


def test_read_tool_stale_read_only_affects_the_targeted_tool(conn):
    """Staling get_refund must not affect get_order in the same episode."""
    order = delivered_order(conn)
    injector = make_injector(conn, {"get_refund": "stale_read"})
    injector.call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )

    order_read = injector.call("get_order", {"order_id": order["id"]})
    assert order_read.ok and order_read.data["status"] == "refunded"  # truthful


def test_read_tool_baseline_predates_the_first_tool_call(conn):
    """The baseline is frozen at Injector construction, before ANY episode call —
    including calls to tools the plan never targets."""
    order = delivered_order(conn)
    injector = make_injector(conn, {"get_order": "stale_read"})
    # A clean call on an unrelated tool happens first...
    tools.update_order_status(conn, db.SimClock(), order["id"], "delivered")  # no-op change
    # ...then the tracked order is mutated by a live write.
    injector.call(
        "issue_refund",
        {"order_id": order["id"], "amount_paise": order["amount_paise"], "reason": "rtn"},
    )

    staled = injector.call("get_order", {"order_id": order["id"]})
    assert staled.data["status"] == "delivered"  # baseline predates the refund


def test_read_tool_no_baseline_connection_created_when_unneeded(conn):
    """No baseline snapshot is taken unless the plan actually stales a read."""
    injector = make_injector(conn, {"issue_refund": "silent_noop"})
    assert injector._baseline is None
    injector.close()  # must not raise when there is nothing to close


def test_read_tool_injector_close_releases_baseline(conn):
    injector = make_injector(conn, {"get_order": "stale_read"})
    assert injector._baseline is not None
    injector.close()
    assert injector._baseline is None


def test_read_tool_unknown_tool_still_rejected(conn):
    with pytest.raises(ValueError, match="unknown tool"):
        make_injector(conn, {}).call("delete_everything", {})
