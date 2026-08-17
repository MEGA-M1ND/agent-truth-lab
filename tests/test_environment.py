"""M1 tests: schema, deterministic seeding, the 5 clean tools, and invariants.

The clean tools must be genuinely correct — injected failures (M2) must be
the only source of wrongness in the experiment. Every happy-path test
asserts both the response envelope AND the resulting DB state.
"""

from __future__ import annotations

import sqlite3

import pytest

from agent_truth_lab.environment import db, invariants, tools

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fresh_env(seed: int = 42) -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, seed)
    return conn


def get_order(conn, order_id):
    return conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()


def get_customer(conn, customer_id):
    return conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()


def first_order_with_status(conn, status):
    row = conn.execute(
        "SELECT * FROM orders WHERE status = ? ORDER BY id LIMIT 1", (status,)
    ).fetchone()
    assert row is not None, f"seed produced no order with status '{status}'"
    return row


def past_due_sub(conn, funded: bool):
    """A past_due subscription whose customer can (or cannot) cover the amount."""
    op = ">=" if funded else "<"
    row = conn.execute(
        f"SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        f" WHERE s.status = 'past_due' AND c.balance_paise {op} s.amount_paise"
        f" ORDER BY s.id LIMIT 1"
    ).fetchone()
    assert row is not None, f"seed produced no past_due subscription with funded={funded}"
    return row


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_all_tables_exist(conn):
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    expected = {
        "customers", "orders", "refunds", "subscriptions", "charges",
        "settlements", "emails_sent", "audit_log",
    }
    assert expected <= names


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO orders (id, customer_id, amount_paise, status)"
            " VALUES (9999, 999, 100, 'placed')"
        )


# ---------------------------------------------------------------------------
# seed data
# ---------------------------------------------------------------------------


def test_seed_is_deterministic():
    a, b = fresh_env(42), fresh_env(42)
    assert db.dump(a) == db.dump(b)


def test_different_seeds_differ():
    assert db.dump(fresh_env(42)) != db.dump(fresh_env(43))


def test_seed_row_counts(conn):
    counts = {
        table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]  # noqa: S608
        for table in ("customers", "orders", "subscriptions")
    }
    assert counts == {"customers": 50, "orders": 200, "subscriptions": 40}


def test_seed_status_mix(conn):
    order_statuses = {
        row["status"]: row["n"]
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM orders GROUP BY status")
    }
    assert set(order_statuses) == set(tools.ORDER_STATUSES)
    assert order_statuses["delivered"] >= 20

    sub_statuses = {
        row["status"]: row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM subscriptions GROUP BY status"
        )
    }
    assert sub_statuses == {"active": 28, "past_due": 8, "cancelled": 4}


def test_seed_past_due_retry_mix(conn):
    """Both retry outcomes must be reachable: some funded, some underfunded."""
    assert past_due_sub(conn, funded=True) is not None
    assert past_due_sub(conn, funded=False) is not None


def test_seed_has_charge_history_on_both_days(conn):
    for day in db.HISTORY_DAYS:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM charges WHERE substr(created_at, 1, 10) = ?"
            " AND status = 'succeeded'",
            (day,),
        ).fetchone()["n"]
        assert n > 0, f"no succeeded charges seeded on {day}"


def test_seed_satisfies_all_invariants(conn):
    assert invariants.check_all(conn) == []


# ---------------------------------------------------------------------------
# issue_refund
# ---------------------------------------------------------------------------


def test_issue_refund_happy_path(conn, clock):
    order = first_order_with_status(conn, "delivered")
    balance_before = get_customer(conn, order["customer_id"])["balance_paise"]

    result = tools.issue_refund(
        conn, clock, order["id"], order["amount_paise"], "customer request"
    )

    assert result.ok and result.http_status == 200
    assert result.data["amount_paise"] == order["amount_paise"]
    assert result.data["status"] == "completed"

    refund = conn.execute(
        "SELECT * FROM refunds WHERE id = ?", (result.data["refund_id"],)
    ).fetchone()
    assert refund["order_id"] == order["id"]
    assert refund["amount_paise"] == order["amount_paise"]
    assert refund["status"] == "completed"
    assert get_order(conn, order["id"])["status"] == "refunded"
    assert (
        get_customer(conn, order["customer_id"])["balance_paise"]
        == balance_before + order["amount_paise"]
    )


def test_issue_refund_partial_amount(conn, clock):
    order = first_order_with_status(conn, "delivered")
    partial = order["amount_paise"] // 2
    result = tools.issue_refund(conn, clock, order["id"], partial, "partial per policy")
    assert result.ok
    assert result.data["amount_paise"] == partial
    # Partial refund still closes the order as refunded (one refund per order).
    assert get_order(conn, order["id"])["status"] == "refunded"


def test_issue_refund_unknown_order(conn, clock):
    result = tools.issue_refund(conn, clock, 424242, 100, "x")
    assert not result.ok and result.http_status == 404


def test_issue_refund_exceeds_order_amount(conn, clock):
    order = first_order_with_status(conn, "delivered")
    result = tools.issue_refund(
        conn, clock, order["id"], order["amount_paise"] + 1, "too much"
    )
    assert not result.ok and result.http_status == 422
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order["id"],)
    ).fetchone()["n"] == 0


def test_issue_refund_non_refundable_status(conn, clock):
    order = first_order_with_status(conn, "placed")
    result = tools.issue_refund(conn, clock, order["id"], 100, "x")
    assert not result.ok and result.http_status == 409


def test_issue_refund_duplicate_rejected(conn, clock):
    order = first_order_with_status(conn, "delivered")
    assert tools.issue_refund(conn, clock, order["id"], 100, "first").ok
    second = tools.issue_refund(conn, clock, order["id"], 100, "second")
    assert not second.ok and second.http_status == 409


def test_issue_refund_bad_amounts(conn, clock):
    order = first_order_with_status(conn, "delivered")
    for bad in (0, -5, "100", 1.5, True):
        result = tools.issue_refund(conn, clock, order["id"], bad, "x")
        assert not result.ok and result.http_status == 422


# ---------------------------------------------------------------------------
# retry_subscription_charge
# ---------------------------------------------------------------------------


def test_retry_charge_success(conn, clock):
    sub = past_due_sub(conn, funded=True)
    balance_before = get_customer(conn, sub["customer_id"])["balance_paise"]

    result = tools.retry_subscription_charge(conn, clock, sub["id"])

    assert result.ok and result.data["status"] == "succeeded"
    charge = conn.execute(
        "SELECT * FROM charges WHERE id = ?", (result.data["charge_id"],)
    ).fetchone()
    assert charge["status"] == "succeeded"
    assert charge["amount_paise"] == sub["amount_paise"]
    after = conn.execute(
        "SELECT * FROM subscriptions WHERE id = ?", (sub["id"],)
    ).fetchone()
    assert after["status"] == "active"
    assert after["next_charge_at"] > sub["next_charge_at"]
    assert (
        get_customer(conn, sub["customer_id"])["balance_paise"]
        == balance_before - sub["amount_paise"]
    )


def test_retry_charge_declined(conn, clock):
    sub = past_due_sub(conn, funded=False)
    balance_before = get_customer(conn, sub["customer_id"])["balance_paise"]

    result = tools.retry_subscription_charge(conn, clock, sub["id"])

    assert not result.ok and result.http_status == 402
    charge = conn.execute(
        "SELECT * FROM charges WHERE id = ?", (result.data["charge_id"],)
    ).fetchone()
    assert charge["status"] == "failed"
    after = conn.execute(
        "SELECT * FROM subscriptions WHERE id = ?", (sub["id"],)
    ).fetchone()
    assert after["status"] == "past_due"
    assert get_customer(conn, sub["customer_id"])["balance_paise"] == balance_before


def test_retry_charge_increments_attempt_no(conn, clock):
    sub = past_due_sub(conn, funded=False)
    first = tools.retry_subscription_charge(conn, clock, sub["id"])
    second = tools.retry_subscription_charge(conn, clock, sub["id"])
    assert second.data["attempt_no"] == first.data["attempt_no"] + 1


def test_retry_charge_unknown_subscription(conn, clock):
    result = tools.retry_subscription_charge(conn, clock, 999999)
    assert not result.ok and result.http_status == 404


def test_retry_charge_not_past_due(conn, clock):
    sub = conn.execute(
        "SELECT * FROM subscriptions WHERE status = 'active' LIMIT 1"
    ).fetchone()
    result = tools.retry_subscription_charge(conn, clock, sub["id"])
    assert not result.ok and result.http_status == 409


# ---------------------------------------------------------------------------
# create_settlement
# ---------------------------------------------------------------------------


def test_create_settlement_math(conn, clock):
    day = db.HISTORY_DAYS[0]
    gross = conn.execute(
        "SELECT COALESCE(SUM(amount_paise), 0) AS t FROM charges"
        " WHERE status = 'succeeded' AND substr(created_at, 1, 10) = ?",
        (day,),
    ).fetchone()["t"]
    refunds_total = conn.execute(
        "SELECT COALESCE(SUM(amount_paise), 0) AS t FROM refunds"
        " WHERE status = 'completed' AND substr(created_at, 1, 10) = ?",
        (day,),
    ).fetchone()["t"]
    assert gross > 0 and refunds_total > 0, "seed must give both components for this test"

    result = tools.create_settlement(conn, clock, day)

    assert result.ok
    expected_fees = gross * 2 // 100
    assert result.data["gross_paise"] == gross
    assert result.data["fees_paise"] == expected_fees
    assert result.data["net_paise"] == gross - expected_fees - refunds_total
    assert result.data["status"] == "pending"
    row = conn.execute(
        "SELECT * FROM settlements WHERE id = ?", (result.data["settlement_id"],)
    ).fetchone()
    assert row["net_paise"] == result.data["net_paise"]
    assert invariants.settlement_math(conn) == []


def test_create_settlement_mark_processed(conn, clock):
    result = tools.create_settlement(conn, clock, db.HISTORY_DAYS[1], mark_processed=True)
    assert result.ok and result.data["status"] == "processed"


def test_create_settlement_duplicate_day(conn, clock):
    assert tools.create_settlement(conn, clock, db.HISTORY_DAYS[0]).ok
    second = tools.create_settlement(conn, clock, db.HISTORY_DAYS[0])
    assert not second.ok and second.http_status == 409


def test_create_settlement_no_activity(conn, clock):
    result = tools.create_settlement(conn, clock, "2020-01-01")
    assert not result.ok and result.http_status == 422


def test_create_settlement_bad_date(conn, clock):
    result = tools.create_settlement(conn, clock, "30-12-2025")
    assert not result.ok and result.http_status == 400


# ---------------------------------------------------------------------------
# update_order_status
# ---------------------------------------------------------------------------


def test_update_order_status_legal(conn, clock):
    order = first_order_with_status(conn, "placed")
    result = tools.update_order_status(conn, clock, order["id"], "shipped")
    assert result.ok
    assert result.data == {
        "order_id": order["id"], "old_status": "placed", "new_status": "shipped",
    }
    assert get_order(conn, order["id"])["status"] == "shipped"


def test_update_order_status_illegal_transition(conn, clock):
    order = first_order_with_status(conn, "placed")
    result = tools.update_order_status(conn, clock, order["id"], "delivered")
    assert not result.ok and result.http_status == 409
    assert get_order(conn, order["id"])["status"] == "placed"


def test_update_order_status_refunded_blocked(conn, clock):
    """'refunded' is reachable only via issue_refund, never via direct update."""
    order = first_order_with_status(conn, "delivered")
    result = tools.update_order_status(conn, clock, order["id"], "refunded")
    assert not result.ok and result.http_status == 409


def test_update_order_status_invalid_status(conn, clock):
    order = first_order_with_status(conn, "placed")
    result = tools.update_order_status(conn, clock, order["id"], "teleported")
    assert not result.ok and result.http_status == 400


def test_update_order_status_unknown_order(conn, clock):
    result = tools.update_order_status(conn, clock, 424242, "shipped")
    assert not result.ok and result.http_status == 404


# ---------------------------------------------------------------------------
# send_customer_email
# ---------------------------------------------------------------------------


def test_send_email_happy_path(conn, clock):
    result = tools.send_customer_email(conn, clock, 1, "order_update", "order:1001")
    assert result.ok
    row = conn.execute(
        "SELECT * FROM emails_sent WHERE id = ?", (result.data["email_id"],)
    ).fetchone()
    assert row["customer_id"] == 1
    assert row["template"] == "order_update"
    assert row["related_entity"] == "order:1001"


def test_send_email_unknown_customer(conn, clock):
    result = tools.send_customer_email(conn, clock, 999, "order_update", "order:1001")
    assert not result.ok and result.http_status == 404


def test_send_email_invalid_template(conn, clock):
    result = tools.send_customer_email(conn, clock, 1, "free_money", "order:1001")
    assert not result.ok and result.http_status == 400


# ---------------------------------------------------------------------------
# invariant checkers detect hand-crafted violations
# ---------------------------------------------------------------------------


def test_invariant_refund_exceeds_order(conn):
    order = first_order_with_status(conn, "delivered")
    conn.execute(
        "INSERT INTO refunds (order_id, customer_id, amount_paise, status, created_at)"
        " VALUES (?, ?, ?, 'completed', '2026-01-10T00:00:00')",
        (order["id"], order["customer_id"], order["amount_paise"] * 2),
    )
    found = invariants.refund_amounts_within_order(conn)
    assert [v.entity_id for v in found] == [order["id"]]
    assert found[0].rule == "refund_le_order"


def test_invariant_duplicate_refunds(conn):
    order = first_order_with_status(conn, "delivered")
    for _ in range(2):
        conn.execute(
            "INSERT INTO refunds (order_id, customer_id, amount_paise, status, created_at)"
            " VALUES (?, ?, 100, 'completed', '2026-01-10T00:00:00')",
            (order["id"], order["customer_id"]),
        )
    found = invariants.single_refund_per_order(conn)
    assert [v.entity_id for v in found] == [order["id"]]


def test_invariant_invalid_order_status(conn):
    conn.execute("UPDATE orders SET status = 'quantum' WHERE id = 1001")
    found = invariants.valid_order_statuses(conn)
    assert [v.entity_id for v in found] == [1001]


def test_invariant_refunded_without_refund_row(conn):
    order = first_order_with_status(conn, "delivered")
    conn.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order["id"],))
    found = invariants.refunded_orders_have_completed_refund(conn)
    assert [v.entity_id for v in found] == [order["id"]]


def test_invariant_settlement_math_tampered(conn, clock):
    result = tools.create_settlement(conn, clock, db.HISTORY_DAYS[0])
    assert result.ok
    conn.execute(
        "UPDATE settlements SET net_paise = net_paise - 12345 WHERE id = ?",
        (result.data["settlement_id"],),
    )
    found = invariants.settlement_math(conn)
    assert [v.entity_id for v in found] == [result.data["settlement_id"]]


def test_invariant_refund_email_without_refund(conn, clock):
    result = tools.send_customer_email(conn, clock, 1, "refund_completed", "refund:99999")
    assert result.ok  # the email service is deliberately dumb
    found = invariants.refund_emails_reference_completed_refund(conn)
    assert len(found) == 1
    assert found[0].rule == "refund_email_references_refund"


def test_invariant_refund_email_malformed_reference(conn, clock):
    tools.send_customer_email(conn, clock, 1, "refund_completed", "order:1001")
    found = invariants.refund_emails_reference_completed_refund(conn)
    assert len(found) == 1


# ---------------------------------------------------------------------------
# clean tools preserve invariants end-to-end
# ---------------------------------------------------------------------------


def test_clean_tool_sequence_preserves_invariants(conn, clock):
    """A realistic multi-tool workflow through clean tools leaves zero violations."""
    order = first_order_with_status(conn, "delivered")
    refund = tools.issue_refund(conn, clock, order["id"], order["amount_paise"], "return")
    assert refund.ok
    assert tools.send_customer_email(
        conn, clock, order["customer_id"], "refund_completed",
        f"refund:{refund.data['refund_id']}",
    ).ok
    assert tools.retry_subscription_charge(conn, clock, past_due_sub(conn, True)["id"]).ok
    assert tools.create_settlement(conn, clock, db.HISTORY_DAYS[1]).ok
    placed = first_order_with_status(conn, "placed")
    assert tools.update_order_status(conn, clock, placed["id"], "shipped").ok

    assert invariants.check_all(conn) == []


# ---------------------------------------------------------------------------
# audit log + envelope
# ---------------------------------------------------------------------------


def test_record_audit(conn, clock):
    db.record_audit(conn, clock, "issue_refund", {"order_id": 1}, {"ok": True})
    row = conn.execute("SELECT * FROM audit_log").fetchone()
    assert row["tool_name"] == "issue_refund"
    assert '"order_id": 1' in row["args_json"]


def test_tool_result_envelope_shape(conn, clock):
    import json

    result = tools.send_customer_email(conn, clock, 1, "order_update", "order:1001")
    parsed = json.loads(result.to_json())
    assert set(parsed) == {"ok", "http_status", "data", "error"}
    assert parsed["ok"] is True and parsed["error"] is None


# ---------------------------------------------------------------------------
# read tools (M7) — never mutate, genuinely correct
# ---------------------------------------------------------------------------


def test_get_order_happy_path(conn, clock):
    order = first_order_with_status(conn, "delivered")
    result = tools.get_order(conn, clock, order["id"])
    assert result.ok and result.data["id"] == order["id"]
    assert result.data["status"] == "delivered"
    assert result.data["amount_paise"] == order["amount_paise"]


def test_get_order_unknown(conn, clock):
    result = tools.get_order(conn, clock, 424242)
    assert not result.ok and result.http_status == 404


def test_get_order_bad_type(conn, clock):
    result = tools.get_order(conn, clock, "not-an-int")
    assert not result.ok and result.http_status == 400


def test_get_refund_empty_is_ok_not_error(conn, clock):
    """No refund yet is a normal state, not a tool failure."""
    order = first_order_with_status(conn, "delivered")
    result = tools.get_refund(conn, clock, order["id"])
    assert result.ok and result.http_status == 200
    assert result.data == {"order_id": order["id"], "refunds": []}


def test_get_refund_reflects_a_real_refund(conn, clock):
    order = first_order_with_status(conn, "delivered")
    issued = tools.issue_refund(conn, clock, order["id"], order["amount_paise"], "rtn")
    result = tools.get_refund(conn, clock, order["id"])
    assert result.ok
    assert len(result.data["refunds"]) == 1
    assert result.data["refunds"][0]["id"] == issued.data["refund_id"]
    assert result.data["refunds"][0]["amount_paise"] == order["amount_paise"]


def test_get_refund_unknown_order(conn, clock):
    result = tools.get_refund(conn, clock, 424242)
    assert not result.ok and result.http_status == 404


def test_get_subscription_happy_path(conn, clock):
    sub = conn.execute(
        "SELECT * FROM subscriptions WHERE status = 'active' LIMIT 1"
    ).fetchone()
    result = tools.get_subscription(conn, clock, sub["id"])
    assert result.ok and result.data["id"] == sub["id"]
    assert result.data["status"] == "active"
    assert result.data["latest_charge"] is None or isinstance(
        result.data["latest_charge"], dict
    )


def test_get_subscription_reflects_latest_charge(conn, clock):
    sub = conn.execute(
        "SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        " WHERE s.status = 'past_due' AND c.balance_paise >= s.amount_paise LIMIT 1"
    ).fetchone()
    charged = tools.retry_subscription_charge(conn, clock, sub["id"])
    assert charged.ok
    result = tools.get_subscription(conn, clock, sub["id"])
    assert result.data["status"] == "active"
    assert result.data["latest_charge"]["status"] == "succeeded"
    assert result.data["latest_charge"]["id"] == charged.data["charge_id"]


def test_get_subscription_unknown(conn, clock):
    result = tools.get_subscription(conn, clock, 999999)
    assert not result.ok and result.http_status == 404


def test_get_settlement_not_created_yet(conn, clock):
    result = tools.get_settlement(conn, clock, "2025-12-30")
    assert not result.ok and result.http_status == 404


def test_get_settlement_reflects_a_real_settlement(conn, clock):
    created = tools.create_settlement(conn, clock, db.HISTORY_DAYS[0])
    result = tools.get_settlement(conn, clock, db.HISTORY_DAYS[0])
    assert result.ok
    assert result.data["id"] == created.data["settlement_id"]
    assert result.data["net_paise"] == created.data["net_paise"]


def test_read_tools_never_mutate(conn, clock):
    before = db.dump(conn)
    order = first_order_with_status(conn, "delivered")
    tools.get_order(conn, clock, order["id"])
    tools.get_refund(conn, clock, order["id"])
    tools.get_subscription(conn, clock, 501)
    tools.get_settlement(conn, clock, "2025-12-30")
    assert db.dump(conn) == before
