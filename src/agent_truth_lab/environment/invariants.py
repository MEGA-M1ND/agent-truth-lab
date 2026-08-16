"""Business invariants over the environment state.

Shared by the test suite and the Arm C independent verifier. Every checker is
a pure read over a connection and returns structured Violation records —
never raises on bad state, never mutates.

The five §4.3 invariants plus two derived static checks (valid status values,
refunded orders backed by a refund row). Full transition-history auditing is
impossible from a state snapshot, so transition legality is enforced at the
tool layer and these checks cover what IS statically checkable.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agent_truth_lab.environment.tools import (
    ORDER_STATUSES,
    SETTLEMENT_FEE_PERCENT,
)


@dataclass(frozen=True)
class Violation:
    """One invariant breach, with enough evidence to explain it."""

    rule: str
    table: str
    entity_id: int | str
    detail: str


def refund_amounts_within_order(conn: sqlite3.Connection) -> list[Violation]:
    """Sum of non-failed refunds per order must not exceed the order amount."""
    rows = conn.execute(
        "SELECT r.order_id, o.amount_paise AS order_amount,"
        " SUM(r.amount_paise) AS refunded"
        " FROM refunds r JOIN orders o ON o.id = r.order_id"
        " WHERE r.status != 'failed' GROUP BY r.order_id"
    ).fetchall()
    return [
        Violation(
            rule="refund_le_order",
            table="refunds",
            entity_id=row["order_id"],
            detail=f"refunds total {row['refunded']} > order amount {row['order_amount']}",
        )
        for row in rows
        if row["refunded"] > row["order_amount"]
    ]


def single_refund_per_order(conn: sqlite3.Connection) -> list[Violation]:
    """At most one non-failed refund per order."""
    rows = conn.execute(
        "SELECT order_id, COUNT(*) AS n FROM refunds"
        " WHERE status != 'failed' GROUP BY order_id HAVING n > 1"
    ).fetchall()
    return [
        Violation(
            rule="single_refund_per_order",
            table="refunds",
            entity_id=row["order_id"],
            detail=f"{row['n']} non-failed refunds for one order",
        )
        for row in rows
    ]


def valid_order_statuses(conn: sqlite3.Connection) -> list[Violation]:
    """Every order status must be a member of the legal status set."""
    rows = conn.execute("SELECT id, status FROM orders").fetchall()
    return [
        Violation(
            rule="valid_order_status",
            table="orders",
            entity_id=row["id"],
            detail=f"unknown status '{row['status']}'",
        )
        for row in rows
        if row["status"] not in ORDER_STATUSES
    ]


def refunded_orders_have_completed_refund(conn: sqlite3.Connection) -> list[Violation]:
    """An order in status 'refunded' must be backed by a completed refund row."""
    rows = conn.execute(
        "SELECT o.id FROM orders o"
        " LEFT JOIN refunds r ON r.order_id = o.id AND r.status = 'completed'"
        " WHERE o.status = 'refunded' AND r.id IS NULL"
    ).fetchall()
    return [
        Violation(
            rule="refunded_order_has_refund",
            table="orders",
            entity_id=row["id"],
            detail="order is 'refunded' but has no completed refund",
        )
        for row in rows
    ]


def settlement_math(conn: sqlite3.Connection) -> list[Violation]:
    """gross must equal that day's succeeded charges; net must equal
    gross - fees - that day's completed refunds, exactly.

    Caveat: this recomputes from current charges/refunds, so it assumes no
    new activity is backdated onto an already-settled day. The SimClock makes
    that impossible in practice — agent activity is always timestamped after
    the seeded history days.
    """
    violations: list[Violation] = []
    for st in conn.execute("SELECT * FROM settlements").fetchall():
        day = st["merchant_day"]
        gross = conn.execute(
            "SELECT COALESCE(SUM(amount_paise), 0) AS total FROM charges"
            " WHERE status = 'succeeded' AND substr(created_at, 1, 10) = ?",
            (day,),
        ).fetchone()["total"]
        refunds_total = conn.execute(
            "SELECT COALESCE(SUM(amount_paise), 0) AS total FROM refunds"
            " WHERE status = 'completed' AND substr(created_at, 1, 10) = ?",
            (day,),
        ).fetchone()["total"]
        expected_fees = gross * SETTLEMENT_FEE_PERCENT // 100
        expected_net = gross - expected_fees - refunds_total
        if st["gross_paise"] != gross:
            violations.append(
                Violation(
                    rule="settlement_math",
                    table="settlements",
                    entity_id=st["id"],
                    detail=f"gross {st['gross_paise']} != charges total {gross} for {day}",
                )
            )
        elif st["net_paise"] != st["gross_paise"] - st["fees_paise"] - refunds_total:
            violations.append(
                Violation(
                    rule="settlement_math",
                    table="settlements",
                    entity_id=st["id"],
                    detail=(
                        f"net {st['net_paise']} != gross {st['gross_paise']}"
                        f" - fees {st['fees_paise']} - refunds {refunds_total}"
                        f" (expected {expected_net} at standard fees)"
                    ),
                )
            )
    return violations


def refund_emails_reference_completed_refund(conn: sqlite3.Connection) -> list[Violation]:
    """A refund_completed email must reference an existing completed refund.

    related_entity convention: 'refund:<id>'.
    """
    violations: list[Violation] = []
    rows = conn.execute(
        "SELECT id, related_entity FROM emails_sent WHERE template = 'refund_completed'"
    ).fetchall()
    for row in rows:
        ref = row["related_entity"]
        prefix, _, raw_id = ref.partition(":")
        if prefix != "refund" or not raw_id.isdigit():
            violations.append(
                Violation(
                    rule="refund_email_references_refund",
                    table="emails_sent",
                    entity_id=row["id"],
                    detail=f"related_entity '{ref}' is not a 'refund:<id>' reference",
                )
            )
            continue
        refund = conn.execute(
            "SELECT id FROM refunds WHERE id = ? AND status = 'completed'", (int(raw_id),)
        ).fetchone()
        if refund is None:
            violations.append(
                Violation(
                    rule="refund_email_references_refund",
                    table="emails_sent",
                    entity_id=row["id"],
                    detail=f"no completed refund with id {raw_id}",
                )
            )
    return violations


_ALL_CHECKS = (
    refund_amounts_within_order,
    single_refund_per_order,
    valid_order_statuses,
    refunded_orders_have_completed_refund,
    settlement_math,
    refund_emails_reference_completed_refund,
)


def check_all(conn: sqlite3.Connection) -> list[Violation]:
    """Run every invariant checker and return all violations found."""
    violations: list[Violation] = []
    for check in _ALL_CHECKS:
        violations.extend(check(conn))
    return violations
