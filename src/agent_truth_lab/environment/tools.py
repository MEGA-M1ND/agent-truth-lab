"""The 5 tools exposed to the agent — clean, genuinely correct implementations.

These are the ONLY mutation paths the agent has. The failure-injection layer
(M2) wraps these functions; nothing here knows injection exists.

Envelope: every tool returns a ToolResult
    {ok: bool, http_status: int, data: {...}, error: str | null}

HTTP status conventions:
    200 success | 400 malformed argument | 402 charge declined
    404 unknown entity | 409 illegal state | 422 bad amount / no activity

Semantic correctness of *workflows* is the agent's job, not the tools'.
Example: send_customer_email validates the customer and template but not
whether a 'refund_completed' email actually corresponds to a completed
refund — a real email service would not know. The invariant checker catches
that mismatch after the fact.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent_truth_lab.environment.db import SimClock

ORDER_STATUSES = frozenset(
    {"placed", "shipped", "delivered", "cancelled", "refund_pending", "refunded"}
)

# Transitions available through update_order_status. 'refunded' is deliberately
# reachable ONLY via issue_refund, so a clean update_order_status call can never
# produce a refunded order without a backing refund row.
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "placed": frozenset({"shipped", "cancelled"}),
    "shipped": frozenset({"delivered", "cancelled"}),
    "delivered": frozenset({"refund_pending", "cancelled"}),
    "refund_pending": frozenset(),
    "cancelled": frozenset(),
    "refunded": frozenset(),
}

# Order states from which issue_refund is permitted.
REFUNDABLE_STATUSES = frozenset({"delivered", "cancelled", "refund_pending"})

EMAIL_TEMPLATES = frozenset(
    {"refund_completed", "payment_receipt", "order_update", "subscription_past_due",
     "settlement_notice"}
)

SETTLEMENT_FEE_PERCENT = 2  # flat 2% of gross, floor division


@dataclass(frozen=True)
class ToolResult:
    """The JSON envelope every tool returns."""

    ok: bool
    http_status: int
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, e.g. for the audit log or the agent-facing JSON."""
        return {"ok": self.ok, "http_status": self.http_status, "data": self.data,
                "error": self.error}

    def to_json(self) -> str:
        """JSON string form as returned to the agent."""
        return json.dumps(self.to_dict(), sort_keys=True)


def _ok(data: dict[str, Any]) -> ToolResult:
    return ToolResult(ok=True, http_status=200, data=data)


def _err(status: int, message: str, data: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(ok=False, http_status=status, data=data or {}, error=message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def issue_refund(
    conn: sqlite3.Connection,
    clock: SimClock,
    order_id: int,
    amount_paise: int,
    reason: str,
) -> ToolResult:
    """Refund an order: insert refund row, mark order refunded, credit customer balance.

    Deliberately takes no idempotency key — retried calls create duplicate
    refunds, which is exactly the exposure the F3 failure mode exploits.
    """
    if not _is_int(order_id):
        return _err(400, "order_id must be an integer")
    if not isinstance(reason, str) or not reason.strip():
        return _err(400, "reason must be a non-empty string")
    if not _is_int(amount_paise) or amount_paise <= 0:
        return _err(422, "amount_paise must be a positive integer")

    order = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if order is None:
        return _err(404, f"order {order_id} not found")
    if order["status"] not in REFUNDABLE_STATUSES:
        return _err(409, f"order {order_id} has status '{order['status']}', not refundable")
    existing = conn.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ? AND status != 'failed'",
        (order_id,),
    ).fetchone()["n"]
    if existing:
        return _err(409, f"order {order_id} already has a non-failed refund")
    if amount_paise > order["amount_paise"]:
        return _err(
            422,
            f"refund amount {amount_paise} exceeds order amount {order['amount_paise']}",
        )

    created_at = clock.now()
    cur = conn.execute(
        "INSERT INTO refunds (order_id, customer_id, amount_paise, status, created_at,"
        " idempotency_key) VALUES (?, ?, ?, 'completed', ?, NULL)",
        (order_id, order["customer_id"], amount_paise, created_at),
    )
    refund_id = cur.lastrowid
    conn.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order_id,))
    conn.execute(
        "UPDATE customers SET balance_paise = balance_paise + ? WHERE id = ?",
        (amount_paise, order["customer_id"]),
    )
    conn.commit()
    return _ok(
        {
            "refund_id": refund_id,
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "amount_paise": amount_paise,
            "status": "completed",
            "created_at": created_at,
        }
    )


def retry_subscription_charge(
    conn: sqlite3.Connection, clock: SimClock, subscription_id: int
) -> ToolResult:
    """Retry the charge on a past_due subscription.

    Outcome is deterministic: the charge succeeds iff the customer's balance
    covers the subscription amount. On success the balance is debited, the
    subscription becomes active, and next_charge_at advances 30 days. On
    decline a failed charge row is still recorded and the envelope is
    ok=false / 402.
    """
    if not _is_int(subscription_id):
        return _err(400, "subscription_id must be an integer")

    sub = conn.execute(
        "SELECT * FROM subscriptions WHERE id = ?", (subscription_id,)
    ).fetchone()
    if sub is None:
        return _err(404, f"subscription {subscription_id} not found")
    if sub["status"] != "past_due":
        return _err(
            409,
            f"subscription {subscription_id} has status '{sub['status']}';"
            " only past_due subscriptions can be retried",
        )

    customer = conn.execute(
        "SELECT * FROM customers WHERE id = ?", (sub["customer_id"],)
    ).fetchone()
    attempt_no = (
        conn.execute(
            "SELECT COALESCE(MAX(attempt_no), 0) AS n FROM charges WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()["n"]
        + 1
    )
    created_at = clock.now()

    if customer["balance_paise"] >= sub["amount_paise"]:
        cur = conn.execute(
            "INSERT INTO charges (subscription_id, amount_paise, status, attempt_no,"
            " created_at) VALUES (?, ?, 'succeeded', ?, ?)",
            (subscription_id, sub["amount_paise"], attempt_no, created_at),
        )
        next_charge_at = clock.in_days(30)
        conn.execute(
            "UPDATE customers SET balance_paise = balance_paise - ? WHERE id = ?",
            (sub["amount_paise"], sub["customer_id"]),
        )
        conn.execute(
            "UPDATE subscriptions SET status = 'active', next_charge_at = ? WHERE id = ?",
            (next_charge_at, subscription_id),
        )
        conn.commit()
        return _ok(
            {
                "charge_id": cur.lastrowid,
                "subscription_id": subscription_id,
                "amount_paise": sub["amount_paise"],
                "status": "succeeded",
                "attempt_no": attempt_no,
                "next_charge_at": next_charge_at,
            }
        )

    cur = conn.execute(
        "INSERT INTO charges (subscription_id, amount_paise, status, attempt_no,"
        " created_at) VALUES (?, ?, 'failed', ?, ?)",
        (subscription_id, sub["amount_paise"], attempt_no, created_at),
    )
    conn.commit()
    return _err(
        402,
        "charge_declined_insufficient_funds",
        data={
            "charge_id": cur.lastrowid,
            "subscription_id": subscription_id,
            "status": "failed",
            "attempt_no": attempt_no,
        },
    )


def create_settlement(
    conn: sqlite3.Connection,
    clock: SimClock,
    merchant_day: str,
    mark_processed: bool = False,
) -> ToolResult:
    """Aggregate a merchant day into a settlement row.

    gross = succeeded charges that day; fees = 2% of gross (floor);
    net = gross - fees - completed refunds that day. One settlement per day.
    """
    if not isinstance(merchant_day, str):
        return _err(400, "merchant_day must be a YYYY-MM-DD string")
    try:
        datetime.strptime(merchant_day, "%Y-%m-%d")
    except ValueError:
        return _err(400, f"merchant_day '{merchant_day}' is not a valid YYYY-MM-DD date")
    if not isinstance(mark_processed, bool):
        return _err(400, "mark_processed must be a boolean")

    existing = conn.execute(
        "SELECT COUNT(*) AS n FROM settlements WHERE merchant_day = ?", (merchant_day,)
    ).fetchone()["n"]
    if existing:
        return _err(409, f"settlement for {merchant_day} already exists")

    gross = conn.execute(
        "SELECT COALESCE(SUM(amount_paise), 0) AS total FROM charges"
        " WHERE status = 'succeeded' AND substr(created_at, 1, 10) = ?",
        (merchant_day,),
    ).fetchone()["total"]
    refunds_total = conn.execute(
        "SELECT COALESCE(SUM(amount_paise), 0) AS total FROM refunds"
        " WHERE status = 'completed' AND substr(created_at, 1, 10) = ?",
        (merchant_day,),
    ).fetchone()["total"]
    if gross == 0 and refunds_total == 0:
        return _err(422, f"no charge or refund activity on {merchant_day}")

    fees = gross * SETTLEMENT_FEE_PERCENT // 100
    net = gross - fees - refunds_total
    status = "processed" if mark_processed else "pending"
    cur = conn.execute(
        "INSERT INTO settlements (merchant_day, gross_paise, fees_paise, net_paise,"
        " status) VALUES (?, ?, ?, ?, ?)",
        (merchant_day, gross, fees, net, status),
    )
    conn.commit()
    return _ok(
        {
            "settlement_id": cur.lastrowid,
            "merchant_day": merchant_day,
            "gross_paise": gross,
            "fees_paise": fees,
            "net_paise": net,
            "status": status,
        }
    )


def update_order_status(
    conn: sqlite3.Connection, clock: SimClock, order_id: int, new_status: str
) -> ToolResult:
    """Move an order along the legal-transition table."""
    if not _is_int(order_id):
        return _err(400, "order_id must be an integer")
    if new_status not in ORDER_STATUSES:
        return _err(400, f"'{new_status}' is not a valid order status")

    order = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if order is None:
        return _err(404, f"order {order_id} not found")
    if new_status == "refunded":
        return _err(409, "orders reach 'refunded' only via issue_refund")
    if new_status not in LEGAL_TRANSITIONS[order["status"]]:
        return _err(
            409,
            f"illegal transition '{order['status']}' -> '{new_status}' for order {order_id}",
        )

    conn.execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order_id))
    conn.commit()
    return _ok({"order_id": order_id, "old_status": order["status"], "new_status": new_status})


def send_customer_email(
    conn: sqlite3.Connection,
    clock: SimClock,
    customer_id: int,
    template: str,
    related_entity: str,
) -> ToolResult:
    """Record an outbound email.

    related_entity is a '<type>:<id>' reference, e.g. 'refund:7' or
    'order:1017'. The email service does not validate the reference
    semantically — that is the agent's responsibility, and the invariant
    checker's to catch.
    """
    if not _is_int(customer_id):
        return _err(400, "customer_id must be an integer")
    if template not in EMAIL_TEMPLATES:
        return _err(400, f"'{template}' is not a valid email template")
    if not isinstance(related_entity, str) or not related_entity.strip():
        return _err(400, "related_entity must be a non-empty string like 'order:1017'")

    customer = conn.execute(
        "SELECT id FROM customers WHERE id = ?", (customer_id,)
    ).fetchone()
    if customer is None:
        return _err(404, f"customer {customer_id} not found")

    created_at = clock.now()
    cur = conn.execute(
        "INSERT INTO emails_sent (customer_id, template, related_entity, created_at)"
        " VALUES (?, ?, ?, ?)",
        (customer_id, template, related_entity, created_at),
    )
    conn.commit()
    return _ok(
        {
            "email_id": cur.lastrowid,
            "customer_id": customer_id,
            "template": template,
            "related_entity": related_entity,
            "created_at": created_at,
        }
    )
