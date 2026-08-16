"""SQLite environment: schema, connection management, deterministic seed data.

Design notes
------------
- Business invariants are deliberately NOT enforced with SQL CHECK constraints.
  The failure-injection layer (M2) must be able to write invariant-violating
  rows, and the independent verifier (M4) must be the component that catches
  them. Only referential integrity (foreign keys) is enforced at the DB level.
- All timestamps come from a SimClock, never wall-clock, so runs are
  byte-for-byte reproducible from a seed. Timestamps are naive ISO-8601
  strings interpreted as UTC.
- All money is integer paise.
- Seeded history lives on merchant days 2025-12-30 and 2025-12-31 so that
  settlement missions have real charge/refund data to aggregate. The
  simulation "present" starts 2026-01-10.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

SCHEMA = """
CREATE TABLE customers (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    balance_paise INTEGER NOT NULL
);

CREATE TABLE orders (
    id            INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    amount_paise  INTEGER NOT NULL,
    status        TEXT NOT NULL
);

CREATE TABLE refunds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        INTEGER NOT NULL REFERENCES orders(id),
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    amount_paise    INTEGER NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    idempotency_key TEXT
);

CREATE TABLE subscriptions (
    id             INTEGER PRIMARY KEY,
    customer_id    INTEGER NOT NULL REFERENCES customers(id),
    plan           TEXT NOT NULL,
    amount_paise   INTEGER NOT NULL,
    status         TEXT NOT NULL,
    next_charge_at TEXT NOT NULL
);

CREATE TABLE charges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL REFERENCES subscriptions(id),
    amount_paise    INTEGER NOT NULL,
    status          TEXT NOT NULL,
    attempt_no      INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE settlements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant_day TEXT NOT NULL,
    gross_paise  INTEGER NOT NULL,
    fees_paise   INTEGER NOT NULL,
    net_paise    INTEGER NOT NULL,
    status       TEXT NOT NULL
);

CREATE TABLE emails_sent (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id    INTEGER NOT NULL REFERENCES customers(id),
    template       TEXT NOT NULL,
    related_entity TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name   TEXT NOT NULL,
    args_json   TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

# Seeded history days used for charges and refunds; settlements missions target these.
HISTORY_DAYS = ("2025-12-30", "2025-12-31")

# Simulation "present": all agent-driven activity is timestamped after seeded history.
SIM_START = "2026-01-10T00:00:00"

_FIRST_NAMES = [
    "Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Ananya", "Kabir", "Sana",
    "Vikram", "Priya", "Arjun", "Nisha", "Dev", "Kavya", "Rahul", "Tara",
    "Nikhil", "Pooja", "Sameer", "Lata",
]
_LAST_NAMES = [
    "Sharma", "Patel", "Reddy", "Iyer", "Khan", "Gupta", "Nair", "Das",
    "Mehta", "Joshi", "Bose", "Rao", "Singh", "Kulkarni", "Chopra", "Verma",
    "Menon", "Pillai", "Saxena", "Bhat",
]

_PLANS = [("basic", 49_900), ("pro", 149_900), ("enterprise", 499_900)]

_ORDER_STATUS_POPULATION = [
    "placed", "shipped", "delivered", "cancelled", "refund_pending", "refunded",
]
_ORDER_STATUS_WEIGHTS = [20, 20, 35, 10, 5, 10]


class SimClock:
    """Deterministic monotonic clock.

    Every call to now() returns the current timestamp and advances the clock
    by a fixed step. Timestamps therefore depend only on call order, never on
    wall-clock time.
    """

    def __init__(self, start: str = SIM_START, step_seconds: int = 1) -> None:
        self.current = datetime.fromisoformat(start)
        self.step = timedelta(seconds=step_seconds)

    def now(self) -> str:
        """Return the current timestamp as ISO-8601 and advance the clock."""
        stamp = self.current.isoformat(timespec="seconds")
        self.current += self.step
        return stamp

    def today(self) -> str:
        """Return the current simulation date (YYYY-MM-DD) without advancing."""
        return self.current.date().isoformat()

    def in_days(self, days: int) -> str:
        """Return the timestamp `days` from now without advancing the clock."""
        return (self.current + timedelta(days=days)).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys enforced."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables. The database must be empty."""
    conn.executescript(SCHEMA)
    conn.commit()


def seed(conn: sqlite3.Connection, seed_value: int) -> None:
    """Populate the environment deterministically from an integer seed.

    Produces ~50 customers, 200 orders in mixed statuses (with completed
    refund rows backing every 'refunded' order), 40 subscriptions on 40
    distinct customers, and two merchant days of charge history.

    Guarantees (relied on by missions and tests):
    - The first 5 past_due subscriptions (by id) belong to customers whose
      balance covers the subscription amount (retry will succeed); the
      remaining past_due subscriptions belong to underfunded customers.
    - Every rng call is order-dependent: changing this function reshuffles
      all seeded data. That is acceptable; seeds are locked per experiment.
    """
    rng = random.Random(seed_value)

    # -- customers 1..50 -------------------------------------------------
    for cid in range(1, 51):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        balance = rng.randrange(0, 5001) * 100  # 0 .. ₹5000 in whole rupees
        conn.execute(
            "INSERT INTO customers (id, name, email, balance_paise) VALUES (?, ?, ?, ?)",
            (cid, f"{first} {last}", f"{first.lower()}.{last.lower()}{cid}@example.com", balance),
        )

    # -- orders 1001..1200 -----------------------------------------------
    statuses = rng.choices(_ORDER_STATUS_POPULATION, weights=_ORDER_STATUS_WEIGHTS, k=200)
    refunded_orders: list[tuple[int, int, int]] = []  # (order_id, customer_id, amount)
    for i, oid in enumerate(range(1001, 1201)):
        customer_id = rng.randrange(1, 51)
        amount = rng.randrange(100, 5001) * 100  # ₹100 .. ₹5000
        conn.execute(
            "INSERT INTO orders (id, customer_id, amount_paise, status) VALUES (?, ?, ?, ?)",
            (oid, customer_id, amount, statuses[i]),
        )
        if statuses[i] == "refunded":
            refunded_orders.append((oid, customer_id, amount))

    # -- refunds backing every refunded order ----------------------------
    # Spread across the two history days so settlements have refunds to subtract.
    for j, (oid, customer_id, amount) in enumerate(refunded_orders):
        created = f"{HISTORY_DAYS[j % 2]}T10:{j % 60:02d}:00"
        conn.execute(
            "INSERT INTO refunds (order_id, customer_id, amount_paise, status,"
            " created_at, idempotency_key) VALUES (?, ?, ?, 'completed', ?, NULL)",
            (oid, customer_id, amount, created),
        )

    # -- subscriptions 501..540 on 40 distinct customers -----------------
    sub_customers = rng.sample(range(1, 51), 40)
    sub_statuses = ["active"] * 28 + ["past_due"] * 8 + ["cancelled"] * 4
    rng.shuffle(sub_statuses)
    subs: list[tuple[int, int, int, str]] = []  # (sub_id, customer_id, amount, status)
    for k, sid in enumerate(range(501, 541)):
        plan, amount = rng.choice(_PLANS)
        status = sub_statuses[k]
        next_charge = "2026-01-15T00:00:00" if status == "active" else "2026-01-05T00:00:00"
        conn.execute(
            "INSERT INTO subscriptions (id, customer_id, plan, amount_paise, status,"
            " next_charge_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, sub_customers[k], plan, amount, status, next_charge),
        )
        subs.append((sid, sub_customers[k], amount, status))

    # -- guarantee a deterministic retry-outcome mix for past_due subs ---
    past_due = [s for s in subs if s[3] == "past_due"]
    for idx, (_sid, customer_id, amount, _status) in enumerate(past_due):
        funded = idx < 5
        balance = amount + 100_000 if funded else amount // 2
        conn.execute(
            "UPDATE customers SET balance_paise = ? WHERE id = ?", (balance, customer_id)
        )

    # -- charge history on the two merchant days -------------------------
    chargeable = [s for s in subs if s[3] != "cancelled"]
    for k in range(60):
        sid, _customer_id, amount, _status = rng.choice(chargeable)
        day = HISTORY_DAYS[k % 2]
        status = "succeeded" if rng.random() < 0.8 else "failed"
        created = f"{day}T{8 + k // 12:02d}:{(k * 7) % 60:02d}:{k % 60:02d}"
        conn.execute(
            "INSERT INTO charges (subscription_id, amount_paise, status, attempt_no,"
            " created_at) VALUES (?, ?, ?, 1, ?)",
            (sid, amount, status, created),
        )

    conn.commit()


def record_audit(
    conn: sqlite3.Connection,
    clock: SimClock,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Append a tool invocation to the audit log. Called by the harness, never the agent."""
    conn.execute(
        "INSERT INTO audit_log (tool_name, args_json, result_json, created_at)"
        " VALUES (?, ?, ?, ?)",
        (tool_name, json.dumps(args, sort_keys=True), json.dumps(result, sort_keys=True),
         clock.now()),
    )
    conn.commit()


def dump(conn: sqlite3.Connection) -> str:
    """Full-text dump of the database, used for determinism checks."""
    return "\n".join(conn.iterdump())
