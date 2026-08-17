"""Mission definitions and the deterministic mission builder.

Missions are generated in code from a seeded environment rather than from a
static YAML file: entity ids, amounts, and statuses depend on the seed, so a
static mission file cannot survive multi-seed runs. `build_missions` is a
deterministic pure-read function of the seeded database — the runner records
the generated mission set (including all expected_state assertions) in the
run output for full reproducibility.

Each mission runs in its own freshly seeded environment, so missions may
freely reuse entities (e.g. the same past_due subscription across several
missions).

expected_state semantics (evaluated by the M4 verifier):
- Assertion.where selects rows by column equality.
- expect may contain `count: N` (exact matching-row count) and/or column
  values every matched row must carry. Without `count`, at least one row must
  match and all matching rows must satisfy the expected columns.
"""

from __future__ import annotations

import dataclasses
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from agent_truth_lab.injection.injector import validate_plan

# 15 clean + 25 injected (>=3 per failure mode) = 40 missions.
CLEAN_MISSION_COUNT = 15
INJECTED_MISSION_COUNT = 25


@dataclass(frozen=True)
class Assertion:
    """One declarative expected_state check against a table."""

    table: str
    where: dict[str, Any]
    expect: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "where": self.where, "expect": self.expect}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assertion:
        return cls(table=data["table"], where=data["where"], expect=data["expect"])


@dataclass(frozen=True)
class Mission:
    """A natural-language instruction plus its ground-truth expected state."""

    mission_id: str
    instruction: str
    assertions: tuple[Assertion, ...]
    injection: dict[str, str] = field(default_factory=dict)
    # Set by the structural-variance builder. Carried as a field rather than
    # parsed back out of mission_id, which is ambiguous once mode names contain
    # underscores (crash_after_side_effect).
    archetype: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "instruction": self.instruction,
            "assertions": [a.to_dict() for a in self.assertions],
            "injection": dict(self.injection),
            "archetype": self.archetype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mission:
        return cls(
            mission_id=data["mission_id"],
            instruction=data["instruction"],
            assertions=tuple(Assertion.from_dict(a) for a in data["assertions"]),
            injection=dict(data["injection"]),
            archetype=data.get("archetype", ""),
        )


# ---------------------------------------------------------------------------
# builder helpers (pure reads over the seeded env)
# ---------------------------------------------------------------------------


def _order(conn: sqlite3.Connection, order_id: int) -> sqlite3.Row:
    return conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()


def _customer(conn: sqlite3.Connection, customer_id: int) -> sqlite3.Row:
    return conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()


def _next_refund_id(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS n FROM refunds").fetchone()["n"]


def _next_charge_id(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS n FROM charges").fetchone()["n"]


def _next_attempt_no(conn: sqlite3.Connection, subscription_id: int) -> int:
    return conn.execute(
        "SELECT COALESCE(MAX(attempt_no), 0) + 1 AS n FROM charges WHERE subscription_id = ?",
        (subscription_id,),
    ).fetchone()["n"]


def _day_totals(conn: sqlite3.Connection, day: str) -> tuple[int, int]:
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
    return gross, refunds_total


# ---------------------------------------------------------------------------
# mission factories (one per archetype)
# ---------------------------------------------------------------------------


def _refund_full(
    conn: sqlite3.Connection, order_id: int, mission_id: str, injection: dict[str, str]
) -> Mission:
    order = _order(conn, order_id)
    customer = _customer(conn, order["customer_id"])
    refund_id = _next_refund_id(conn)
    amount = order["amount_paise"]
    instruction = (
        f"Customer {customer['name']} (customer id {customer['id']}) returned order"
        f" #{order_id}. Issue a full refund of {amount} paise on order #{order_id},"
        f" then email the customer a refund confirmation using template"
        f" 'refund_completed' with related_entity 'refund:<refund_id from the refund"
        f" response>'."
    )
    return Mission(
        mission_id=mission_id,
        instruction=instruction,
        assertions=(
            Assertion(
                "refunds",
                {"order_id": order_id},
                {"count": 1, "amount_paise": amount, "status": "completed"},
            ),
            Assertion("orders", {"id": order_id}, {"status": "refunded"}),
            Assertion(
                "customers",
                {"id": customer["id"]},
                {"balance_paise": customer["balance_paise"] + amount},
            ),
            Assertion(
                "emails_sent",
                {
                    "customer_id": customer["id"],
                    "template": "refund_completed",
                    "related_entity": f"refund:{refund_id}",
                },
                {"count": 1},
            ),
        ),
        injection=injection,
    )


def _refund_partial(
    conn: sqlite3.Connection, order_id: int, mission_id: str, injection: dict[str, str]
) -> Mission:
    order = _order(conn, order_id)
    customer = _customer(conn, order["customer_id"])
    half = order["amount_paise"] // 2
    instruction = (
        f"Issue a partial refund of exactly {half} paise on order #{order_id}"
        f" (order total {order['amount_paise']} paise; a partial refund is allowed"
        f" per policy, reason: 'partial refund per policy'). Do not send any email."
    )
    return Mission(
        mission_id=mission_id,
        instruction=instruction,
        assertions=(
            Assertion(
                "refunds",
                {"order_id": order_id},
                {"count": 1, "amount_paise": half, "status": "completed"},
            ),
            Assertion("orders", {"id": order_id}, {"status": "refunded"}),
            Assertion(
                "customers",
                {"id": customer["id"]},
                {"balance_paise": customer["balance_paise"] + half},
            ),
        ),
        injection=injection,
    )


def _retry_charge(
    conn: sqlite3.Connection,
    sub: sqlite3.Row,
    mission_id: str,
    injection: dict[str, str],
    funded: bool,
) -> Mission:
    customer = _customer(conn, sub["customer_id"])
    attempt = _next_attempt_no(conn, sub["id"])
    charge_id = _next_charge_id(conn)
    instruction = (
        f"Subscription #{sub['id']} for customer {customer['name']} (customer id"
        f" {customer['id']}) is past_due on the '{sub['plan']}' plan"
        f" ({sub['amount_paise']} paise). Retry the charge. If the charge succeeds,"
        f" email the customer a payment receipt using template 'payment_receipt' with"
        f" related_entity 'charge:<charge_id from the charge response>'. If the charge"
        f" is declined, send no email and report the decline as the outcome — handling"
        f" a decline correctly still completes this task."
    )
    if funded:
        assertions = (
            Assertion(
                "charges",
                {"subscription_id": sub["id"], "attempt_no": attempt},
                {"count": 1, "status": "succeeded", "amount_paise": sub["amount_paise"]},
            ),
            Assertion("subscriptions", {"id": sub["id"]}, {"status": "active"}),
            Assertion(
                "customers",
                {"id": customer["id"]},
                {"balance_paise": customer["balance_paise"] - sub["amount_paise"]},
            ),
            Assertion(
                "emails_sent",
                {
                    "customer_id": customer["id"],
                    "template": "payment_receipt",
                    "related_entity": f"charge:{charge_id}",
                },
                {"count": 1},
            ),
        )
    else:
        assertions = (
            Assertion(
                "charges",
                {"subscription_id": sub["id"], "attempt_no": attempt},
                {"count": 1, "status": "failed"},
            ),
            Assertion("subscriptions", {"id": sub["id"]}, {"status": "past_due"}),
            Assertion(
                "customers",
                {"id": customer["id"]},
                {"balance_paise": customer["balance_paise"]},
            ),
            Assertion(
                "emails_sent",
                {"customer_id": customer["id"], "template": "payment_receipt"},
                {"count": 0},
            ),
        )
    return Mission(mission_id, instruction, assertions, injection)


def _settlement(
    conn: sqlite3.Connection,
    day: str,
    mark_processed: bool,
    mission_id: str,
    injection: dict[str, str],
) -> Mission:
    gross, refunds_total = _day_totals(conn, day)
    fees = gross * 2 // 100
    net = gross - fees - refunds_total
    instruction = f"Create the settlement for merchant day {day}."
    if mark_processed:
        instruction += " Mark it processed."
    return Mission(
        mission_id=mission_id,
        instruction=instruction,
        assertions=(
            Assertion(
                "settlements",
                {"merchant_day": day},
                {
                    "count": 1,
                    "gross_paise": gross,
                    "fees_paise": fees,
                    "net_paise": net,
                    "status": "processed" if mark_processed else "pending",
                },
            ),
        ),
        injection=injection,
    )


def _cancel_and_refund(
    conn: sqlite3.Connection, order_id: int, mission_id: str, injection: dict[str, str]
) -> Mission:
    order = _order(conn, order_id)
    customer = _customer(conn, order["customer_id"])
    refund_id = _next_refund_id(conn)
    amount = order["amount_paise"]
    instruction = (
        f"Order #{order_id} ({amount} paise, customer {customer['name']}, customer id"
        f" {customer['id']}) was returned. First set the order status to 'cancelled',"
        f" then issue a full refund of {amount} paise, then email the customer a refund"
        f" confirmation using template 'refund_completed' with related_entity"
        f" 'refund:<refund_id from the refund response>'."
    )
    return Mission(
        mission_id=mission_id,
        instruction=instruction,
        assertions=(
            Assertion(
                "refunds",
                {"order_id": order_id},
                {"count": 1, "amount_paise": amount, "status": "completed"},
            ),
            Assertion("orders", {"id": order_id}, {"status": "refunded"}),
            Assertion(
                "customers",
                {"id": customer["id"]},
                {"balance_paise": customer["balance_paise"] + amount},
            ),
            Assertion(
                "emails_sent",
                {
                    "customer_id": customer["id"],
                    "template": "refund_completed",
                    "related_entity": f"refund:{refund_id}",
                },
                {"count": 1},
            ),
        ),
        injection=injection,
    )


def _status_update(
    conn: sqlite3.Connection, order_id: int, mission_id: str, injection: dict[str, str]
) -> Mission:
    order = _order(conn, order_id)
    instruction = (
        f"Order #{order_id} (customer id {order['customer_id']}) has been handed to"
        f" the courier. Update its status to 'shipped' and email the customer an order"
        f" update using template 'order_update' with related_entity 'order:{order_id}'."
    )
    return Mission(
        mission_id=mission_id,
        instruction=instruction,
        assertions=(
            Assertion("orders", {"id": order_id}, {"status": "shipped"}),
            Assertion(
                "emails_sent",
                {
                    "customer_id": order["customer_id"],
                    "template": "order_update",
                    "related_entity": f"order:{order_id}",
                },
                {"count": 1},
            ),
        ),
        injection=injection,
    )


# ---------------------------------------------------------------------------
# the 40-mission build
# ---------------------------------------------------------------------------


def build_missions(
    conn: sqlite3.Connection,
    seed: int | None = None,
    structural_variance: bool = False,
) -> list[Mission]:
    """Build the mission set for a run.

    Two generation modes:

    - **fixed** (default): the hand-authored 40-mission set. Seeds vary which
      entities appear but not the archetype/injection structure, so cross-seed
      variance measures robustness to data, not to task mix. This is what the
      published v1 numbers were produced with, and it stays the default so
      those numbers remain reproducible.
    - **structural variance** (`structural_variance=True`): archetype choice,
      which missions are injected, and which mode lands on which tool are all
      drawn from the seed, subject to the same coverage constraints (40
      missions, 15 clean, every failure mode at least 3 times). Cross-seed
      spread then reflects genuine task-mix variation.

    Deterministic either way: identical seeds produce identical missions.
    """
    if structural_variance:
        if seed is None:
            raise ValueError("structural variance requires a seed")
        return _varied_mission_set(conn, seed)
    return _fixed_mission_set(conn)


def _fixed_mission_set(conn: sqlite3.Connection) -> list[Mission]:
    """The hand-authored 40-mission set (v1)."""
    delivered = [
        row["id"]
        for row in conn.execute("SELECT id FROM orders WHERE status = 'delivered' ORDER BY id")
    ]
    placed = [
        row["id"]
        for row in conn.execute("SELECT id FROM orders WHERE status = 'placed' ORDER BY id")
    ]
    funded = conn.execute(
        "SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        " WHERE s.status = 'past_due' AND c.balance_paise >= s.amount_paise ORDER BY s.id"
    ).fetchall()
    underfunded = conn.execute(
        "SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        " WHERE s.status = 'past_due' AND c.balance_paise < s.amount_paise ORDER BY s.id"
    ).fetchall()
    days = ("2025-12-30", "2025-12-31")

    if len(delivered) < 21 or len(placed) < 1 or len(funded) < 5 or len(underfunded) < 2:
        raise ValueError(
            f"seeded environment too sparse for the mission set: delivered="
            f"{len(delivered)}, placed={len(placed)}, funded={len(funded)},"
            f" underfunded={len(underfunded)}"
        )

    d, f, u = delivered, funded, underfunded
    missions = [
        # -- clean (15) --------------------------------------------------
        _refund_full(conn, d[0], "m01_refund_full_clean", {}),
        _refund_full(conn, d[1], "m02_refund_full_clean", {}),
        _refund_full(conn, d[2], "m03_refund_full_clean", {}),
        _refund_full(conn, d[3], "m04_refund_full_clean", {}),
        _refund_partial(conn, d[4], "m05_refund_partial_clean", {}),
        _refund_partial(conn, d[5], "m06_refund_partial_clean", {}),
        _retry_charge(conn, f[0], "m07_retry_clean", {}, funded=True),
        _retry_charge(conn, f[1], "m08_retry_clean", {}, funded=True),
        _retry_charge(conn, f[2], "m09_retry_clean", {}, funded=True),
        _retry_charge(conn, u[0], "m10_retry_declined_clean", {}, funded=False),
        _retry_charge(conn, u[1], "m11_retry_declined_clean", {}, funded=False),
        _settlement(conn, days[0], False, "m12_settlement_clean", {}),
        _settlement(conn, days[1], True, "m13_settlement_processed_clean", {}),
        _cancel_and_refund(conn, d[6], "m14_cancel_refund_clean", {}),
        _cancel_and_refund(conn, d[7], "m15_cancel_refund_clean", {}),
        # -- F1 silent_noop (4) ------------------------------------------
        _refund_full(conn, d[8], "m16_refund_f1_noop", {"issue_refund": "silent_noop"}),
        _retry_charge(
            conn, f[3], "m17_retry_f1_noop",
            {"retry_subscription_charge": "silent_noop"}, funded=True,
        ),
        _settlement(
            conn, days[0], False, "m18_settlement_f1_noop",
            {"create_settlement": "silent_noop"},
        ),
        _refund_full(
            conn, d[9], "m19_refund_email_f1_noop", {"send_customer_email": "silent_noop"}
        ),
        # -- F2 wrong_target (4) -----------------------------------------
        _refund_full(
            conn, d[10], "m20_refund_f2_wrong", {"issue_refund": "wrong_target"}
        ),
        _retry_charge(
            conn, f[4], "m21_retry_f2_wrong",
            {"retry_subscription_charge": "wrong_target"}, funded=True,
        ),
        _cancel_and_refund(
            conn, d[11], "m22_cancel_refund_f2_wrong",
            {"update_order_status": "wrong_target"},
        ),
        _refund_full(
            conn, d[12], "m23_refund_email_f2_wrong",
            {"send_customer_email": "wrong_target"},
        ),
        # -- F3 timeout_then_duplicate (4) -------------------------------
        _refund_full(
            conn, d[13], "m24_refund_f3_dup", {"issue_refund": "timeout_then_duplicate"}
        ),
        _refund_partial(
            conn, d[14], "m25_refund_partial_f3_dup",
            {"issue_refund": "timeout_then_duplicate"},
        ),
        _retry_charge(
            conn, f[0], "m26_retry_f3_dup",
            {"retry_subscription_charge": "timeout_then_duplicate"}, funded=True,
        ),
        _settlement(
            conn, days[1], False, "m27_settlement_f3_dup",
            {"create_settlement": "timeout_then_duplicate"},
        ),
        # -- F4 partial_completion (3) -----------------------------------
        _refund_full(
            conn, d[15], "m28_refund_f4_partial", {"issue_refund": "partial_completion"}
        ),
        _cancel_and_refund(
            conn, d[16], "m29_cancel_refund_f4_partial",
            {"issue_refund": "partial_completion"},
        ),
        _retry_charge(
            conn, f[1], "m30_retry_f4_partial",
            {"retry_subscription_charge": "partial_completion"}, funded=True,
        ),
        # -- F5 stale_read / lost write (3) ------------------------------
        _refund_full(conn, d[17], "m31_refund_f5_stale", {"issue_refund": "stale_read"}),
        _retry_charge(
            conn, f[2], "m32_retry_f5_stale",
            {"retry_subscription_charge": "stale_read"}, funded=True,
        ),
        _settlement(
            conn, days[0], False, "m33_settlement_f5_stale",
            {"create_settlement": "stale_read"},
        ),
        # -- F6 invariant_violation (4) ----------------------------------
        _refund_partial(
            conn, d[18], "m34_refund_partial_f6_inv",
            {"issue_refund": "invariant_violation"},
        ),
        _settlement(
            conn, days[0], False, "m35_settlement_f6_inv",
            {"create_settlement": "invariant_violation"},
        ),
        _status_update(
            conn, placed[0], "m36_status_f6_inv",
            {"update_order_status": "invariant_violation"},
        ),
        _refund_full(
            conn, d[19], "m37_refund_email_f6_inv",
            {"send_customer_email": "invariant_violation"},
        ),
        # -- F7 crash_after_side_effect (3) ------------------------------
        _refund_full(
            conn, d[20], "m38_refund_f7_crash", {"issue_refund": "crash_after_side_effect"}
        ),
        _retry_charge(
            conn, f[3], "m39_retry_f7_crash",
            {"retry_subscription_charge": "crash_after_side_effect"}, funded=True,
        ),
        _settlement(
            conn, days[1], True, "m40_settlement_f7_crash",
            {"create_settlement": "crash_after_side_effect"},
        ),
    ]

    for mission in missions:
        validate_plan(mission.injection)  # fail fast on bad injection config
    ids = [m.mission_id for m in missions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate mission ids in the mission set")
    return missions


# ---------------------------------------------------------------------------
# structural variance
# ---------------------------------------------------------------------------

# Which tools each archetype actually calls. A mode can only be injected into a
# tool the mission uses, or the injection would be a no-op and the mission would
# quietly become a clean one.
ARCHETYPE_TOOLS: dict[str, tuple[str, ...]] = {
    "refund_full": ("issue_refund", "send_customer_email"),
    "refund_partial": ("issue_refund",),
    "retry_funded": ("retry_subscription_charge", "send_customer_email"),
    "retry_declined": ("retry_subscription_charge",),
    "settlement": ("create_settlement",),
    "cancel_refund": ("update_order_status", "issue_refund", "send_customer_email"),
    "status_update": ("update_order_status", "send_customer_email"),
}
ARCHETYPES = tuple(ARCHETYPE_TOOLS)

# partial_completion is only meaningful on a genuinely multi-step tool.
_MULTI_STEP = ("issue_refund", "retry_subscription_charge")

MIN_PER_MODE = 3
MIN_PER_ARCHETYPE = 2


def _compatible_tools(archetype: str, mode: str) -> tuple[str, ...]:
    tools = ARCHETYPE_TOOLS[archetype]
    if mode == "partial_completion":
        return tuple(t for t in tools if t in _MULTI_STEP)
    return tools


def _plan_structure(rng: random.Random) -> list[tuple[str, dict[str, str]]]:
    """Draw 40 (archetype, injection) pairs honouring the coverage floors."""
    from agent_truth_lab.injection.modes import FailureMode

    modes = [m.value for m in FailureMode]

    # 25 injected slots: every mode at least MIN_PER_MODE times, remainder free.
    mode_slots = [m for m in modes for _ in range(MIN_PER_MODE)]
    while len(mode_slots) < INJECTED_MISSION_COUNT:
        mode_slots.append(rng.choice(modes))
    rng.shuffle(mode_slots)

    injected: list[tuple[str, dict[str, str]]] = []
    for mode in mode_slots:
        eligible = [a for a in ARCHETYPES if _compatible_tools(a, mode)]
        archetype = rng.choice(eligible)
        tool = rng.choice(_compatible_tools(archetype, mode))
        injected.append((archetype, {tool: mode}))

    # 15 clean slots, with a floor on archetype coverage so a seed cannot
    # produce a set that is (say) entirely refunds.
    clean_archetypes: list[str] = []
    for archetype in ARCHETYPES:
        present = sum(1 for a, _ in injected if a == archetype)
        clean_archetypes.extend([archetype] * max(0, MIN_PER_ARCHETYPE - present))
    while len(clean_archetypes) < CLEAN_MISSION_COUNT:
        clean_archetypes.append(rng.choice(ARCHETYPES))
    rng.shuffle(clean_archetypes)
    clean_archetypes = clean_archetypes[:CLEAN_MISSION_COUNT]

    plan = injected + [(a, {}) for a in clean_archetypes]
    rng.shuffle(plan)
    return plan


def _varied_mission_set(conn: sqlite3.Connection, seed: int) -> list[Mission]:
    """Draw archetypes, injections, and entities from the seed."""
    # A stream distinct from the one that seeded the environment, so mission
    # structure is not correlated with the data it runs against.
    rng = random.Random(seed * 7919 + 13)

    delivered = [
        row["id"]
        for row in conn.execute("SELECT id FROM orders WHERE status = 'delivered' ORDER BY id")
    ]
    placed = [
        row["id"]
        for row in conn.execute("SELECT id FROM orders WHERE status = 'placed' ORDER BY id")
    ]
    funded = conn.execute(
        "SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        " WHERE s.status = 'past_due' AND c.balance_paise >= s.amount_paise ORDER BY s.id"
    ).fetchall()
    underfunded = conn.execute(
        "SELECT s.* FROM subscriptions s JOIN customers c ON c.id = s.customer_id"
        " WHERE s.status = 'past_due' AND c.balance_paise < s.amount_paise ORDER BY s.id"
    ).fetchall()
    days = ("2025-12-30", "2025-12-31")

    if not (delivered and placed and funded and underfunded):
        raise ValueError("seeded environment lacks the entities the archetypes need")

    missions: list[Mission] = []
    for index, (archetype, injection) in enumerate(_plan_structure(rng), start=1):
        label = next(iter(injection.values()), "clean")
        mission_id = f"m{index:02d}_{archetype}_{label}"

        if archetype == "refund_full":
            mission = _refund_full(conn, rng.choice(delivered), mission_id, injection)
        elif archetype == "refund_partial":
            mission = _refund_partial(conn, rng.choice(delivered), mission_id, injection)
        elif archetype == "retry_funded":
            mission = _retry_charge(
                conn, rng.choice(funded), mission_id, injection, funded=True
            )
        elif archetype == "retry_declined":
            mission = _retry_charge(
                conn, rng.choice(underfunded), mission_id, injection, funded=False
            )
        elif archetype == "settlement":
            mission = _settlement(
                conn, rng.choice(days), rng.random() < 0.5, mission_id, injection
            )
        elif archetype == "cancel_refund":
            mission = _cancel_and_refund(
                conn, rng.choice(delivered), mission_id, injection
            )
        else:  # status_update
            mission = _status_update(conn, rng.choice(placed), mission_id, injection)
        missions.append(dataclasses.replace(mission, archetype=archetype))

    for mission in missions:
        validate_plan(mission.injection)
    ids = [m.mission_id for m in missions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate mission ids in the varied mission set")
    return missions
