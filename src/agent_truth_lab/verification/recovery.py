"""Arm D's recovery playbook — rule-based, deterministic, no LLM in the loop.

Given a FAILED verification, recovery does four things in order:

  re-read state -> classify the divergence -> repair -> re-verify

Every repair decision is a rule. An LLM may be used *only* to render the
human-readable incident summary for an escalation (see `render_incident`,
which takes an optional client and defaults to a deterministic report) —
never to decide what to repair.

Two invariants make this safe to run against a payments environment:

1. **Recovery never worsens state.** The post-repair snapshot is compared
   against the pre-repair one; if it is worse by any measure (fewer satisfied
   assertions, more invariant violations, more out-of-frame changes), the
   entire repair is discarded and the original snapshot is restored. The
   experiment measures recovery-induced damage and this guard is what keeps
   it at zero.
2. **Repairs run on a copy.** The caller passes the episode's snapshot; the
   result carries a *new* dump. Arm C's measurement reads the pristine
   snapshot and is never contaminated.

Recovery also gets a capability the agent did not have: an idempotency key on
refunds. That asymmetry is the point — assurance and repair logic is allowed
to use a safer API than the actor being audited.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_truth_lab.agent.missions import Mission
from agent_truth_lab.environment import db, tools
from agent_truth_lab.verification.verifier import (
    AssertionResult,
    StateChange,
    Verdict,
    VerificationResult,
    verify,
)

# A cap on repair work, so a pathological case escalates instead of looping.
MAX_ACTIONS = 24

# Repairs are applied in this order so that money movement settles last: the
# balance repair sets the exact expected value after every other effect lands.
REPAIR_ORDER = (
    "refunds", "charges", "settlements", "emails_sent", "orders", "subscriptions",
    "customers",
)


class DivergenceKind(StrEnum):
    MISSING_EFFECT = "missing_effect"
    WRONG_TARGET = "wrong_target"
    DUPLICATE = "duplicate"
    PARTIAL = "partial"
    INVARIANT_BREACH = "invariant_breach"
    VALUE_MISMATCH = "value_mismatch"
    UNKNOWN = "unknown"


class RecoveryOutcome(StrEnum):
    NOT_NEEDED = "NOT_NEEDED"
    RECOVERED = "RECOVERED"
    ESCALATED = "ESCALATED"


@dataclass(frozen=True)
class Diagnosis:
    """One classified divergence between intended and actual state."""

    kind: DivergenceKind
    table: str
    detail: str
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "table": self.table,
            "detail": self.detail,
            "reversible": self.reversible,
        }


@dataclass(frozen=True)
class RecoveryAction:
    """One repair the playbook applied (or declined to apply)."""

    action: str
    table: str
    target: Any
    detail: str
    applied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "table": self.table,
            "target": self.target,
            "detail": self.detail,
            "applied": self.applied,
        }


@dataclass
class RecoveryResult:
    """The full record of a recovery attempt."""

    outcome: RecoveryOutcome
    diagnoses: list[Diagnosis] = field(default_factory=list)
    actions: list[RecoveryAction] = field(default_factory=list)
    rolled_back: bool = False
    db_dump: str = ""
    pre_verification: VerificationResult | None = None
    post_verification: VerificationResult | None = None
    incident: str | None = None

    @property
    def recovered(self) -> bool:
        return self.outcome is RecoveryOutcome.RECOVERED

    @property
    def escalated(self) -> bool:
        return self.outcome is RecoveryOutcome.ESCALATED

    @property
    def caused_damage(self) -> bool:
        """True only if recovery left the system worse off than it found it.

        Always False by construction while the rollback guard holds; the
        experiment reports it as a measured value rather than an assumption.
        """
        if self.post_verification is None or self.pre_verification is None:
            return False
        if self.rolled_back:
            return False
        return is_worse(self.post_verification, self.pre_verification)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "actions": [a.to_dict() for a in self.actions],
            "rolled_back": self.rolled_back,
            "caused_damage": self.caused_damage,
            "incident": self.incident,
            "pre_verification": (
                self.pre_verification.to_dict() if self.pre_verification else None
            ),
            "post_verification": (
                self.post_verification.to_dict() if self.post_verification else None
            ),
        }


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------


def _diagnose_assertion(result: AssertionResult) -> Diagnosis:
    """Classify a single failed assertion."""
    table = result.assertion.table
    if result.expected_count is not None:
        if result.actual_count > result.expected_count:
            return Diagnosis(
                DivergenceKind.DUPLICATE,
                table,
                f"{result.actual_count} rows where {result.expected_count} expected",
                reversible=table != "emails_sent",  # an email cannot be un-sent
            )
        if result.actual_count < result.expected_count:
            return Diagnosis(
                DivergenceKind.MISSING_EFFECT,
                table,
                f"{result.actual_count} rows where {result.expected_count} expected",
            )
    if result.actual_count == 0:
        return Diagnosis(DivergenceKind.MISSING_EFFECT, table, "no matching row")
    return Diagnosis(DivergenceKind.VALUE_MISMATCH, table, result.detail)


def diagnose(result: VerificationResult) -> list[Diagnosis]:
    """Classify every divergence in a failed verification. Pure function."""
    diagnoses = [
        _diagnose_assertion(r) for r in result.assertion_results if not r.satisfied
    ]
    for change in result.unexpected_changes:
        diagnoses.append(
            Diagnosis(
                DivergenceKind.WRONG_TARGET,
                change.table,
                f"out-of-frame {change.kind} on row {change.row_id}: {change.detail}",
                reversible=change.table != "emails_sent",
            )
        )
    for violation in result.violations:
        diagnoses.append(
            Diagnosis(
                DivergenceKind.INVARIANT_BREACH,
                violation.table,
                f"{violation.rule}: {violation.detail}",
                reversible=violation.rule != "refund_email_references_refund",
            )
        )
    if result.is_partial:
        diagnoses.append(
            Diagnosis(
                DivergenceKind.PARTIAL,
                "-",
                f"{result.satisfied_assertions}/{len(result.assertion_results)}"
                " assertions satisfied — multi-step operation stopped midway",
            )
        )
    return diagnoses or [Diagnosis(DivergenceKind.UNKNOWN, "-", "no divergence isolated")]


# ---------------------------------------------------------------------------
# repair primitives
# ---------------------------------------------------------------------------


def idempotent_refund(
    conn: sqlite3.Connection,
    clock: db.SimClock,
    order_id: int,
    amount_paise: int,
    reason: str,
    idempotency_key: str,
) -> tuple[bool, str]:
    """Issue a refund at most once for a given key. Returns (applied, detail).

    The capability the agent's tool deliberately lacks: a replayed call with a
    known key is a no-op instead of a duplicate side effect.
    """
    existing = conn.execute(
        "SELECT id FROM refunds WHERE idempotency_key = ?", (idempotency_key,)
    ).fetchone()
    if existing is not None:
        return False, f"idempotency key already applied (refund {existing['id']})"

    result = tools.issue_refund(conn, clock, order_id, amount_paise, reason)
    if not result.ok:
        return False, f"refund rejected: {result.error}"
    conn.execute(
        "UPDATE refunds SET idempotency_key = ? WHERE id = ?",
        (idempotency_key, result.data["refund_id"]),
    )
    conn.commit()
    return True, f"issued refund {result.data['refund_id']} with key {idempotency_key}"


def _baseline(seed: int) -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, seed)
    return conn


def _reverse_change(
    conn: sqlite3.Connection, baseline: sqlite3.Connection, change: StateChange
) -> RecoveryAction:
    """Undo one out-of-frame change by restoring the seeded baseline row."""
    original = baseline.execute(
        f"SELECT * FROM {change.table} WHERE id = ?",  # noqa: S608 - fixed table set
        (change.row_id,),
    ).fetchone()

    if change.table == "emails_sent" and change.kind == "added":
        return RecoveryAction(
            "escalate_irreversible", change.table, change.row_id,
            "an email was sent to the wrong customer and cannot be recalled",
            applied=False,
        )

    if change.kind == "added":
        # Reverse the money the stray effect moved, then drop the row.
        if change.table == "refunds":
            row = conn.execute(
                "SELECT * FROM refunds WHERE id = ?", (change.row_id,)
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE customers SET balance_paise = balance_paise - ? WHERE id = ?",
                    (row["amount_paise"], row["customer_id"]),
                )
        elif change.table == "charges":
            row = conn.execute(
                "SELECT c.amount_paise, s.customer_id FROM charges c"
                " JOIN subscriptions s ON s.id = c.subscription_id WHERE c.id = ?",
                (change.row_id,),
            ).fetchone()
            if row is not None and row["amount_paise"]:
                conn.execute(
                    "UPDATE customers SET balance_paise = balance_paise + ? WHERE id = ?",
                    (row["amount_paise"], row["customer_id"]),
                )
        conn.execute(
            f"DELETE FROM {change.table} WHERE id = ?",  # noqa: S608 - fixed table set
            (change.row_id,),
        )
        conn.commit()
        return RecoveryAction(
            "reverse_stray_write", change.table, change.row_id,
            "deleted the out-of-frame row and reversed its balance effect",
        )

    if change.kind == "modified" and original is not None:
        # sqlite3.Row iterates over values, so .keys() is the correct access here.
        columns = [k for k in original.keys() if k != "id"]  # noqa: SIM118
        assignments = ", ".join(f"{c} = ?" for c in columns)
        conn.execute(
            f"UPDATE {change.table} SET {assignments} WHERE id = ?",  # noqa: S608
            (*[original[c] for c in columns], change.row_id),
        )
        conn.commit()
        return RecoveryAction(
            "restore_baseline_row", change.table, change.row_id,
            "restored the row to its pre-episode values",
        )

    return RecoveryAction(
        "escalate_irreversible", change.table, change.row_id,
        f"cannot reverse a '{change.kind}' change automatically", applied=False,
    )


def _reassign_stray_refunds(
    conn: sqlite3.Connection,
    pre: VerificationResult,
    handled: set[tuple[str, int]],
) -> list[RecoveryAction]:
    """Re-point a misapplied refund at the order it was meant for.

    The compensating action for F2 (wrong_target). Deleting the stray row and
    issuing a fresh refund looks equivalent but is not: anything already
    referencing the original refund — a confirmation email, a downstream
    ledger entry — is orphaned by the delete, which trades one invariant
    violation for another. Correcting the misposted record in place reverses
    the money on the wrong account *and* keeps referential integrity intact.
    """
    missing = [
        r
        for r in pre.assertion_results
        if not r.satisfied
        and r.assertion.table == "refunds"
        and r.actual_count < (r.expected_count if r.expected_count is not None else 1)
    ]
    strays = [
        c for c in pre.unexpected_changes if c.table == "refunds" and c.kind == "added"
    ]
    actions: list[RecoveryAction] = []

    for result, change in zip(missing, strays, strict=False):
        row = conn.execute(
            "SELECT * FROM refunds WHERE id = ?", (change.row_id,)
        ).fetchone()
        target_order_id = result.assertion.where.get("order_id")
        target_amount = result.assertion.expect.get("amount_paise")
        if row is None or target_order_id is None or target_amount is None:
            continue
        order = conn.execute(
            "SELECT * FROM orders WHERE id = ?", (target_order_id,)
        ).fetchone()
        if order is None or target_amount > order["amount_paise"]:
            continue

        conn.execute(
            "UPDATE refunds SET order_id = ?, customer_id = ?, amount_paise = ?,"
            " status = 'completed' WHERE id = ?",
            (target_order_id, order["customer_id"], target_amount, row["id"]),
        )
        conn.commit()
        # The stray order and customer are restored to baseline by the generic
        # reversal pass; only the refund row itself is claimed here.
        handled.add(("refunds", row["id"]))
        actions.append(
            RecoveryAction(
                "reassign_misapplied_refund",
                "refunds",
                row["id"],
                f"re-pointed refund {row['id']} from order {row['order_id']}"
                f" to order {target_order_id} ({target_amount} paise)",
            )
        )
    return actions


def _repair_refunds(
    conn: sqlite3.Connection, clock: db.SimClock, result: AssertionResult, mission: Mission
) -> list[RecoveryAction]:
    where, expect = result.assertion.where, result.assertion.expect
    order_id = where.get("order_id")
    expected = result.expected_count if result.expected_count is not None else 1
    actions: list[RecoveryAction] = []

    if result.actual_count > expected:
        # Duplicate: keep the oldest, compensate and delete the rest.
        extras = conn.execute(
            "SELECT * FROM refunds WHERE order_id = ? ORDER BY id",
            (order_id,),
        ).fetchall()[expected:]
        for row in extras:
            conn.execute(
                "UPDATE customers SET balance_paise = balance_paise - ? WHERE id = ?",
                (row["amount_paise"], row["customer_id"]),
            )
            conn.execute("DELETE FROM refunds WHERE id = ?", (row["id"],))
            actions.append(
                RecoveryAction(
                    "compensate_duplicate_refund", "refunds", row["id"],
                    f"reversed duplicate refund of {row['amount_paise']} paise",
                )
            )
        conn.commit()
        return actions

    if result.actual_count < expected and order_id is not None:
        amount = expect.get("amount_paise")
        if amount is None:
            return [
                RecoveryAction(
                    "escalate_underspecified", "refunds", order_id,
                    "assertion does not pin the refund amount", applied=False,
                )
            ]
        applied, detail = idempotent_refund(
            conn, clock, order_id, amount, "automated recovery",
            f"recovery:{mission.mission_id}:order:{order_id}",
        )
        return [
            RecoveryAction("retry_refund_idempotent", "refunds", order_id, detail, applied)
        ]

    # Right number of rows, wrong values (e.g. F6's corrupted amount).
    for row in conn.execute(
        "SELECT * FROM refunds WHERE order_id = ? ORDER BY id", (order_id,)
    ).fetchall():
        for column, want in expect.items():
            if column == "count" or row[column] == want:
                continue
            conn.execute(
                f"UPDATE refunds SET {column} = ? WHERE id = ?",  # noqa: S608
                (want, row["id"]),
            )
            actions.append(
                RecoveryAction(
                    "correct_refund_value", "refunds", row["id"],
                    f"set {column} from {row[column]!r} to {want!r}",
                )
            )
    conn.commit()
    return actions


def _repair_emails(
    conn: sqlite3.Connection, clock: db.SimClock, result: AssertionResult
) -> list[RecoveryAction]:
    where = result.assertion.where
    expected = result.expected_count if result.expected_count is not None else 1
    if result.actual_count > expected:
        return [
            RecoveryAction(
                "escalate_irreversible", "emails_sent", where.get("customer_id"),
                f"{result.actual_count} emails sent where {expected} expected;"
                " an email cannot be recalled",
                applied=False,
            )
        ]
    if result.actual_count >= expected:
        return []
    customer_id = where.get("customer_id")
    template = where.get("template")
    related = where.get("related_entity")
    if not all([customer_id, template, related]):
        return [
            RecoveryAction(
                "escalate_underspecified", "emails_sent", customer_id,
                "assertion does not pin the email's template/reference", applied=False,
            )
        ]
    sent = tools.send_customer_email(conn, clock, customer_id, template, related)
    return [
        RecoveryAction(
            "send_missing_email", "emails_sent", customer_id,
            f"sent '{template}' referencing {related}" if sent.ok else f"failed: {sent.error}",
            applied=sent.ok,
        )
    ]


def _repair_settlements(
    conn: sqlite3.Connection, clock: db.SimClock, result: AssertionResult
) -> list[RecoveryAction]:
    day = result.assertion.where.get("merchant_day")
    expect = result.assertion.expect
    if result.actual_count == 0 and day is not None:
        created = tools.create_settlement(
            conn, clock, day, mark_processed=expect.get("status") == "processed"
        )
        return [
            RecoveryAction(
                "create_missing_settlement", "settlements", day,
                f"created settlement for {day}" if created.ok else f"failed: {created.error}",
                applied=created.ok,
            )
        ]
    actions: list[RecoveryAction] = []
    for row in conn.execute(
        "SELECT * FROM settlements WHERE merchant_day = ? ORDER BY id", (day,)
    ).fetchall():
        for column, want in expect.items():
            if column == "count" or row[column] == want:
                continue
            conn.execute(
                f"UPDATE settlements SET {column} = ? WHERE id = ?",  # noqa: S608
                (want, row["id"]),
            )
            actions.append(
                RecoveryAction(
                    "recompute_settlement", "settlements", row["id"],
                    f"set {column} from {row[column]!r} to {want!r}",
                )
            )
    conn.commit()
    return actions


def _repair_generic(
    conn: sqlite3.Connection, result: AssertionResult
) -> list[RecoveryAction]:
    """Column-level correction for orders, customers, subscriptions, charges.

    Repairs write directly rather than through the tool layer: a repair is an
    operator action, and the legal-transition table constrains the agent, not
    the incident response that is putting state back where it belongs.
    """
    table = result.assertion.table
    where, expect = result.assertion.where, result.assertion.expect
    expected_columns = {k: v for k, v in expect.items() if k != "count"}
    actions: list[RecoveryAction] = []

    if result.actual_count == 0:
        if table != "charges" or "subscription_id" not in where:
            return [
                RecoveryAction(
                    "escalate_unrepairable", table, where,
                    "no row matches the assertion and none can be safely synthesized",
                    applied=False,
                )
            ]
        sub = conn.execute(
            "SELECT * FROM subscriptions WHERE id = ?", (where["subscription_id"],)
        ).fetchone()
        if sub is None:
            return [
                RecoveryAction(
                    "escalate_unrepairable", table, where, "subscription not found",
                    applied=False,
                )
            ]
        conn.execute(
            "INSERT INTO charges (subscription_id, amount_paise, status, attempt_no,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (
                sub["id"],
                expected_columns.get("amount_paise", sub["amount_paise"]),
                expected_columns.get("status", "succeeded"),
                where.get("attempt_no", 1),
                "2026-01-11T00:00:00",
            ),
        )
        conn.commit()
        return [
            RecoveryAction(
                "record_missing_charge", table, sub["id"],
                "recorded the missing charge attempt",
            )
        ]

    clause = " AND ".join(f"{col} = ?" for col in where) or "1 = 1"
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE {clause}",  # noqa: S608 - validated by the verifier
        tuple(where.values()),
    ).fetchall()
    for row in rows:
        for column, want in expected_columns.items():
            if row[column] == want:
                continue
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE id = ?",  # noqa: S608
                (want, row["id"]),
            )
            actions.append(
                RecoveryAction(
                    "correct_state_value", table, row["id"],
                    f"set {column} from {row[column]!r} to {want!r}",
                )
            )
    conn.commit()
    return actions


# ---------------------------------------------------------------------------
# the playbook
# ---------------------------------------------------------------------------


def is_worse(after: VerificationResult, before: VerificationResult) -> bool:
    """Did the repair leave the system in a worse state than it found it?"""
    if after.verdict is Verdict.INDETERMINATE and before.verdict is not Verdict.INDETERMINATE:
        return True
    return (
        after.satisfied_assertions < before.satisfied_assertions
        or len(after.violations) > len(before.violations)
        or len(after.unexpected_changes) > len(before.unexpected_changes)
    )


def render_incident(
    mission: Mission,
    diagnoses: list[Diagnosis],
    actions: list[RecoveryAction],
    verification: VerificationResult,
    rolled_back: bool,
) -> str:
    """Deterministic structured incident report. No LLM involved.

    An LLM may optionally be used downstream to prose-ify this text for a
    human, but never to decide the diagnosis or the actions above.
    """
    lines = [
        f"INCIDENT: mission {mission.mission_id} could not be auto-recovered",
        f"instruction: {mission.instruction}",
        f"verdict after recovery: {verification.verdict}",
        f"rolled back: {rolled_back}",
        "",
        "divergences:",
    ]
    lines += [
        f"  - [{d.kind}] {d.table}: {d.detail}"
        + ("" if d.reversible else "  (IRREVERSIBLE)")
        for d in diagnoses
    ]
    lines.append("")
    lines.append("actions attempted:")
    lines += [
        f"  - {a.action} on {a.table}({a.target}): {a.detail}"
        + ("" if a.applied else "  [NOT APPLIED]")
        for a in actions
    ] or ["  (none)"]
    outstanding = [r.detail for r in verification.assertion_results if not r.satisfied]
    if outstanding:
        lines += ["", "outstanding expected_state failures:"]
        lines += [f"  - {detail}" for detail in outstanding]
    if verification.violations:
        lines += ["", "outstanding invariant violations:"]
        lines += [f"  - {v.rule} on {v.table}:{v.entity_id} — {v.detail}"
                  for v in verification.violations]
    lines += ["", "REQUIRES HUMAN REVIEW: money movement could not be reconciled."]
    return "\n".join(lines)


def recover(mission: Mission, db_dump: str, seed: int) -> RecoveryResult:
    """Diagnose and repair a failed episode on a copy of its final state."""
    pre = verify(mission, db_dump, seed)

    if pre.verdict is Verdict.VERIFIED:
        return RecoveryResult(
            outcome=RecoveryOutcome.NOT_NEEDED, db_dump=db_dump, pre_verification=pre,
            post_verification=pre,
        )
    if pre.verdict is Verdict.INDETERMINATE:
        diagnoses = [
            Diagnosis(DivergenceKind.UNKNOWN, "-", pre.error or "state not readable", False)
        ]
        return RecoveryResult(
            outcome=RecoveryOutcome.ESCALATED,
            diagnoses=diagnoses,
            db_dump=db_dump,
            pre_verification=pre,
            post_verification=pre,
            incident=render_incident(mission, diagnoses, [], pre, rolled_back=False),
        )

    diagnoses = diagnose(pre)
    actions: list[RecoveryAction] = []

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db_dump)
    conn.execute("PRAGMA foreign_keys = ON")
    clock = db.SimClock(start="2026-02-01T00:00:00")
    baseline = _baseline(seed)

    handled: set[tuple[str, int]] = set()
    try:
        # 1. Re-point misapplied writes onto their intended target, so that
        #    references to them survive (see _reassign_stray_refunds).
        actions += _reassign_stray_refunds(conn, pre, handled)

        # 2. Undo the remaining collateral damage, so repairs land on a clean frame.
        for change in pre.unexpected_changes:
            if len(actions) >= MAX_ACTIONS:
                break
            if (change.table, change.row_id) in handled:
                continue
            actions.append(_reverse_change(conn, baseline, change))

        # 3. Repair the intended target, money last.
        failed = [r for r in pre.assertion_results if not r.satisfied]
        for table in REPAIR_ORDER:
            for result in [r for r in failed if r.assertion.table == table]:
                if len(actions) >= MAX_ACTIONS:
                    break
                if table == "refunds":
                    actions += _repair_refunds(conn, clock, result, mission)
                elif table == "emails_sent":
                    actions += _repair_emails(conn, clock, result)
                elif table == "settlements":
                    actions += _repair_settlements(conn, clock, result)
                else:
                    actions += _repair_generic(conn, result)
        repaired_dump = db.dump(conn)
    finally:
        conn.close()
        baseline.close()

    post = verify(mission, repaired_dump, seed)

    # 4. The guarantee: never leave the system worse than we found it.
    if is_worse(post, pre):
        rolled_back_result = RecoveryResult(
            outcome=RecoveryOutcome.ESCALATED,
            diagnoses=diagnoses,
            actions=actions,
            rolled_back=True,
            db_dump=db_dump,          # the original, untouched snapshot
            pre_verification=pre,
            post_verification=pre,    # rollback restores the pre-repair state
        )
        rolled_back_result.incident = render_incident(
            mission, diagnoses, actions, pre, rolled_back=True
        )
        return rolled_back_result

    # 5. Irreversible collateral damage escalates even when the mission's own
    #    assertions now hold: a third party who received a payments email is
    #    real-world damage the expected_state spec cannot see, and reporting
    #    "recovered" there would be exactly the kind of false success this
    #    experiment exists to measure.
    irreversible = [
        a for a in actions if a.action == "escalate_irreversible" and not a.applied
    ]

    if post.verdict is Verdict.VERIFIED and not irreversible:
        return RecoveryResult(
            outcome=RecoveryOutcome.RECOVERED,
            diagnoses=diagnoses,
            actions=actions,
            db_dump=repaired_dump,
            pre_verification=pre,
            post_verification=post,
        )

    return RecoveryResult(
        outcome=RecoveryOutcome.ESCALATED,
        diagnoses=diagnoses,
        actions=actions,
        db_dump=repaired_dump,
        pre_verification=pre,
        post_verification=post,
        incident=render_incident(mission, diagnoses, actions, post, rolled_back=False),
    )
