"""The verification arms — four ways of answering "did the mission succeed?".

Arm A  self-report      : parse the agent's own closing claim.
Arm B  tool-response    : trust the envelopes (200 == success).
Arm C  state verifier   : deterministic check of the real post-episode state.
Arm D  verifier+recovery: Arm C, then a rule-based recovery playbook (M5).

A and B are pure post-hoc analyses of the recorded trajectory and never touch
the database. C reads a post-episode snapshot on its own connection. All
three run over the *same* recorded episode, so the arms are compared on
identical agent behavior rather than on separate runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agent_truth_lab.agent.missions import Mission
from agent_truth_lab.verification.recovery import (
    RecoveryOutcome,
    RecoveryResult,
    recover,
)
from agent_truth_lab.verification.verifier import (
    Verdict,
    VerificationResult,
    verify,
)

COMPLETE_MARKER = re.compile(r"TASK_COMPLETE\b", re.IGNORECASE)
FAILED_MARKER = re.compile(r"TASK_FAILED\b", re.IGNORECASE)

# Fallback heuristic, used only when the agent omits the mandated marker.
# Deliberately conservative and keyword-based (never an LLM): the measurement
# path must stay cheaper and more trustworthy than the thing it audits.
_FAILURE_HINTS = (
    "could not", "couldn't", "unable to", "failed", "was declined", "declined",
    "error", "did not complete", "cannot",
)
_SUCCESS_HINTS = (
    "completed", "successfully", "has been issued", "have been", "done", "sent the",
)


class Outcome(StrEnum):
    """The comparable verdict an arm reports for one episode."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class ArmReport:
    """One arm's verdict on one episode, with the reasoning behind it."""

    arm: str
    outcome: Outcome
    rationale: str
    parse_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "outcome": str(self.outcome),
            "rationale": self.rationale,
            "parse_method": self.parse_method,
        }


def parse_outcome(final_text: str | None) -> tuple[Outcome, str]:
    """Parse the agent's claimed outcome. Returns (outcome, parse_method).

    Primary path is the mandated TASK_COMPLETE/TASK_FAILED marker, searched
    anywhere in the closing message (models often precede it with prose); if
    both appear, the later one wins. When the marker is missing entirely we
    fall back to a keyword heuristic and label it, so the report can state how
    many episodes needed the fallback.
    """
    if final_text is None or not final_text.strip():
        return Outcome.INDETERMINATE, "no_final_message"

    complete = [m.start() for m in COMPLETE_MARKER.finditer(final_text)]
    failed = [m.start() for m in FAILED_MARKER.finditer(final_text)]
    if complete or failed:
        last_complete = complete[-1] if complete else -1
        last_failed = failed[-1] if failed else -1
        outcome = Outcome.SUCCESS if last_complete > last_failed else Outcome.FAILURE
        return outcome, "marker"

    lowered = final_text.lower()
    if any(hint in lowered for hint in _FAILURE_HINTS):
        return Outcome.FAILURE, "heuristic"
    if any(hint in lowered for hint in _SUCCESS_HINTS):
        return Outcome.SUCCESS, "heuristic"
    return Outcome.INDETERMINATE, "unparseable"


def arm_a(record: dict[str, Any]) -> ArmReport:
    """Arm A — believe the agent's own account of what happened."""
    outcome, method = parse_outcome(record.get("final_text"))
    if method == "no_final_message":
        rationale = (
            "episode ended without a closing message"
            f" (stop_reason={record.get('stop_reason')})"
        )
    else:
        rationale = f"claimed outcome parsed via {method}"
    return ArmReport("A", outcome, rationale, parse_method=method)


def arm_b(record: dict[str, Any]) -> ArmReport:
    """Arm B — believe the tool responses; the agent's words are ignored."""
    calls = record.get("tool_calls", [])
    if not calls:
        return ArmReport(
            "B",
            Outcome.FAILURE,
            "no tool call returned a result"
            f" (stop_reason={record.get('stop_reason')})",
        )
    failures = [c for c in calls if not c["result"]["ok"]]
    if failures:
        statuses = ", ".join(str(c["result"]["http_status"]) for c in failures)
        return ArmReport(
            "B", Outcome.FAILURE, f"{len(failures)}/{len(calls)} calls not ok ({statuses})"
        )
    return ArmReport("B", Outcome.SUCCESS, f"all {len(calls)} tool calls returned ok/200")


VERDICT_TO_OUTCOME = {
    Verdict.VERIFIED: Outcome.SUCCESS,
    Verdict.FAILED: Outcome.FAILURE,
    Verdict.INDETERMINATE: Outcome.INDETERMINATE,
}


def arm_c(result: VerificationResult) -> ArmReport:
    """Arm C — believe only the state, checked deterministically."""
    if result.verdict is Verdict.VERIFIED:
        rationale = f"all {len(result.assertion_results)} assertions hold; no violations"
    elif result.verdict is Verdict.INDETERMINATE:
        rationale = result.error or "verification could not be completed"
    else:
        failed = [r.detail for r in result.assertion_results if not r.satisfied]
        broken = [f"{v.rule}({v.table}:{v.entity_id})" for v in result.violations]
        parts = []
        if failed:
            parts.append(f"{len(failed)} assertion(s) failed: {'; '.join(failed)}")
        if broken:
            parts.append(f"invariants breached: {', '.join(broken)}")
        rationale = " | ".join(parts)
    return ArmReport("C", VERDICT_TO_OUTCOME[result.verdict], rationale)


def arm_d(mission: Mission, record: dict[str, Any]) -> tuple[ArmReport, RecoveryResult]:
    """Arm D — verify, then attempt rule-based recovery, then verify again.

    Runs on a copy of the episode snapshot, so Arm C's measurement of the
    same episode is never contaminated. D reports SUCCESS only when the state
    is verified *after* recovery.
    """
    result = recover(mission, record["db_dump"], record["seed"])
    final = result.post_verification

    # D reports SUCCESS only when the playbook actually closed the incident.
    # An escalation is a FAILURE report even if the mission's own assertions
    # now hold — recovery escalates on damage the spec cannot see, and this
    # arm is deliberately conservative about claiming success.
    if final is None or final.verdict is Verdict.INDETERMINATE:
        outcome = Outcome.INDETERMINATE
    elif result.outcome in (RecoveryOutcome.RECOVERED, RecoveryOutcome.NOT_NEEDED):
        outcome = Outcome.SUCCESS
    else:
        outcome = Outcome.FAILURE

    if result.outcome is RecoveryOutcome.NOT_NEEDED:
        rationale = "state already correct; no recovery needed"
    elif result.recovered:
        applied = [a.action for a in result.actions if a.applied]
        rationale = f"auto-recovered via {len(applied)} action(s): {', '.join(applied)}"
    else:
        kinds = sorted({str(d.kind) for d in result.diagnoses})
        rationale = (
            f"escalated after {len(result.actions)} action(s);"
            f" divergences: {', '.join(kinds)}"
            + (" (rolled back — repair would have worsened state)" if result.rolled_back
               else "")
        )
    return ArmReport("D", outcome, rationale), result


@dataclass
class EpisodeEvaluation:
    """One episode judged by every arm, plus the ground-truth verification."""

    mission_id: str
    seed: int
    failure_mode: str
    verification: VerificationResult
    reports: dict[str, ArmReport]
    recovery: RecoveryResult | None = None
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    episode_latency_seconds: float = 0.0
    crashed: bool = False
    # The episode ended because the API call failed, not because of anything
    # the experiment was testing.
    api_error: bool = False

    @property
    def ground_truth_satisfied(self) -> bool | None:
        """True/False from the verifier; None when it genuinely cannot decide.

        Note the deliberate circularity: Arm C *is* the ground-truth evaluator,
        so Arm C's false-success rate is zero by construction. That is a
        property of the design, not an empirical finding, and the report says
        so. Arm C's measurable cost is its INDETERMINATE rate.
        """
        if self.verification.verdict is Verdict.INDETERMINATE:
            return None
        return self.verification.verdict is Verdict.VERIFIED

    def ground_truth_for(self, arm: str) -> bool | None:
        """Ground truth as it stands for a given arm.

        Arms A-C are scored against the episode's final state. Arm D *changes*
        that state, so it is scored against the post-recovery state — scoring
        D against the pre-recovery state would count every successful repair
        as a false success.
        """
        if arm != "D" or self.recovery is None:
            return self.ground_truth_satisfied
        final = self.recovery.post_verification
        if final is None or final.verdict is Verdict.INDETERMINATE:
            return None
        return final.verdict is Verdict.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "seed": self.seed,
            "failure_mode": self.failure_mode,
            "ground_truth_satisfied": self.ground_truth_satisfied,
            "verification": self.verification.to_dict(),
            "reports": {arm: r.to_dict() for arm, r in self.reports.items()},
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "usage_input_tokens": self.usage_input_tokens,
            "usage_output_tokens": self.usage_output_tokens,
            "episode_latency_seconds": self.episode_latency_seconds,
            "crashed": self.crashed,
        }


def failure_mode_of(mission: Mission) -> str:
    """The mission's injected mode label, or 'clean' when nothing is injected."""
    if not mission.injection:
        return "clean"
    return sorted(mission.injection.values())[0]


def evaluate_episode(
    record: dict[str, Any], mission: Mission, include_recovery: bool = True
) -> EpisodeEvaluation:
    """Run every arm over one recorded episode.

    Arm C reads the pristine snapshot; Arm D recovers on a copy of it, so the
    two measurements stay independent even though they share an episode.
    """
    result = verify(mission, record["db_dump"], record["seed"])
    reports = {"A": arm_a(record), "B": arm_b(record), "C": arm_c(result)}
    recovery = None
    if include_recovery:
        reports["D"], recovery = arm_d(mission, record)
    return EpisodeEvaluation(
        mission_id=mission.mission_id,
        seed=record["seed"],
        failure_mode=failure_mode_of(mission),
        verification=result,
        reports=reports,
        recovery=recovery,
        usage_input_tokens=record.get("usage_input_tokens", 0),
        usage_output_tokens=record.get("usage_output_tokens", 0),
        episode_latency_seconds=record.get("latency_seconds", 0.0),
        crashed=record.get("crashed", False),
        api_error=record.get("stop_reason") == "api_error",
    )
