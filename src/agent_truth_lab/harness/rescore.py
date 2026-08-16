"""Offline re-scoring of stored runs — sensitivity analysis, zero API cost.

Every episode record carries the mission spec, the full trajectory, and a dump
of the database as it stood when the episode ended. Verification is pure and
deterministic, so an entire run can be re-judged from disk without calling the
model again. This module uses that to answer the obvious reviewer question:

    "Your headline number depends on how you defined 'correct state'.
     How much does it move if you define it the other way?"

It scores the same recorded episodes against both ground-truth definitions:

  frame-scoped  the mission's assertions + global invariants (the default)
  strict        the above, plus: any write outside the mission's declared
                frame is damage

The interesting cell is Arm C — as implemented it is frame-scoped, so scoring
it against *strict* ground truth is not circular and exposes exactly how much
the verifier misses.

Usage:
    atl-rescore                       # newest run set in results/
    atl-rescore --stamp 20260816_230829
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_truth_lab.agent.missions import Mission
from agent_truth_lab.harness import metrics
from agent_truth_lab.verification import arms
from agent_truth_lab.verification.arms import ArmReport, EpisodeEvaluation
from agent_truth_lab.verification.verifier import Verdict, VerificationResult, verify

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

GROUND_TRUTHS = ("frame-scoped", "strict")


@dataclass
class RescoredEpisode:
    """One stored episode re-judged under both ground-truth definitions."""

    mission_id: str
    seed: int
    failure_mode: str
    reports: dict[str, ArmReport]
    frame: VerificationResult
    strict: VerificationResult

    @property
    def flipped(self) -> bool:
        """Passed the frame-scoped check but fails the strict one."""
        return (
            self.frame.verdict is Verdict.VERIFIED
            and self.strict.verdict is Verdict.FAILED
        )

    def as_evaluation(self, ground_truth: str) -> EpisodeEvaluation:
        """The same arm reports, scored against the chosen ground truth."""
        return EpisodeEvaluation(
            mission_id=self.mission_id,
            seed=self.seed,
            failure_mode=self.failure_mode,
            verification=self.frame if ground_truth == "frame-scoped" else self.strict,
            reports=self.reports,
        )


def find_stamps(results_dir: Path) -> list[str]:
    stamps = {
        path.stem.rsplit("_", 1)[0].replace("run_", "")
        for path in results_dir.glob("run_*.json")
    }
    return sorted(stamps)


def rescore_run(results_dir: Path, stamp: str) -> list[RescoredEpisode]:
    """Replay every stored episode for a run stamp under both definitions."""
    files = sorted(results_dir.glob(f"run_{stamp}_*.json"))
    if not files:
        raise FileNotFoundError(f"no run_{stamp}_*.json in {results_dir}")

    rescored: list[RescoredEpisode] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for episode in data["episodes"]:
            record = episode["record"]
            mission = Mission.from_dict(record["mission"])
            seed, dump = record["seed"], record["db_dump"]

            frame = verify(mission, dump, seed)
            strict = verify(mission, dump, seed, strict_frame=True)
            rescored.append(
                RescoredEpisode(
                    mission_id=mission.mission_id,
                    seed=seed,
                    failure_mode=arms.failure_mode_of(mission),
                    # A and B read the trajectory; C is the frame-scoped verifier
                    # exactly as the experiment implements it.
                    reports={
                        "A": arms.arm_a(record),
                        "B": arms.arm_b(record),
                        "C": arms.arm_c(frame),
                    },
                    frame=frame,
                    strict=strict,
                )
            )
    return rescored


def build_table(rescored: list[RescoredEpisode]) -> dict[str, Any]:
    """False success rate per arm under each ground-truth definition."""
    table: dict[str, Any] = {"episodes": len(rescored), "ground_truths": {}}
    for ground_truth in GROUND_TRUTHS:
        evaluations = [e.as_evaluation(ground_truth) for e in rescored]
        violated = sum(1 for e in evaluations if e.ground_truth_satisfied is False)
        table["ground_truths"][ground_truth] = {
            "episodes_violated": violated,
            "arms": {
                arm: metrics.compute_arm_metrics(evaluations, arm).to_dict()
                for arm in ("A", "B", "C")
            },
        }
    flipped = [e for e in rescored if e.flipped]
    table["flipped"] = {
        "count": len(flipped),
        "rate": len(flipped) / len(rescored) if rescored else 0.0,
        "episodes": [
            {
                "mission_id": e.mission_id,
                "seed": e.seed,
                "failure_mode": e.failure_mode,
                "damage": [c.to_dict() for c in e.frame.unexpected_changes],
            }
            for e in flipped
        ],
    }
    return table


def print_table(table: dict[str, Any]) -> None:
    episodes = table["episodes"]
    print("=" * 74)
    print("SENSITIVITY - false success rate under each ground-truth definition")
    print("=" * 74)
    print(f"  episodes re-scored offline: {episodes} (no API calls)")
    print()
    header = f"  {'arm':<26}" + "".join(f"{gt:>22}" for gt in GROUND_TRUTHS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    labels = {
        "A": "A  agent self-report",
        "B": "B  tool responses",
        "C": "C  verifier (frame-scoped)",
    }
    for arm in ("A", "B", "C"):
        row = f"  {labels[arm]:<26}"
        for ground_truth in GROUND_TRUTHS:
            stats = table["ground_truths"][ground_truth]["arms"][arm]
            row += f"{stats['false_success_rate']:>21.1%} "
        print(row)
    print()
    for ground_truth in GROUND_TRUTHS:
        violated = table["ground_truths"][ground_truth]["episodes_violated"]
        print(f"  episodes counted as violated ({ground_truth}): {violated}/{episodes}")
    flipped = table["flipped"]
    print(
        f"\n  episodes that flip VERIFIED -> FAILED under the strict definition:"
        f" {flipped['count']} ({flipped['rate']:.1%})"
    )
    for episode in flipped["episodes"][:6]:
        damage = episode["damage"][0]["detail"] if episode["damage"] else "-"
        print(
            f"    seed {episode['seed']} {episode['mission_id']}"
            f" [{episode['failure_mode']}]: {damage[:60]}"
        )
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score stored runs under both ground-truth definitions."
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--stamp", help="run stamp; defaults to the newest")
    parser.add_argument("--out", type=Path, help="write the comparison as JSON")
    args = parser.parse_args(argv)

    stamp = args.stamp
    if stamp is None:
        stamps = find_stamps(args.results)
        if not stamps:
            print(f"no run_*.json found in {args.results}")
            return 1
        stamp = stamps[-1]

    print(f"re-scoring run {stamp}")
    rescored = rescore_run(args.results, stamp)
    table = build_table(rescored)
    print_table(table)

    out_path = args.out or args.results / f"sensitivity_{stamp}.json"
    out_path.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"  wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
