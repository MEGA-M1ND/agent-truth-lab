"""The experiment runner: missions x seeds, evaluated by every arm.

Reproducibility contract — a run is fully determined by:
  * the config file (model, seeds, turn/token limits, arms),
  * the seeds, which fix the environment, the mission set, and the injection
    plan (injection itself is a pure function of DB state and call order), and
  * the recorded episodes, which make every downstream measurement replayable
    even though the LLM itself is the one stochastic component.

Every run writes the full episode records (trajectories, tool envelopes, and
the post-episode database dump) so the arms can be re-scored later without
re-running the agent.

Usage:
    python -m agent_truth_lab.harness.runner                  # full run
    python -m agent_truth_lab.harness.runner --estimate-only  # cost estimate
    python -m agent_truth_lab.harness.runner --limit 4 --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agent_truth_lab.agent import loop, missions
from agent_truth_lab.environment import db
from agent_truth_lab.harness import metrics
from agent_truth_lab.verification import arms

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment.yaml"
RESULTS_DIR = REPO_ROOT / "results"

# USD per million tokens (input, output). Used only for the cost estimate the
# runner prints before spending anything.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}
FALLBACK_PRICE = (5.00, 25.00)

# Measured on the M3 smoke run (Haiku 4.5, 2 clean missions): tokens per
# episode. Input dominates because the tool schemas and system prompt are
# resent every turn.
TOKENS_PER_EPISODE = (4100, 260)
# With read tools: 4 extra tool schemas resent every turn, plus occasional
# extra turns for a check-then-act pattern. Conservative bump pending a
# measured average.
TOKENS_PER_EPISODE_READ_TOOLS = (5200, 340)


@dataclass
class RunConfig:
    """Everything that determines a run, loaded from config/experiment.yaml."""

    model: str
    seeds: list[int]
    max_turns: int
    max_tokens: int
    temperature: float | None
    arms: list[str]
    structural_variance: bool = False
    models: list[str] = field(default_factory=list)
    read_tools: bool = False
    resolve_conflicts: bool = False

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # `models` (a list) is the multi-model form; `model` remains supported
        # so older configs keep working.
        models = list(data.get("models") or ([data["model"]] if data.get("model") else []))
        if not models:
            raise ValueError("config must set either 'model' or 'models'")
        return cls(
            model=models[0],
            models=models,
            seeds=list(data["seeds"]),
            max_turns=int(data["max_turns"]),
            max_tokens=int(data["max_tokens"]),
            temperature=data.get("temperature"),
            arms=list(data.get("arms", ["A", "B", "C", "D"])),
            structural_variance=bool(data.get("structural_variance", False)),
            read_tools=bool(data.get("read_tools", False)),
            resolve_conflicts=bool(data.get("resolve_conflicts", False)),
        )

    def for_model(self, model: str) -> RunConfig:
        """A copy pinned to one model."""
        return replace(self, model=model, models=[model])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "models": self.models,
            "seeds": self.seeds,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "arms": self.arms,
            "structural_variance": self.structural_variance,
            "read_tools": self.read_tools,
            "resolve_conflicts": self.resolve_conflicts,
        }


def model_slug(model: str) -> str:
    """Filename-safe form of a model id."""
    return model.replace(".", "-").replace("/", "-")


def build_mission_set(
    seed: int, structural_variance: bool = False
) -> list[missions.Mission]:
    """The mission set for a seed — deterministic from the seeded environment."""
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, seed)
    try:
        return missions.build_missions(
            conn, seed=seed, structural_variance=structural_variance
        )
    finally:
        conn.close()


def estimate_cost(
    model: str, episodes: int, read_tools: bool = False
) -> tuple[float, int, int]:
    """(usd, input_tokens, output_tokens) projected for a run of `episodes`."""
    price_in, price_out = PRICES.get(model, FALLBACK_PRICE)
    per_episode = TOKENS_PER_EPISODE_READ_TOOLS if read_tools else TOKENS_PER_EPISODE
    total_in = per_episode[0] * episodes
    total_out = per_episode[1] * episodes
    usd = total_in / 1e6 * price_in + total_out / 1e6 * price_out
    return usd, total_in, total_out


def print_plan(config: RunConfig, mission_count: int) -> float:
    """Print the run plan and its cost estimate before anything is spent."""
    episodes_per_model = mission_count * len(config.seeds)
    total = 0.0
    print("=" * 68)
    print("AgentTruthLab - experiment plan")
    print("=" * 68)
    print(f"  models         : {', '.join(config.models)}")
    print(f"  seeds          : {config.seeds}")
    print(f"  missions/seed  : {mission_count}")
    print(f"  episodes       : {episodes_per_model} per model,"
          f" {episodes_per_model * len(config.models)} total")
    print(f"  mission set    : "
          f"{'structural variance (from seed)' if config.structural_variance else 'fixed'}")
    print(f"  arms           : {', '.join(config.arms)}")
    read_tools_label = "off"
    if config.read_tools:
        read_tools_label = (
            "on + trust-the-read-on-conflict instruction"
            if config.resolve_conflicts else "on (agent can self-verify)"
        )
    print(f"  read tools     : {read_tools_label}")
    print(f"  max turns      : {config.max_turns}, max tokens: {config.max_tokens}")
    for model in config.models:
        usd, tokens_in, tokens_out = estimate_cost(
            model, episodes_per_model, config.read_tools
        )
        total += usd
        print(
            f"    {model:<22} ~{tokens_in:,} in / ~{tokens_out:,} out"
            f"  ~${usd:.2f}"
        )
    print(f"  EST. API COST  : ~${total:.2f} (from measured per-episode averages)")
    print("=" * 68)
    return total


def run_seed(
    config: RunConfig,
    seed: int,
    client: Any,
    limit: int | None = None,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], list[arms.EpisodeEvaluation]]:
    """Run every mission for one seed and evaluate each episode with all arms."""
    mission_set = build_mission_set(seed, config.structural_variance)
    if limit is not None:
        mission_set = mission_set[:limit]

    records: list[dict[str, Any]] = []
    evaluations: list[arms.EpisodeEvaluation] = []
    include_recovery = "D" in config.arms

    for index, mission in enumerate(mission_set, start=1):
        record = loop.run_episode(
            mission,
            seed,
            config.model,
            client,
            max_turns=config.max_turns,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            include_read_tools=config.read_tools,
            resolve_conflicts=config.resolve_conflicts,
        )
        evaluation = arms.evaluate_episode(
            record.to_dict(), mission, include_recovery=include_recovery
        )
        records.append(record.to_dict())
        evaluations.append(evaluation)

        if progress:
            verdicts = " ".join(
                f"{arm}={str(evaluation.reports[arm].outcome)[:4]}"
                for arm in sorted(evaluation.reports)
            )
            print(
                f"  [{index:>2}/{len(mission_set)}] {mission.mission_id:<34}"
                f" {evaluation.failure_mode:<24} {verdicts}"
            )
    return records, evaluations


def write_run_output(
    out_dir: Path,
    stamp: str,
    seed: int,
    config: RunConfig,
    records: list[dict[str, Any]],
    evaluations: list[arms.EpisodeEvaluation],
    run_metrics: metrics.RunMetrics,
) -> Path:
    """Write the full per-seed record, including every trajectory and snapshot."""
    path = out_dir / f"run_{stamp}_{seed}.json"
    payload = {
        "stamp": stamp,
        "seed": seed,
        "config": config.to_dict(),
        "metrics": run_metrics.to_dict(),
        "episodes": [
            {"record": record, "evaluation": evaluation.to_dict()}
            for record, evaluation in zip(records, evaluations, strict=True)
        ],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def write_aggregate_csv(out_dir: Path, stamp: str, runs: list[metrics.RunMetrics]) -> Path:
    """Flat per-seed, per-arm CSV — the table behind the headline chart."""
    path = out_dir / f"aggregate_{stamp}.csv"
    rows = [row for run in runs for row in run.headline_rows()]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize(runs: list[metrics.RunMetrics]) -> dict[str, Any]:
    """Combine per-seed runs into the reported mean-and-range summary."""
    combined = metrics.aggregate_runs(runs)
    combined["state"] = [run.state.to_dict() for run in runs if run.state]
    combined["cost"] = [run.cost.to_dict() for run in runs if run.cost]
    combined["recovery"] = [run.recovery.to_dict() for run in runs if run.recovery]
    return combined


def print_summary(runs: list[metrics.RunMetrics], combined: dict[str, Any]) -> None:
    """Print the headline table."""
    print()
    print("=" * 68)
    print("RESULTS - false success rate by arm (mean [min, max] across seeds)")
    print("=" * 68)
    labels = {
        "A": "A  agent self-report",
        "B": "B  tool responses (200 = success)",
        "C": "C  independent state verifier",
        "D": "D  verifier + recovery",
    }
    for arm, stats in combined["arms"].items():
        fsr = stats["false_success_rate"]
        ffr = stats["false_failure_rate"]
        print(
            f"  {labels.get(arm, arm):<36}"
            f" FSR {fsr['mean']:.3f} [{fsr['min']:.3f}, {fsr['max']:.3f}]"
            f"   FFR {ffr['mean']:.3f}"
        )
    api_errors = sum(s.get("api_errors", 0) for s in combined.get("state", []))
    if api_errors:
        episodes = sum(s["episodes"] for s in combined.get("state", []))
        print(
            f"\n  WARNING: {api_errors}/{episodes} episodes ended in an API error"
            " (connection drop / rate limit)."
        )
        print(
            "  These are infrastructure noise, not evidence about agent"
            " truthfulness — re-run the affected seeds before publishing."
        )
    if combined["recovery"]:
        totals = combined["recovery"]
        needed = sum(r["needed"] for r in totals)
        recovered = sum(r["recovered"] for r in totals)
        damaged = sum(r["damaged"] for r in totals)
        rate = recovered / needed if needed else 0.0
        print(
            f"\n  Arm D recovery: {recovered}/{needed} auto-recovered ({rate:.1%});"
            f" recovery-induced damage: {damaged}"
        )
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentTruthLab experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="*", help="override config seeds")
    parser.add_argument("--model", help="run only this model, overriding the config")
    parser.add_argument(
        "--read-tools", action="store_true",
        help="give the agent read-only tools (get_order, get_refund, ...) so it can"
        " attempt to verify its own writes, overriding the config",
    )
    parser.add_argument(
        "--resolve-conflicts", action="store_true",
        help="also instruct the agent to trust a read over an earlier write's claimed"
        " success when they conflict (M9); implies --read-tools",
    )
    parser.add_argument("--limit", type=int, help="run only the first N missions per seed")
    parser.add_argument(
        "--estimate-only", action="store_true", help="print the cost estimate and exit"
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR, help="directory for run outputs"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="render charts and findings.md as soon as the run finishes",
    )
    args = parser.parse_args(argv)

    config = RunConfig.load(args.config)
    if args.seeds:
        config.seeds = args.seeds
    if args.model:
        config = config.for_model(args.model)
    if args.read_tools:
        config = replace(config, read_tools=True)
    if args.resolve_conflicts:
        config = replace(config, read_tools=True, resolve_conflicts=True)

    mission_count = len(build_mission_set(config.seeds[0], config.structural_variance))
    if args.limit is not None:
        mission_count = min(mission_count, args.limit)
    print_plan(config, mission_count)

    if args.estimate_only:
        return 0

    import anthropic  # imported here so --estimate-only works without the SDK

    client = anthropic.Anthropic()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = time.monotonic()
    summary_paths: list[Path] = []

    for model in config.models:
        model_config = config.for_model(model)
        slug = model_slug(model)
        tag = f"{stamp}_{slug}" if len(config.models) > 1 else stamp
        print(f"\n{'=' * 68}\nMODEL: {model}\n{'=' * 68}")

        runs: list[metrics.RunMetrics] = []
        for seed in model_config.seeds:
            print(f"\n--- seed {seed} ---")
            records, evaluations = run_seed(model_config, seed, client, limit=args.limit)
            run_metrics = metrics.compute_run_metrics(
                evaluations, seed=seed, arms=tuple(model_config.arms)
            )
            runs.append(run_metrics)
            path = write_run_output(
                args.out, tag, seed, model_config, records, evaluations, run_metrics
            )
            print(f"  wrote {path.name}")

        combined = summarize(runs)
        csv_path = write_aggregate_csv(args.out, tag, runs)
        summary_path = args.out / f"summary_{tag}.json"
        summary_path.write_text(
            json.dumps(
                {"config": model_config.to_dict(), "summary": combined}, indent=1
            ),
            encoding="utf-8",
        )
        summary_paths.append(summary_path)

        print_summary(runs, combined)
        actual_in = sum(run.cost.total_input_tokens for run in runs if run.cost)
        actual_out = sum(run.cost.total_output_tokens for run in runs if run.cost)
        price_in, price_out = PRICES.get(model, FALLBACK_PRICE)
        actual_usd = actual_in / 1e6 * price_in + actual_out / 1e6 * price_out
        print(
            f"  {model}: {actual_in:,} in / {actual_out:,} out"
            f"  -  actual cost ${actual_usd:.3f}"
        )
        print(f"  outputs: {csv_path.name}, {summary_path.name}")

    print(f"\nwall clock: {time.monotonic() - started:.0f}s")

    if args.report:
        from agent_truth_lab.harness import report

        print()
        multi = len(summary_paths) > 1
        for summary_path, model in zip(summary_paths, config.models, strict=True):
            # Without a per-model prefix each render would overwrite the last,
            # leaving charts that silently belong to whichever model ran last.
            report_args = ["--summary", str(summary_path), "--results", str(args.out)]
            if multi:
                slug = model_slug(model)
                report_args += [
                    "--prefix", f"{slug}_",
                    "--findings", str(args.out / f"findings_{slug}.md"),
                ]
            report.main(report_args)
    else:
        print("\nNext: python -m agent_truth_lab.harness.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
