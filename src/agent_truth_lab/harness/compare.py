"""Model comparison: does a more capable agent overclaim less?

Takes two or more run summaries and renders the false success rate per arm,
grouped by model. This is the chart that answers the question the single-model
result raises but cannot settle.

Usage:
    atl-compare                                   # every summary in results/
    atl-compare --summaries a.json b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent_truth_lab.harness.report import (  # noqa: E402
    ARM_LABELS,
    DARK,
    LIGHT,
    Theme,
    _style,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

# Categorical slots 1 and 2 of the validated palette, per model.
MODEL_COLORS_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
MODEL_COLORS_DARK = ("#3987e5", "#d95926", "#199e70")


def load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        summaries.append(data)
    # Cheapest model first, so the chart reads in increasing capability.
    order = {"claude-haiku-4-5": 0, "claude-sonnet-5": 1, "claude-opus-5": 2}
    summaries.sort(key=lambda s: order.get(s["config"]["model"], 99))
    return summaries


def comparison_chart(
    summaries: list[dict[str, Any]], theme: Theme, out_path: Path
) -> Path:
    """Grouped bars: false success rate per arm, one group of bars per model."""
    arms = [a for a in ("A", "B", "C", "D") if a in summaries[0]["summary"]["arms"]]
    colors = MODEL_COLORS_LIGHT if theme.name == "light" else MODEL_COLORS_DARK

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=theme.surface)
    group_height = 0.78
    bar_height = group_height / len(summaries)

    for index, summary in enumerate(summaries):
        model = summary["config"]["model"]
        stats = summary["summary"]["arms"]
        means = [stats[a]["false_success_rate"]["mean"] for a in arms]
        lows = [stats[a]["false_success_rate"]["min"] for a in arms]
        highs = [stats[a]["false_success_rate"]["max"] for a in arms]
        offsets = [
            i + (index - (len(summaries) - 1) / 2) * bar_height for i in range(len(arms))
        ]
        errors = [
            [max(0.0, m - lo) for m, lo in zip(means, lows, strict=True)],
            [max(0.0, hi - m) for m, hi in zip(means, highs, strict=True)],
        ]
        ax.barh(
            offsets, means, height=bar_height * 0.86, color=colors[index % len(colors)],
            label=model, xerr=errors,
            error_kw={"ecolor": theme.text_secondary, "elinewidth": 1.1,
                      "capsize": 3, "capthick": 1.1},
        )
        for offset, mean, high in zip(offsets, means, highs, strict=True):
            ax.text(
                max(mean, high) + 0.022, offset, f"{mean:.0%}",
                va="center", ha="left", fontsize=9.5, fontweight="bold",
                color=theme.text_primary,
            )

    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARM_LABELS[a] for a in arms], fontsize=10,
                       color=theme.text_primary)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="x", color=theme.grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    _style(ax, theme)

    seeds = summaries[0]["summary"]["seeds"]
    ax.set_title(
        "Does a more capable agent overclaim less?",
        fontsize=14, fontweight="bold", color=theme.text_primary, loc="left", pad=32,
    )
    ax.text(
        0, 1.035,
        "False success rate by arm and model"
        f" · {len(seeds)} seeds with randomized task mix, mean with range",
        transform=ax.transAxes, fontsize=9.5, color=theme.text_secondary,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=theme.text_secondary)

    fig.subplots_adjust(left=0.24, right=0.97, top=0.80, bottom=0.12)
    fig.savefig(out_path, dpi=200, facecolor=theme.surface)
    plt.close(fig)
    return out_path


def comparison_table(summaries: list[dict[str, Any]]) -> str:
    """Markdown table of the same numbers."""
    arms = [a for a in ("A", "B", "C", "D") if a in summaries[0]["summary"]["arms"]]
    models = [s["config"]["model"] for s in summaries]
    lines = [
        "| Arm | " + " | ".join(f"`{m}`" for m in models) + " |",
        "|-----|" + "|".join(["---"] * len(models)) + "|",
    ]
    for arm in arms:
        cells = []
        for summary in summaries:
            stat = summary["summary"]["arms"][arm]["false_success_rate"]
            cells.append(f"{stat['mean']:.1%} [{stat['min']:.1%}, {stat['max']:.1%}]")
        lines.append(f"| {ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    arms = [a for a in ("A", "B", "C", "D") if a in summaries[0]["summary"]["arms"]]
    print("=" * 74)
    print("MODEL COMPARISON - false success rate (mean [min, max] across seeds)")
    print("=" * 74)
    header = f"  {'arm':<28}" + "".join(
        f"{s['config']['model']:>22}" for s in summaries
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm in arms:
        row = f"  {ARM_LABELS[arm]:<28}"
        for summary in summaries:
            stat = summary["summary"]["arms"][arm]["false_success_rate"]
            row += f"{stat['mean']:>21.1%} "
        print(row)
    print()
    for summary in summaries:
        recovery = summary["summary"].get("recovery") or []
        needed = sum(r["needed"] for r in recovery)
        recovered = sum(r["recovered"] for r in recovery)
        state = summary["summary"].get("state") or []
        violated = sum(s["ground_truth_violated"] for s in state)
        episodes = sum(s["episodes"] for s in state)
        rate = recovered / needed if needed else 0.0
        print(
            f"  {summary['config']['model']:<22}"
            f" episodes damaged {violated}/{episodes},"
            f" auto-recovered {recovered}/{needed} ({rate:.0%})"
        )
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare runs across models.")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--summaries", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, help="output PNG (light variant)")
    args = parser.parse_args(argv)

    paths = args.summaries or sorted(args.results.glob("summary_*.json"))
    if len(paths) < 2:
        print(
            f"need at least two summaries to compare, found {len(paths)} in"
            f" {args.results}"
        )
        return 1

    summaries = load_summaries(list(paths))
    print_comparison(summaries)

    base = args.out or args.results / "model_comparison.png"
    for theme in (LIGHT, DARK):
        target = base if theme.name == "light" else base.with_name(
            f"{base.stem}_dark{base.suffix}"
        )
        comparison_chart(summaries, theme, target)
        print(f"  wrote {target.name}")

    table_path = args.results / "model_comparison.md"
    table_path.write_text(comparison_table(summaries) + "\n", encoding="utf-8")
    print(f"  wrote {table_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
