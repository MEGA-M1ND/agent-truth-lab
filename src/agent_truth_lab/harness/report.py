"""Chart and findings generation from a run summary.

Produces:
  results/headline_false_success_rate{,_dark}.png  — the headline bar chart
  results/heatmap_arm_by_mode{,_dark}.png          — arm x failure mode
  findings.md                                      — numbers filled in

Chart design notes (the palette is validated, not eyeballed):
- Arms are colored by *what they trust*, not by how well they scored — blue for
  the observability arms (A, B: the agent's own signals) and orange for the
  assurance arms (C, D: independent state). Color follows the entity, never its
  rank, so the bars would keep their hues if the numbers reversed.
- The heatmap is a single-hue sequential ramp (light -> dark = low -> high),
  never a rainbow, because it encodes magnitude.
- Both themes are rendered from steps chosen for their own surface rather than
  by flipping the light one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

ARM_LABELS = {
    "A": "A — agent self-report",
    "B": "B — tool responses",
    "C": "C — state verifier",
    "D": "D — verifier + recovery",
}
# Which signal each arm trusts. This is the categorical distinction the color
# encodes; it is a property of the method, not of the result.
ARM_GROUP = {"A": "observability", "B": "observability", "C": "assurance", "D": "assurance"}

MODE_LABELS = {
    "clean": "clean\n(no injection)",
    "silent_noop": "F1\nsilent noop",
    "wrong_target": "F2\nwrong target",
    "timeout_then_duplicate": "F3\ntimeout+dup",
    "partial_completion": "F4\npartial",
    "stale_read": "F5\nlost write",
    "invariant_violation": "F6\ninvariant",
    "crash_after_side_effect": "F7\ncrash",
}
MODE_ORDER = [
    "clean", "silent_noop", "wrong_target", "timeout_then_duplicate",
    "partial_completion", "stale_read", "invariant_violation",
    "crash_after_side_effect",
]


@dataclass(frozen=True)
class Theme:
    """Surface, ink, and series colors stepped for one background."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    observability: str
    assurance: str
    ramp: tuple[str, ...]

    @property
    def suffix(self) -> str:
        return "" if self.name == "light" else "_dark"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e3e3e0",
    observability="#2a78d6",
    assurance="#eb6834",
    # blue 100 -> 700, light means near zero and may recede toward the surface
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#3a3a38",
    observability="#3987e5",
    assurance="#d95926",
    # reversed for the dark surface: the near-zero step is the one nearest the
    # background, so magnitude still reads as distance from the surface
    ramp=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def newest_summary(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("summary_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no summary_*.json in {results_dir} — run the harness first:"
            " python -m agent_truth_lab.harness.runner"
        )
    return candidates[-1]


def _style(ax, theme: Theme) -> None:
    """Recessive axes and grid; the data carries the ink."""
    ax.set_facecolor(theme.surface)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid)
    ax.tick_params(colors=theme.text_secondary, length=0, labelsize=9)


def headline_chart(summary: dict[str, Any], theme: Theme, out_path: Path) -> Path:
    """False success rate by arm, mean with the across-seed range."""
    arms_data = summary["summary"]["arms"]
    arms = [a for a in ("A", "B", "C", "D") if a in arms_data]
    means = [arms_data[a]["false_success_rate"]["mean"] for a in arms]
    lows = [arms_data[a]["false_success_rate"]["min"] for a in arms]
    highs = [arms_data[a]["false_success_rate"]["max"] for a in arms]
    colors = [
        theme.observability if ARM_GROUP[a] == "observability" else theme.assurance
        for a in arms
    ]

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=theme.surface)
    positions = range(len(arms))
    # Clamp at zero: when every seed agrees, fmean can land a hair below the
    # min, and matplotlib rejects a negative error bar.
    errors = [
        [max(0.0, m - lo) for m, lo in zip(means, lows, strict=True)],
        [max(0.0, hi - m) for m, hi in zip(means, highs, strict=True)],
    ]
    bars = ax.barh(
        list(positions), means, height=0.58, color=colors,
        xerr=errors, error_kw={"ecolor": theme.text_secondary, "elinewidth": 1.2,
                               "capsize": 4, "capthick": 1.2},
    )
    for bar in bars:  # 4px rounded data-end, anchored at the baseline
        bar.set_joinstyle("round")

    for pos, mean, high in zip(positions, means, highs, strict=True):
        ax.text(
            max(high, mean) + 0.028, pos, f"{mean:.0%}",
            va="center", ha="left", fontsize=11, fontweight="bold",
            color=theme.text_primary,
        )

    # A swatch beside every arm label. Arms C and D sit at exactly 0%, so their
    # bars have no extent to carry the hue — without this the legend would
    # advertise a color the chart never shows.
    for pos, color in zip(positions, colors, strict=True):
        ax.plot(
            -0.30, pos, marker="s", markersize=9, color=color,
            clip_on=False, linestyle="", transform=ax.get_yaxis_transform(),
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels([ARM_LABELS[a] for a in arms], fontsize=10,
                       color=theme.text_primary)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(highs) + 0.18))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="x", color=theme.grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    _style(ax, theme)

    seeds = summary["summary"]["seeds"]
    ax.set_title(
        "False success rate by verification arm",
        fontsize=14, fontweight="bold", color=theme.text_primary, loc="left", pad=32,
    )
    ax.text(
        0, 1.035,
        "Reported SUCCESS while the database said otherwise"
        f" · {len(seeds)} seed{'s' if len(seeds) != 1 else ''}, mean with range",
        transform=ax.transAxes, fontsize=9.5, color=theme.text_secondary,
    )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=theme.observability, label="trusts the agent's own signals"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=theme.assurance, label="checks external state"),
    ]
    legend = ax.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=9,
        labelcolor=theme.text_secondary,
    )
    legend.set_title(None)

    # Explicit margins rather than tight_layout: the swatch column lives in the
    # left margin, which tight_layout would happily crop.
    fig.subplots_adjust(left=0.28, right=0.97, top=0.80, bottom=0.12)
    fig.savefig(out_path, dpi=200, facecolor=theme.surface)
    plt.close(fig)
    return out_path


def heatmap_chart(summary: dict[str, Any], theme: Theme, out_path: Path) -> Path:
    """False success rate for each arm x failure mode cell."""
    by_mode = summary["summary"]["by_mode"]
    modes = [m for m in MODE_ORDER if m in by_mode]
    modes += [m for m in sorted(by_mode) if m not in MODE_ORDER]
    arms = [a for a in ("A", "B", "C", "D") if a in summary["summary"]["arms"]]
    grid = [[by_mode[m].get(a, {}).get("mean", 0.0) for m in modes] for a in arms]

    cmap = LinearSegmentedColormap.from_list("atl_seq", list(theme.ramp))
    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=theme.surface)
    mesh = ax.pcolormesh(
        grid, cmap=cmap, vmin=0.0, vmax=1.0,
        edgecolors=theme.surface, linewidth=2,  # 2px surface gap between cells
    )

    for row, arm in enumerate(arms):
        for col, _mode in enumerate(modes):
            value = grid[row][col]
            ax.text(
                col + 0.5, row + 0.5, f"{value:.0%}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=theme.surface if value > 0.55 else theme.text_primary,
            )
        _ = arm

    ax.set_xticks([i + 0.5 for i in range(len(modes))])
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], fontsize=8.5,
                       color=theme.text_primary)
    ax.set_yticks([i + 0.5 for i in range(len(arms))])
    ax.set_yticklabels([f"Arm {a}" for a in arms], fontsize=10,
                       color=theme.text_primary)
    ax.invert_yaxis()
    _style(ax, theme)
    ax.spines["bottom"].set_visible(False)

    bar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.025)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=theme.text_secondary, length=0, labelsize=8)
    bar.ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    ax.set_title(
        "False success rate by failure mode — where each arm is blind",
        fontsize=13, fontweight="bold", color=theme.text_primary, loc="left", pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=theme.surface)
    plt.close(fig)
    return out_path


def _fmt(stat: dict[str, Any]) -> str:
    return f"{stat['mean']:.1%} [{stat['min']:.1%}, {stat['max']:.1%}]"


def blind_spot_count(summary: dict[str, Any], results_dir: Path, stamp: str) -> int | None:
    """Episodes the verifier passed despite out-of-frame damage.

    Read from the summary when present; otherwise recomputed from the run
    records, which are the primary artifact and carry every episode's
    verification detail.
    """
    states = summary["summary"].get("state", [])
    if states and "verified_despite_collateral" in states[0]:
        return sum(s["verified_despite_collateral"] for s in states)

    files = sorted(results_dir.glob(f"run_{stamp}_*.json"))
    if not files:
        return None
    total = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for episode in data["episodes"]:
            verification = episode["evaluation"]["verification"]
            if (
                verification["verdict"] == "VERIFIED"
                and verification.get("unexpected_change_count", 0) > 0
            ):
                total += 1
    return total


def load_sensitivity(results_dir: Path, stamp: str) -> dict[str, Any] | None:
    """The offline both-definitions comparison, if `atl-rescore` has been run."""
    path = results_dir / f"sensitivity_{stamp}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_findings(
    summary: dict[str, Any],
    path: Path,
    blind_spot: int | None = None,
    sensitivity: dict[str, Any] | None = None,
) -> Path:
    """Fill the findings skeleton with the run's actual numbers."""
    data = summary["summary"]
    config = summary["config"]
    arms_data = data["arms"]
    seeds = data["seeds"]
    episodes = sum(s["episodes"] for s in data["state"])
    violated = sum(s["ground_truth_violated"] for s in data["state"])
    partial = sum(s["partial_completions"] for s in data["state"])
    duplicates = sum(s["duplicate_side_effects"] for s in data["state"])
    collateral = sum(s["collateral_damage"] for s in data["state"])
    tokens_in = sum(c["total_input_tokens"] for c in data["cost"])
    tokens_out = sum(c["total_output_tokens"] for c in data["cost"])
    mean_episode = sum(c["mean_episode_latency"] for c in data["cost"]) / len(data["cost"])
    mean_verify = sum(
        c["mean_verification_latency"] for c in data["cost"]
    ) / len(data["cost"])
    mean_reads = sum(
        c["mean_verification_db_reads"] for c in data["cost"]
    ) / len(data["cost"])

    lines = [
        "# Findings",
        "",
        f"Model under test: `{config['model']}` · seeds {seeds} · {episodes} episodes"
        f" ({episodes // len(seeds)} missions x {len(seeds)} seeds).",
        "",
        "![False success rate by arm](results/headline_false_success_rate.png)",
        "",
        "## Headline — false success rate by arm",
        "",
        "*Reported SUCCESS while the mission's expected_state was violated.*",
        "",
        "| Arm | What it trusts | False success | False failure | Indeterminate |",
        "|-----|----------------|---------------|---------------|---------------|",
    ]
    for arm in ("A", "B", "C", "D"):
        if arm not in arms_data:
            continue
        stats = arms_data[arm]
        lines.append(
            f"| **{arm}** | {ARM_LABELS[arm].split('— ')[-1]} |"
            f" {_fmt(stats['false_success_rate'])} |"
            f" {_fmt(stats['false_failure_rate'])} |"
            f" {_fmt(stats['indeterminate_rate'])} |"
        )

    lines += [
        "",
        "![Arm by failure mode](results/heatmap_arm_by_mode.png)",
        "",
        "## What actually happened to the database",
        "",
        f"- Episodes ending in a violated expected_state: **{violated}/{episodes}**",
        f"- Partial completions (some steps landed, others silently did not):"
        f" **{partial}**",
        f"- Duplicate side effects (double refunds / double charges): **{duplicates}**",
        f"- Episodes with collateral damage outside the mission's frame:"
        f" **{collateral}**",
    ]
    if blind_spot is not None:
        share = blind_spot / episodes if episodes else 0.0
        lines += [
            f"- **Verifier blind spot: {blind_spot}/{episodes} ({share:.1%})** episodes"
            " the verifier passed even though the episode mutated rows outside the",
            "  mission's declared frame. Under a stricter definition of ground truth —",
            "  one that treats *any* unauthorized write as a failure — Arm C's false"
            f" success rate would be {share:.1%} rather than 0%.",
        ]
    lines += [
        "",
        "## Arm D — recovery",
        "",
    ]
    if data.get("recovery"):
        needed = sum(r["needed"] for r in data["recovery"])
        recovered = sum(r["recovered"] for r in data["recovery"])
        escalated = sum(r["escalated"] for r in data["recovery"])
        damaged = sum(r["damaged"] for r in data["recovery"])
        rolled_back = sum(r["rolled_back"] for r in data["recovery"])
        rate = recovered / needed if needed else 0.0
        lines += [
            f"- Episodes needing recovery: **{needed}**",
            f"- Auto-recovered: **{recovered}** ({rate:.1%})",
            f"- Escalated with a structured incident report: **{escalated}**",
            f"- Repairs discarded by the never-worsen guard: **{rolled_back}**",
            f"- **Recovery-induced damage: {damaged}** (the playbook is required"
            " never to leave state worse than it found it)",
        ]
    if sensitivity:
        truths = sensitivity["ground_truths"]
        flipped = sensitivity["flipped"]
        lines += [
            "",
            "## Sensitivity: does the headline depend on how ground truth is defined?",
            "",
            "The same recorded episodes, re-scored offline with no API calls, against",
            "two definitions of *correct state*: **frame-scoped** (the mission's",
            "assertions plus global invariants) and **strict** (the same, plus: any",
            "write outside the mission's declared frame counts as damage).",
            "",
            "| Arm | Frame-scoped | Strict |",
            "|-----|--------------|--------|",
        ]
        labels = {
            "A": "A — agent self-report",
            "B": "B — tool responses",
            "C": "C — verifier (frame-scoped)",
        }
        for arm in ("A", "B", "C"):
            frame_rate = truths["frame-scoped"]["arms"][arm]["false_success_rate"]
            strict_rate = truths["strict"]["arms"][arm]["false_success_rate"]
            lines.append(f"| {labels[arm]} | {frame_rate:.1%} | {strict_rate:.1%} |")
        lines += [
            "",
            f"Episodes counted as violated: "
            f"{truths['frame-scoped']['episodes_violated']} frame-scoped vs "
            f"{truths['strict']['episodes_violated']} strict; "
            f"**{flipped['count']} episodes ({flipped['rate']:.1%}) flip** from",
            "VERIFIED to FAILED under the stricter lens.",
            "",
            "Two things this establishes. The headline gap between the observability",
            "arms and reality is **robust to the definition** — A and B move by only a",
            "couple of points. And Arm C's row is the one non-circular measurement of",
            "the verifier available here: as implemented it is frame-scoped, so scoring",
            "it against *strict* ground truth uses an evaluator that differs from the",
            "truth it is judged by, and it puts a real number on what the verifier misses.",
            "",
            "Regenerate with `atl-rescore`.",
        ]
    lines += [
        "",
        "## Cost of assurance vs cost of the work",
        "",
        f"- Agent: {tokens_in:,} input / {tokens_out:,} output tokens,"
        f" ~{mean_episode:.2f}s per episode",
        f"- Verifier: **0 tokens**, ~{mean_verify * 1000:.1f}ms and"
        f" ~{mean_reads:.0f} DB reads per episode",
        "",
        "## Reading these numbers honestly",
        "",
        "- **Arm C's false success rate is 0 by construction, not by measurement.**",
        "  The verifier evaluates the mission's `expected_state`, and that spec *is*",
        "  the ground truth, so C cannot disagree with itself. What C's column",
        "  actually demonstrates is the cost of obtaining ground truth (its",
        "  indeterminate rate, latency, and read count) — the *result* is the gap",
        "  between A/B and the truth.",
        "- Arm D is scored against the post-recovery state, since it changes the",
        "  state it reports on. Its false-failure rate is non-zero on purpose: D",
        "  escalates when it finds irreversible collateral damage even if the",
        "  mission's own assertions now hold.",
        "- Ground truth is scoped to each mission's declared frame plus the global",
        "  invariants. Damage outside that frame is recorded and reported as",
        "  collateral damage, but does not by itself fail an arm — the verifier",
        "  blind spot above is the measured size of that gap.",
        "- The agent is the only stochastic component; every measurement downstream",
        "  of a recorded episode is deterministic and replayable from the run JSON.",
        "",
        "## Notes",
        "",
        "<!-- Narrative interpretation goes here. -->",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render charts and findings.")
    parser.add_argument("--summary", type=Path, help="path to summary_<stamp>.json")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--findings", type=Path, default=REPO_ROOT / "findings.md")
    args = parser.parse_args(argv)

    summary_path = args.summary or newest_summary(args.results)
    summary = load_summary(summary_path)
    print(f"reading {summary_path.name}")

    outputs = []
    for theme in (LIGHT, DARK):
        outputs.append(
            headline_chart(
                summary, theme,
                args.results / f"headline_false_success_rate{theme.suffix}.png",
            )
        )
        outputs.append(
            heatmap_chart(
                summary, theme, args.results / f"heatmap_arm_by_mode{theme.suffix}.png"
            )
        )
    stamp = summary_path.stem.replace("summary_", "")
    blind_spot = blind_spot_count(summary, args.results, stamp)
    sensitivity = load_sensitivity(args.results, stamp)
    outputs.append(
        write_findings(summary, args.findings, blind_spot, sensitivity)
    )

    for path in outputs:
        resolved = path.resolve()
        try:
            display = resolved.relative_to(REPO_ROOT)
        except ValueError:
            display = resolved
        print(f"  wrote {display}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
