"""Tests for the cross-model comparison output."""

from __future__ import annotations

import json

import pytest

from agent_truth_lab.harness import compare


def make_summary(model: str, rates: dict[str, float]) -> dict:
    """A summary shaped like the runner's, with the rates we want to compare."""
    return {
        "config": {"model": model, "seeds": [42, 43, 44]},
        "summary": {
            "seeds": [42, 43, 44],
            "arms": {
                arm: {
                    "false_success_rate": {
                        "mean": rate,
                        "min": max(0.0, rate - 0.05),
                        "max": min(1.0, rate + 0.05),
                        "values": [rate],
                    },
                    "false_failure_rate": {"mean": 0.0, "min": 0.0, "max": 0.0,
                                           "values": [0.0]},
                    "indeterminate_rate": {"mean": 0.0, "min": 0.0, "max": 0.0,
                                           "values": [0.0]},
                }
                for arm, rate in rates.items()
            },
            "by_mode": {},
            "state": [{"episodes": 120, "ground_truth_violated": 60}],
            "recovery": [{"needed": 60, "recovered": 48}],
            "cost": [{"total_input_tokens": 1, "total_output_tokens": 1}],
        },
    }


def write_summaries(tmp_path, summaries):
    paths = []
    for index, summary in enumerate(summaries):
        path = tmp_path / f"summary_x_{index}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)
    return paths


RATES_SMALL = {"A": 0.52, "B": 0.40, "C": 0.0, "D": 0.0}
RATES_LARGE = {"A": 0.31, "B": 0.25, "C": 0.0, "D": 0.0}


def test_comparison_table_lists_every_model(tmp_path):
    summaries = compare.load_summaries(
        write_summaries(
            tmp_path,
            [make_summary("claude-haiku-4-5", RATES_SMALL),
             make_summary("claude-sonnet-5", RATES_LARGE)],
        )
    )
    table = compare.comparison_table(summaries)

    assert "claude-haiku-4-5" in table and "claude-sonnet-5" in table
    assert "52.0%" in table and "31.0%" in table
    # One header, one separator, one row per arm.
    assert len(table.splitlines()) == 2 + len(RATES_SMALL)


def test_summaries_sorted_by_capability(tmp_path):
    """The chart should read in increasing capability regardless of file order."""
    summaries = compare.load_summaries(
        write_summaries(
            tmp_path,
            [make_summary("claude-sonnet-5", RATES_LARGE),
             make_summary("claude-haiku-4-5", RATES_SMALL)],
        )
    )
    assert [s["config"]["model"] for s in summaries] == [
        "claude-haiku-4-5", "claude-sonnet-5",
    ]


def test_comparison_chart_renders_both_themes(tmp_path):
    summaries = compare.load_summaries(
        write_summaries(
            tmp_path,
            [make_summary("claude-haiku-4-5", RATES_SMALL),
             make_summary("claude-sonnet-5", RATES_LARGE)],
        )
    )
    for theme in (compare.LIGHT, compare.DARK):
        out = tmp_path / f"cmp_{theme.name}.png"
        compare.comparison_chart(summaries, theme, out)
        assert out.exists() and out.stat().st_size > 5_000


def test_cli_writes_chart_and_table(tmp_path):
    write_summaries(
        tmp_path,
        [make_summary("claude-haiku-4-5", RATES_SMALL),
         make_summary("claude-sonnet-5", RATES_LARGE)],
    )
    exit_code = compare.main(["--results", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "model_comparison.png").exists()
    assert (tmp_path / "model_comparison_dark.png").exists()
    assert "claude-sonnet-5" in (tmp_path / "model_comparison.md").read_text(
        encoding="utf-8"
    )


def test_cli_refuses_a_single_summary(tmp_path, capsys):
    write_summaries(tmp_path, [make_summary("claude-haiku-4-5", RATES_SMALL)])
    assert compare.main(["--results", str(tmp_path)]) == 1
    assert "at least two" in capsys.readouterr().out


@pytest.mark.parametrize("zero_arm", ["C", "D"])
def test_zero_rate_models_still_render(tmp_path, zero_arm):
    """Bars of width zero must not break the renderer (they are the good case)."""
    rates = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    summaries = compare.load_summaries(
        write_summaries(
            tmp_path,
            [make_summary("claude-haiku-4-5", rates),
             make_summary("claude-sonnet-5", rates)],
        )
    )
    out = tmp_path / f"zero_{zero_arm}.png"
    compare.comparison_chart(summaries, compare.LIGHT, out)
    assert out.exists()
