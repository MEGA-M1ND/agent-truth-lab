"""Tests for RunConfig and the runner CLI's flag wiring.

Focused on --resolve-conflicts specifically: it must imply --read-tools (the
instruction is meaningless without a read channel to act on), and this is
exactly the kind of one-line mistake that would silently invalidate a paid
experiment run rather than fail loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_truth_lab.harness.runner import RunConfig, main

CONFIG_BASE = {
    "models": ["claude-haiku-4-5"],
    "seeds": [42],
    "max_turns": 8,
    "max_tokens": 1024,
    "temperature": 0.0,
    "arms": ["A", "B", "C", "D"],
}


def write_config(tmp_path: Path, **overrides) -> Path:
    data = {**CONFIG_BASE, **overrides}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_read_tools_and_resolve_conflicts_default_false(tmp_path):
    config = RunConfig.load(write_config(tmp_path))
    assert config.read_tools is False
    assert config.resolve_conflicts is False


def test_resolve_conflicts_loads_from_config(tmp_path):
    config = RunConfig.load(
        write_config(tmp_path, read_tools=True, resolve_conflicts=True)
    )
    assert config.read_tools is True
    assert config.resolve_conflicts is True


def test_resolve_conflicts_round_trips_through_to_dict(tmp_path):
    config = RunConfig.load(
        write_config(tmp_path, read_tools=True, resolve_conflicts=True)
    )
    assert config.to_dict()["resolve_conflicts"] is True


def test_cli_resolve_conflicts_flag_implies_read_tools(tmp_path, capsys):
    config_path = write_config(tmp_path)  # read_tools/resolve_conflicts both default False

    exit_code = main(
        ["--config", str(config_path), "--resolve-conflicts", "--estimate-only"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "read tools" in out
    assert "trust-the-read-on-conflict" in out


def test_cli_read_tools_flag_alone_does_not_set_resolve_conflicts(tmp_path, capsys):
    config_path = write_config(tmp_path)

    exit_code = main(["--config", str(config_path), "--read-tools", "--estimate-only"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "on (agent can self-verify)" in out
    assert "trust-the-read-on-conflict" not in out


def test_cli_neither_flag_reports_off(tmp_path, capsys):
    config_path = write_config(tmp_path)

    exit_code = main(["--config", str(config_path), "--estimate-only"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "read tools     : off" in out


@pytest.mark.parametrize("read_tools", [True, False])
def test_estimate_cost_uses_read_tool_token_estimate(read_tools):
    from agent_truth_lab.harness.runner import (
        TOKENS_PER_EPISODE,
        TOKENS_PER_EPISODE_READ_TOOLS,
        estimate_cost,
    )

    _, tokens_in, _ = estimate_cost("claude-haiku-4-5", 1, read_tools=read_tools)
    expected = (
        TOKENS_PER_EPISODE_READ_TOOLS[0] if read_tools else TOKENS_PER_EPISODE[0]
    )
    assert tokens_in == expected
