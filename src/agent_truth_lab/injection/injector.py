"""The injection wrapper the agent loop calls in place of the clean tools.

The agent cannot observe this layer: injected responses use the same envelope
as clean ones. Which tool misbehaves, and how, comes from a per-mission plan
`{tool_name: failure_mode}`. Everything is deterministic given the database
state and the call sequence (see modes.py).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

from agent_truth_lab.environment.db import SimClock
from agent_truth_lab.environment.tools import ToolResult
from agent_truth_lab.injection.modes import (
    CLEAN_TOOLS,
    HANDLERS,
    MULTI_STEP_TOOLS,
    EpisodeCrash,
    FailureMode,
    clean_call,
)

__all__ = ["EpisodeCrash", "Injector", "validate_plan"]


def validate_plan(plan: dict[str, str]) -> dict[str, FailureMode]:
    """Validate and normalize an injection plan. Raises ValueError on bad config."""
    validated: dict[str, FailureMode] = {}
    for tool_name, mode_name in plan.items():
        if tool_name not in CLEAN_TOOLS:
            raise ValueError(f"injection plan targets unknown tool '{tool_name}'")
        try:
            mode = FailureMode(mode_name)
        except ValueError as exc:
            raise ValueError(f"unknown failure mode '{mode_name}'") from exc
        if mode is FailureMode.PARTIAL_COMPLETION and tool_name not in MULTI_STEP_TOOLS:
            raise ValueError(
                f"partial_completion requires a multi-step tool; '{tool_name}' is single-step"
            )
        validated[tool_name] = mode
    return validated


class Injector:
    """Dispatches tool calls, applying the configured failure mode when one exists.

    Stateful: tracks per-tool call counts within an episode (F3 behaves
    differently on the first call vs retries). One Injector per episode.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: SimClock,
        plan: dict[str, str] | None = None,
    ) -> None:
        self.conn = conn
        self.clock = clock
        self.plan = validate_plan(plan or {})
        self._calls: Counter[str] = Counter()

    def call(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool call, injected or clean. Raises EpisodeCrash under F7."""
        if tool_name not in CLEAN_TOOLS:
            raise ValueError(f"unknown tool '{tool_name}'")
        self._calls[tool_name] += 1
        mode = self.plan.get(tool_name)
        if mode is None:
            return clean_call(self.conn, self.clock, tool_name, args)
        handler = HANDLERS[mode]
        return handler(self.conn, self.clock, tool_name, args, self._calls[tool_name])
