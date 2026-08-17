"""M3 tests: mission builder + agent loop driven by a scripted fake client.

The fake client mimics the shape of Anthropic Messages API responses
(attribute access on content blocks, stop_reason, usage), so the loop under
test is byte-for-byte the production loop with no network involved.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent_truth_lab.agent import loop, missions
from agent_truth_lab.environment import db

SEED = 42

# ---------------------------------------------------------------------------
# fake Anthropic client
# ---------------------------------------------------------------------------


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def fake_response(blocks, stop_reason):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


class FakeClient:
    """Replays scripted responses; repeats the last one if the loop asks for more."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._next = 0
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        index = min(self._next, len(self._responses) - 1)
        self._next += 1
        return self._responses[index]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fresh_env(seed: int = SEED) -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, seed)
    return conn


def reload_dump(dump: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(dump)
    return conn


def find_mission(mission_list, mission_id):
    return next(m for m in mission_list if m.mission_id == mission_id)


@pytest.fixture(scope="module")
def mission_set():
    conn = fresh_env()
    try:
        return missions.build_missions(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# mission builder
# ---------------------------------------------------------------------------


def test_mission_counts(mission_set):
    assert len(mission_set) == 40
    clean = [m for m in mission_set if not m.injection]
    injected = [m for m in mission_set if m.injection]
    assert len(clean) == missions.CLEAN_MISSION_COUNT
    assert len(injected) == missions.INJECTED_MISSION_COUNT


def test_every_failure_mode_covered_at_least_three_times(mission_set):
    from agent_truth_lab.injection.modes import FailureMode

    counts = {mode.value: 0 for mode in FailureMode}
    for mission in mission_set:
        for mode in mission.injection.values():
            counts[mode] += 1
    assert all(n >= 3 for n in counts.values()), counts


def test_mission_ids_unique(mission_set):
    ids = [m.mission_id for m in mission_set]
    assert len(set(ids)) == len(ids)


def test_missions_reference_real_entities(mission_set):
    conn = fresh_env()
    mission = find_mission(mission_set, "m01_refund_full_clean")
    refund_assertion = next(a for a in mission.assertions if a.table == "refunds")
    order = conn.execute(
        "SELECT * FROM orders WHERE id = ?", (refund_assertion.where["order_id"],)
    ).fetchone()
    assert order is not None and order["status"] == "delivered"
    assert refund_assertion.expect["amount_paise"] == order["amount_paise"]
    conn.close()


def test_build_missions_is_deterministic():
    conn_a, conn_b = fresh_env(), fresh_env()
    a = [m.to_dict() for m in missions.build_missions(conn_a)]
    b = [m.to_dict() for m in missions.build_missions(conn_b)]
    assert a == b
    conn_a.close()
    conn_b.close()


def test_mission_round_trip(mission_set):
    for mission in mission_set:
        assert missions.Mission.from_dict(mission.to_dict()) == mission


# ---------------------------------------------------------------------------
# structural variance
# ---------------------------------------------------------------------------


def varied(seed: int):
    conn = fresh_env(seed)
    try:
        return missions.build_missions(conn, seed=seed, structural_variance=True)
    finally:
        conn.close()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_varied_set_honours_coverage_constraints(seed):
    """Variance may reshuffle the mix, never break the experiment's floors."""
    from agent_truth_lab.injection.modes import FailureMode

    mission_set = varied(seed)
    assert len(mission_set) == 40
    assert sum(1 for m in mission_set if not m.injection) == missions.CLEAN_MISSION_COUNT
    assert sum(1 for m in mission_set if m.injection) == missions.INJECTED_MISSION_COUNT

    counts = {mode.value: 0 for mode in FailureMode}
    for mission in mission_set:
        for mode in mission.injection.values():
            counts[mode] += 1
    assert all(n >= missions.MIN_PER_MODE for n in counts.values()), counts


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_varied_injections_target_tools_the_mission_uses(seed):
    """Injecting into a tool the mission never calls would be a silent no-op."""
    for mission in varied(seed):
        if not mission.injection:
            continue
        allowed = missions.ARCHETYPE_TOOLS[mission.archetype]
        for tool, mode in mission.injection.items():
            assert tool in allowed, f"{mission.mission_id}: {tool} not in {allowed}"
            if mode == "partial_completion":
                assert tool in ("issue_refund", "retry_subscription_charge")


def test_varied_set_is_deterministic():
    assert [m.to_dict() for m in varied(42)] == [m.to_dict() for m in varied(42)]


def test_seeds_produce_structurally_different_sets():
    """The point of the mode: seeds vary task mix, not just entity ids."""
    structures = {
        seed: [(m.archetype, tuple(sorted(m.injection.items()))) for m in varied(seed)]
        for seed in (42, 43, 44)
    }
    assert structures[42] != structures[43]
    assert structures[43] != structures[44]


def test_varied_set_covers_multiple_archetypes():
    for seed in (42, 43, 44):
        archetypes = {m.archetype for m in varied(seed)}
        assert len(archetypes) >= 5, archetypes


def test_structural_variance_requires_a_seed():
    conn = fresh_env()
    with pytest.raises(ValueError, match="requires a seed"):
        missions.build_missions(conn, structural_variance=True)
    conn.close()


def test_fixed_set_is_still_the_default(mission_set):
    """The published v1 numbers must stay reproducible."""
    conn = fresh_env()
    assert [m.to_dict() for m in missions.build_missions(conn)] == [
        m.to_dict() for m in mission_set
    ]
    conn.close()


# ---------------------------------------------------------------------------
# agent loop
# ---------------------------------------------------------------------------


def test_loop_clean_refund_episode(mission_set):
    mission = find_mission(mission_set, "m01_refund_full_clean")
    env = fresh_env()
    order_id = next(a for a in mission.assertions if a.table == "orders").where["id"]
    order = env.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    env.close()

    client = FakeClient(
        [
            fake_response(
                [
                    text_block("Issuing the refund."),
                    tool_block(
                        "tu_1",
                        "issue_refund",
                        {
                            "order_id": order_id,
                            "amount_paise": order["amount_paise"],
                            "reason": "customer return",
                        },
                    ),
                ],
                "tool_use",
            ),
            fake_response(
                [
                    tool_block(
                        "tu_2",
                        "send_customer_email",
                        {
                            "customer_id": order["customer_id"],
                            "template": "refund_completed",
                            "related_entity": "refund:21",
                        },
                    )
                ],
                "tool_use",
            ),
            fake_response(
                [text_block("TASK_COMPLETE: refunded and emailed the customer.")],
                "end_turn",
            ),
        ]
    )

    record = loop.run_episode(mission, SEED, "fake-model", client)

    assert not record.crashed and record.stop_reason == "end_turn"
    assert record.final_text.startswith("TASK_COMPLETE")
    assert [c.tool_name for c in record.tool_calls] == [
        "issue_refund", "send_customer_email",
    ]
    assert all(c.result["ok"] for c in record.tool_calls)
    assert record.usage_input_tokens == 300 and record.usage_output_tokens == 150

    # trajectory shape: user, assistant, tool_results, assistant, results, assistant
    roles = [m["role"] for m in record.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    tool_result = record.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu_1"
    assert json.loads(tool_result["content"])["ok"] is True

    # DB state actually changed, and the dump is reloadable
    post = reload_dump(record.db_dump)
    assert post.execute(
        "SELECT status FROM orders WHERE id = ?", (order_id,)
    ).fetchone()["status"] == "refunded"
    audit = post.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()["n"]
    assert audit == 2
    post.close()


def test_loop_crash_episode_records_committed_side_effect(mission_set):
    mission = find_mission(mission_set, "m38_refund_f7_crash")
    order_id = next(a for a in mission.assertions if a.table == "orders").where["id"]
    env = fresh_env()
    amount = env.execute(
        "SELECT amount_paise FROM orders WHERE id = ?", (order_id,)
    ).fetchone()["amount_paise"]
    env.close()

    client = FakeClient(
        [
            fake_response(
                [
                    tool_block(
                        "tu_1",
                        "issue_refund",
                        {"order_id": order_id, "amount_paise": amount, "reason": "rtn"},
                    )
                ],
                "tool_use",
            ),
        ]
    )

    record = loop.run_episode(mission, SEED, "fake-model", client)

    assert record.crashed and record.stop_reason == "crashed"
    assert record.final_text is None
    assert record.tool_calls == []  # the agent never saw a result
    post = reload_dump(record.db_dump)
    n = post.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order_id,)
    ).fetchone()["n"]
    assert n == 1  # ...but the side effect committed
    post.close()


def test_loop_error_envelope_flows_back_to_agent(mission_set):
    mission = find_mission(mission_set, "m01_refund_full_clean")
    order_id = next(a for a in mission.assertions if a.table == "orders").where["id"]

    client = FakeClient(
        [
            fake_response(
                [
                    tool_block(
                        "tu_1",
                        "issue_refund",
                        {"order_id": order_id, "amount_paise": 10**9, "reason": "x"},
                    )
                ],
                "tool_use",
            ),
            fake_response(
                [text_block("TASK_FAILED: refund amount exceeds the order total.")],
                "end_turn",
            ),
        ]
    )

    record = loop.run_episode(mission, SEED, "fake-model", client)

    assert record.tool_calls[0].result["ok"] is False
    assert record.tool_calls[0].result["http_status"] == 422
    assert record.final_text.startswith("TASK_FAILED")
    envelope = json.loads(record.messages[2]["content"][0]["content"])
    assert envelope["ok"] is False


def test_loop_max_turns_cap(mission_set):
    mission = find_mission(mission_set, "m12_settlement_clean")
    endless = fake_response(
        [tool_block("tu_x", "create_settlement", {"merchant_day": "2020-01-01"})],
        "tool_use",
    )
    client = FakeClient([endless])

    record = loop.run_episode(mission, SEED, "fake-model", client, max_turns=3)

    assert record.stop_reason == "max_turns"
    assert record.final_text is None
    assert len(client.requests) == 3


def test_loop_omits_temperature_when_none(mission_set):
    mission = find_mission(mission_set, "m12_settlement_clean")
    client = FakeClient([fake_response([text_block("TASK_FAILED: n/a")], "end_turn")])

    loop.run_episode(mission, SEED, "fake-model", client, temperature=None)

    assert "temperature" not in client.requests[0]


def test_episode_record_serializes(mission_set):
    mission = find_mission(mission_set, "m12_settlement_clean")
    client = FakeClient(
        [fake_response([text_block("TASK_COMPLETE: done")], "end_turn")]
    )
    record = loop.run_episode(mission, SEED, "fake-model", client)
    payload = json.dumps(record.to_dict())
    assert '"m12_settlement_clean"' in payload


# ---------------------------------------------------------------------------
# read tools (M7)
# ---------------------------------------------------------------------------


def test_read_tools_excluded_by_default(mission_set):
    mission = find_mission(mission_set, "m12_settlement_clean")
    client = FakeClient([fake_response([text_block("TASK_COMPLETE: done")], "end_turn")])

    loop.run_episode(mission, SEED, "fake-model", client)

    sent_tools = {t["name"] for t in client.requests[0]["tools"]}
    assert sent_tools.isdisjoint({t["name"] for t in loop.READ_TOOL_SCHEMAS})
    assert loop.READ_TOOLS_ADDENDUM not in client.requests[0]["system"]


def test_read_tools_included_when_opted_in(mission_set):
    mission = find_mission(mission_set, "m12_settlement_clean")
    client = FakeClient([fake_response([text_block("TASK_COMPLETE: done")], "end_turn")])

    loop.run_episode(mission, SEED, "fake-model", client, include_read_tools=True)

    sent_tools = {t["name"] for t in client.requests[0]["tools"]}
    assert {t["name"] for t in loop.READ_TOOL_SCHEMAS} <= sent_tools
    assert loop.READ_TOOLS_ADDENDUM in client.requests[0]["system"]
    assert loop.CONFLICT_RESOLUTION_ADDENDUM not in client.requests[0]["system"]


def test_resolve_conflicts_appends_the_instruction(mission_set):
    mission = find_mission(mission_set, "m12_settlement_clean")
    client = FakeClient([fake_response([text_block("TASK_COMPLETE: done")], "end_turn")])

    loop.run_episode(
        mission, SEED, "fake-model", client,
        include_read_tools=True, resolve_conflicts=True,
    )

    system = client.requests[0]["system"]
    assert loop.READ_TOOLS_ADDENDUM in system
    assert loop.CONFLICT_RESOLUTION_ADDENDUM in system
    # the conflict instruction must come after the read-tools addendum
    assert system.index(loop.READ_TOOLS_ADDENDUM) < system.index(
        loop.CONFLICT_RESOLUTION_ADDENDUM
    )


def test_resolve_conflicts_without_read_tools_has_no_effect(mission_set):
    """The instruction is meaningless without a read channel to act on."""
    mission = find_mission(mission_set, "m12_settlement_clean")
    client = FakeClient([fake_response([text_block("TASK_COMPLETE: done")], "end_turn")])

    loop.run_episode(
        mission, SEED, "fake-model", client,
        include_read_tools=False, resolve_conflicts=True,
    )

    system = client.requests[0]["system"]
    assert loop.CONFLICT_RESOLUTION_ADDENDUM not in system
    sent_tools = {t["name"] for t in client.requests[0]["tools"]}
    assert sent_tools.isdisjoint({t["name"] for t in loop.READ_TOOL_SCHEMAS})


def test_agent_can_call_a_read_tool_and_it_reaches_the_injector(mission_set):
    """An agent-issued get_order call flows through the loop exactly like a write."""
    mission = find_mission(mission_set, "m01_refund_full_clean")
    order_id = next(a for a in mission.assertions if a.table == "orders").where["id"]

    client = FakeClient(
        [
            fake_response(
                [tool_block("tu_1", "get_order", {"order_id": order_id})], "tool_use"
            ),
            fake_response([text_block("TASK_COMPLETE: checked first.")], "end_turn"),
        ]
    )

    record = loop.run_episode(
        mission, SEED, "fake-model", client, include_read_tools=True
    )

    assert record.tool_calls[0].tool_name == "get_order"
    assert record.tool_calls[0].result["ok"] is True
    assert record.tool_calls[0].result["data"]["id"] == order_id


def test_read_tool_call_does_not_mutate_the_episode_db(mission_set):
    mission = find_mission(mission_set, "m01_refund_full_clean")
    order_id = next(a for a in mission.assertions if a.table == "orders").where["id"]
    client = FakeClient(
        [
            fake_response(
                [tool_block("tu_1", "get_refund", {"order_id": order_id})], "tool_use"
            ),
            fake_response([text_block("TASK_FAILED: nothing done.")], "end_turn"),
        ]
    )

    record = loop.run_episode(
        mission, SEED, "fake-model", client, include_read_tools=True
    )

    post = reload_dump(record.db_dump)
    n = post.execute(
        "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order_id,)
    ).fetchone()["n"]
    assert n == 0  # a pure read leaves no trace
