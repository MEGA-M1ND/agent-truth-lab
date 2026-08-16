"""M3 smoke test: run 2 clean missions against the real Anthropic API.

Usage:  python scripts/smoke_m3.py
Needs ANTHROPIC_API_KEY. Uses the model from config/experiment.yaml (a cheap
model by design). Writes episode JSONs to results/smoke/ and prints a summary
plus a cost estimate.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import anthropic
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_truth_lab.agent import loop, missions  # noqa: E402
from agent_truth_lab.environment import db  # noqa: E402

# $/MTok (input, output) for the cheap models we might configure.
PRICES = {"claude-haiku-4-5": (1.00, 5.00)}

SMOKE_MISSIONS = ["m01_refund_full_clean", "m12_settlement_clean"]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; aborting.")
        return 1

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "experiment.yaml").read_text()
    )
    model = config["model"]
    seed = config["seeds"][0]

    conn = db.connect(":memory:")
    db.init_db(conn)
    db.seed(conn, seed)
    mission_set = {m.mission_id: m for m in missions.build_missions(conn)}
    conn.close()

    out_dir = Path(__file__).resolve().parents[1] / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    total_in = total_out = 0
    for mission_id in SMOKE_MISSIONS:
        mission = mission_set[mission_id]
        print(f"\n=== {mission_id} (model={model}, seed={seed}) ===")
        print(f"instruction: {mission.instruction}")
        record = loop.run_episode(
            mission,
            seed,
            model,
            client,
            max_turns=config["max_turns"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        loop.save_episode(record, str(out_dir / f"{mission_id}.json"))
        total_in += record.usage_input_tokens
        total_out += record.usage_output_tokens

        print(f"stop_reason: {record.stop_reason}  crashed: {record.crashed}")
        for call in record.tool_calls:
            print(
                f"  tool: {call.tool_name}({call.args}) ->"
                f" ok={call.result['ok']} http={call.result['http_status']}"
            )
        print(f"final: {record.final_text!r}")

        # quick eyeball of the post-episode state for the touched tables
        post = sqlite3.connect(":memory:")
        post.row_factory = sqlite3.Row
        post.executescript(record.db_dump)
        for assertion in mission.assertions:
            clause = " AND ".join(f"{k} = ?" for k in assertion.where)
            rows = post.execute(
                f"SELECT * FROM {assertion.table} WHERE {clause}",  # noqa: S608
                tuple(assertion.where.values()),
            ).fetchall()
            print(f"  {assertion.table} where {assertion.where}: {len(rows)} row(s)")
            for row in rows:
                print(f"    {dict(row)}")
        post.close()

    price_in, price_out = PRICES.get(model, (5.00, 25.00))
    cost = total_in / 1e6 * price_in + total_out / 1e6 * price_out
    print(
        f"\ntokens: {total_in} in / {total_out} out  —  estimated cost ${cost:.4f}"
        f" for {len(SMOKE_MISSIONS)} episodes"
    )
    print(f"episodes saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
