# Findings

Model under test: `claude-haiku-4-5` · seeds [42, 43, 44] · 120 episodes (40 missions x 3 seeds).

![False success rate by arm](results/headline_false_success_rate.png)

## Headline — false success rate by arm

*Reported SUCCESS while the mission's expected_state was violated.*

| Arm | What it trusts | False success | False failure | Indeterminate |
|-----|----------------|---------------|---------------|---------------|
| **A** | agent self-report | 52.5% [52.5%, 52.5%] | 0.0% [0.0%, 0.0%] | 7.5% [7.5%, 7.5%] |
| **B** | tool responses | 40.0% [40.0%, 40.0%] | 7.5% [7.5%, 7.5%] | 0.0% [0.0%, 0.0%] |
| **C** | state verifier | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| **D** | verifier + recovery | 0.0% [0.0%, 0.0%] | 5.0% [5.0%, 5.0%] | 0.0% [0.0%, 0.0%] |

![Arm by failure mode](results/heatmap_arm_by_mode.png)

## What actually happened to the database

- Episodes ending in a violated expected_state: **69/120**
- Partial completions (some steps landed, others silently did not): **54**
- Duplicate side effects (double refunds / double charges): **9**
- Episodes with collateral damage outside the mission's frame: **21**
- **Verifier blind spot: 3/120 (2.5%)** episodes the verifier passed even though the episode mutated rows outside the
  mission's declared frame. Under a stricter definition of ground truth —
  one that treats *any* unauthorized write as a failure — Arm C's false success rate would be 2.5% rather than 0%.

## Arm D — recovery

- Episodes needing recovery: **69**
- Auto-recovered: **54** (78.3%)
- Escalated with a structured incident report: **15**
- Repairs discarded by the never-worsen guard: **0**
- **Recovery-induced damage: 0** (the playbook is required never to leave state worse than it found it)

## Sensitivity: does the headline depend on how ground truth is defined?

The same recorded episodes, re-scored offline with no API calls, against
two definitions of *correct state*: **frame-scoped** (the mission's
assertions plus global invariants) and **strict** (the same, plus: any
write outside the mission's declared frame counts as damage).

| Arm | Frame-scoped | Strict |
|-----|--------------|--------|
| A — agent self-report | 52.5% | 55.0% |
| B — tool responses | 40.0% | 42.5% |
| C — verifier (frame-scoped) | 0.0% | 2.5% |

Episodes counted as violated: 69 frame-scoped vs 72 strict; **3 episodes (2.5%) flip** from
VERIFIED to FAILED under the stricter lens.

Two things this establishes. The headline gap between the observability
arms and reality is **robust to the definition** — A and B move by only a
couple of points. And Arm C's row is the one non-circular measurement of
the verifier available here: as implemented it is frame-scoped, so scoring
it against *strict* ground truth uses an evaluator that differs from the
truth it is judged by, and it puts a real number on what the verifier misses.

Regenerate with `atl-rescore`.

## Cost of assurance vs cost of the work

- Agent: 524,175 input / 33,118 output tokens, ~4.02s per episode
- Verifier: **0 tokens**, ~7.8ms and ~20 DB reads per episode

## Reading these numbers honestly

- **Arm C's false success rate is 0 by construction, not by measurement.**
  The verifier evaluates the mission's `expected_state`, and that spec *is*
  the ground truth, so C cannot disagree with itself. What C's column
  actually demonstrates is the cost of obtaining ground truth (its
  indeterminate rate, latency, and read count) — the *result* is the gap
  between A/B and the truth.
- Arm D is scored against the post-recovery state, since it changes the
  state it reports on. Its false-failure rate is non-zero on purpose: D
  escalates when it finds irreversible collateral damage even if the
  mission's own assertions now hold.
- Ground truth is scoped to each mission's declared frame plus the global
  invariants. Damage outside that frame is recorded and reported as
  collateral damage, but does not by itself fail an arm — the verifier
  blind spot above is the measured size of that gap.
- The agent is the only stochastic component; every measurement downstream
  of a recorded episode is deterministic and replayable from the run JSON.

## Notes

<!-- Narrative interpretation goes here. -->
