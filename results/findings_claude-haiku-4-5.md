# Findings

Model under test: `claude-haiku-4-5` · seeds [42, 43, 44] · 120 episodes (40 missions x 3 seeds).

![False success rate by arm](results/headline_false_success_rate.png)

## Headline — false success rate by arm

*Reported SUCCESS while the mission's expected_state was violated.*

| Arm | What it trusts | False success | False failure | Indeterminate |
|-----|----------------|---------------|---------------|---------------|
| **A** | agent self-report | 50.8% [47.5%, 55.0%] | 0.0% [0.0%, 0.0%] | 10.8% [7.5%, 15.0%] |
| **B** | tool responses | 39.2% [35.0%, 42.5%] | 12.5% [5.0%, 17.5%] | 0.0% [0.0%, 0.0%] |
| **C** | state verifier | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| **D** | verifier + recovery | 0.0% [0.0%, 0.0%] | 3.3% [2.5%, 5.0%] | 0.0% [0.0%, 0.0%] |

![Arm by failure mode](results/heatmap_arm_by_mode.png)

## What actually happened to the database

- Episodes ending in a violated expected_state: **66/120**
- Partial completions (some steps landed, others silently did not): **55**
- Duplicate side effects (double refunds / double charges): **16**
- Episodes with collateral damage outside the mission's frame: **16**
- **Verifier blind spot: 0/120 (0.0%)** episodes the verifier passed even though the episode mutated rows outside the
  mission's declared frame. Under a stricter definition of ground truth —
  one that treats *any* unauthorized write as a failure — Arm C's false success rate would be 0.0% rather than 0%.

## Arm D — recovery

- Episodes needing recovery: **66**
- Auto-recovered: **48** (72.7%)
- Escalated with a structured incident report: **18**
- Repairs discarded by the never-worsen guard: **0**
- **Recovery-induced damage: 0** (the playbook is required never to leave state worse than it found it)

## Cost of assurance vs cost of the work

- Agent: 490,313 input / 29,809 output tokens, ~3.70s per episode
- Verifier: **0 tokens**, ~6.6ms and ~20 DB reads per episode

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
