# Experiment log

Dated record of design decisions and runs. Maintained by the architect; entries below
were written as each milestone landed.

## 2026-08-16 — M1 environment

- Business invariants are enforced **only in code**, never as SQL CHECK constraints: the
  database must accept invariant-violating writes so the F6 injector can create them and
  the verifier can be the thing that catches them. Only foreign keys are enforced in SQL.
- All timestamps come from a deterministic `SimClock` (sim present = 2026-01-10, seeded
  history on 2025-12-30/31). Wall-clock timestamps would break byte-identical reseeding.
- `refunded` is reachable **only** via `issue_refund`, never `update_order_status`, so a
  clean tool can never produce an unbacked refunded order.
- A declined charge returns `ok:false / 402` rather than a success envelope with a failed
  payload, which keeps Arm B's "all calls ok" signal meaningful.
- Refund step 3 credits the customer's balance (the spec said "decrement merchant
  liability" but defines no merchant table) — approved.

## 2026-08-16 — M2 injection

- **F5 `stale_read` is implemented as a lost write** (the spec's "or vice versa" option):
  the tool executes, the response is computed from post-operation state, then the
  transaction is reverted. The response is therefore byte-identical to a durable
  execution, which is what distinguishes it from F1's fabricated envelope.
- **F3's retry path bypasses the clean tool's duplicate guard.** M1's `issue_refund`
  rejects a second refund per order, which would make "retry creates a duplicate"
  impossible. The injector performs the raw side effects instead, simulating the TOCTOU
  race a read-then-write guard cannot prevent under concurrency — which is how real
  duplicate refunds happen behind idempotency-key-less APIs.
- Injection uses **no randomness at all**: wrong targets follow next-eligible-with-
  wraparound, corruptions use a fixed digit swap. Injection is a pure function of DB state
  and call order, which is stronger than "deterministic from the run seed".
- `partial_completion` is rejected at config time for single-step tools.

## 2026-08-16 — M3 agent loop and missions

- **Missions are generated in Python, not static YAML** (deviation from the spec's
  example). Entity ids and amounts are seed-dependent, so a static file cannot survive the
  3-seed requirement. `build_missions` is a deterministic pure read of the seeded DB and
  the fully-expanded mission set is recorded in every run's JSON.
- Assertions pin predicted auto-increment ids (`refund:25`), which is what lets the
  verifier catch F6's dangling-reference corruption. Confirmed correct against the live
  API in the smoke run.
- The system prompt permits **one retry on 5xx/timeouts** — without it, F3 would measure
  false failures instead of duplicate side effects — and mandates `TASK_COMPLETE:` /
  `TASK_FAILED:` markers so Arm A's parser is deterministic.
- The LLM is the only stochastic component; everything downstream of a recorded episode is
  deterministic and replayable.

## 2026-08-16 — M4 arms A–C and metrics

- **Arm C's false success rate is 0 by construction.** The verifier evaluates the mission's
  `expected_state`, and that spec *is* ground truth, so C cannot disagree with itself. This
  is stated in the code, the findings, and the README rather than presented as a result.
- The **frame check** (diffing the post-episode DB against a fresh reseed) is deliberately
  *evidence, not verdict*: it proves F2's wrong-target damage instead of merely inferring
  it, and it is what M5's never-worsen guard is built on. Keeping it out of the verdict
  avoids inflating Arm C's false-failure rate on benign extra actions.
- `INDETERMINATE` is reserved for what the verifier genuinely cannot decide (unreadable
  snapshot, assertion naming a missing table/column) — never as a hedge.
- False-success/false-failure rates are computed over determinate episodes only; the
  indeterminate rate is reported separately so nothing is silently scored as a pass.

## 2026-08-16 — M5 Arm D recovery

- The never-worsen guard **caught a real bug in the first repair strategy**: deleting a
  stray refund orphaned the confirmation email that referenced it, trading one invariant
  violation for another. The guard discarded the repair and escalated. Fix: re-point the
  misapplied refund at the intended order **in place**, which reverses the money on the
  wrong account while preserving referential integrity.
- Recovery is granted a capability the agent lacks — an **idempotency key on refunds** — so
  a replayed repair is a no-op. That asymmetry is the thesis in miniature.
- Repairs write **directly to the database, not through the tool layer**: injected tools
  cannot be trusted to perform a repair, and the legal-transition table constrains the
  agent, not the incident response restoring state.
- **Irreversible collateral damage forces escalation** even when the mission's assertions
  pass. This is why Arm D shows a non-zero false-*failure* rate; reported, not tuned away.

## 2026-08-16 — M6 full run

First full run: `claude-haiku-4-5`, seeds [42, 43, 44], 40 missions x 3 seeds = 120
episodes. Cost $0.69, wall clock 486s.

| Arm | False success | False failure |
|-----|---------------|---------------|
| A — self-report | 52.5% | 0.0% |
| B — tool responses | 40.0% | 7.5% |
| C — state verifier | 0.0% (by construction) | 0.0% |
| D — verifier + recovery | 0.0% | 5.0% |

Arm D: 54/69 auto-recovered (78.3%), 15 escalated, **0 recovery-induced damage**,
0 rollbacks. Verification cost 0 tokens / ~8ms / ~20 DB reads per episode against ~4s and
~4,600 tokens for the agent.

- **Measured verifier blind spot: 3/120 (2.5%).** In `m22_cancel_refund_f2_wrong` across
  all three seeds, the injected wrong-target write cancelled an unrelated order while the
  mission's own assertions still passed, so Arm C returned VERIFIED. Under a stricter
  ground truth that treats any unauthorized write as failure, Arm C's false success rate
  would be 2.5%. Reported in both README and findings rather than left implicit.
- Variance across seeds is near zero because seeds vary the *data*, not the mission
  *structure*. This means the effect is robust to data variation; it is not evidence of a
  wide distribution having been sampled.

## 2026-08-16 — M6a sensitivity analysis (resolves the open decision)

The open question was whether out-of-frame mutations should fail the Arm C verdict. Rather
than pick one definition and lose the comparison, both are now supported and **both are
reported**: `verify(..., strict_frame=True)` selects the stricter lens, and `atl-rescore`
re-judges stored runs under each.

No re-run was needed. Every episode record carries the mission spec and a dump of the
database as the episode left it, and verification is pure, so all 120 episodes were
re-scored from disk at **zero API cost** — the first practical use of the replayability
property the recording format was designed for.

| Arm | Frame-scoped | Strict |
|-----|--------------|--------|
| A — agent self-report | 52.5% | 55.0% |
| B — tool responses | 40.0% | 42.5% |
| C — verifier (frame-scoped) | 0.0% | 2.5% |

- The headline gap is **robust to the definition** — A and B move ~2 points.
- 3 episodes (2.5%) flip VERIFIED to FAILED, all `m22_cancel_refund_f2_wrong`.
- **Arm C's strict column is the only non-circular measurement of the verifier here.**
  Because C is frame-scoped, judging it against strict truth uses an evaluator that differs
  from the truth judging it, which is exactly what the M4 circularity note said was missing.

The default remains frame-scoped, so the reported headline is the conservative one.

### Remaining known limitation

Seeds vary the data, not the mission structure, so cross-seed variance is near zero. Adding
structural variance (randomized archetype mix and injection assignment per seed) would need
a re-run and would break comparability with this run's numbers — a good M7, not a patch.
