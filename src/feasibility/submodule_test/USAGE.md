# submodule_test — quick usage

Regression harness for **submodule bumps**. Replays a fixed set of scenarios in both simulators
and compares against a committed baseline. Full rationale is in each module's top docstring.

## When to run it

After `git submodule update --remote` (or any ostrich / helhest_stack pointer change), **before**
you commit the new pointer.

## The three commands

```bash
# 1. after a bump — the usual one. Exits non-zero if something changed.
python src/feasibility/submodule_test/run_check.py +tier=smoke

# 2. before trusting a pointer update — longer, more trials, extra terrains.
python src/feasibility/submodule_test/run_check.py +tier=full

# 3. accept a change you have looked at and decided is fine. Commit the .npz it writes.
python src/feasibility/submodule_test/run_check.py +tier=smoke +update=true
```

Other flags: `+scenarios=[box,rough]`, `+repeat=3` (report run-to-run spread, no comparison),
`+device=cuda:0`, `+mu=`, `+k_turn=`, `+no_h5=true`.

Self-tests, no GPU: `python src/feasibility/submodule_test/scenarios.py`,
`python src/feasibility/submodule_test/metrics.py`.

## Reading the report

| line | meaning | what to do |
|---|---|---|
| `submodule_test smoke: OK` | passed | nothing |
| `health: N FLAG(S)` | a rollout is not physically usable (NaN, off the map, teleport, speed spike, hstack residual) | **always investigate** — a baseline update cannot accept these |
| `<metric> CHANGED on k/n trial(s)` | a value the harness trusts moved outside tolerance | look at it, then either fix or `+update=true` |
| `noise-dominated metric(s) moved` | moved, but ostrich scatter makes the verdict a coin flip | informational — read it, don't act on it alone |
| `metrics not asserted` | ostrich scatter exceeds the floor for those columns | informational; they never fail |

`h_*` metrics are helhest_stack and are **bit-exact** — any `h_*` CHANGED is real, no matter how
small. `o_*` are ostrich and are intrinsically noisy (tens of degrees of yaw scatter between
identical runs), so only its well-conditioned cells are asserted.

**A green run means:** nothing blew up, helhest_stack is unchanged, ostrich did not change
grossly. It does *not* mean ostrich is unchanged — that is what the "moved" lines are for.

## Looking at a flagged run

Every run also writes `outputs/submodule_test_<scenario>.h5` in the usual comparator schema:

```bash
python src/feasibility/replay/gl_replay.py --file outputs/submodule_test_box.h5 --id 1 --which both
python src/feasibility/plotting/batch_comparator_viewer.py --file outputs/submodule_test_box.h5 --id 1
```

## Baselines

`baselines/baseline_{smoke,full}.npz`, tracked in git. They record what the two sims did on the
day they were written — no notion of "correct". Each stores the ostrich/helhest_stack SHAs and
dirty flags it was recorded at, so a baseline is self-identifying. When you `+update`, say which
submodule bump you are accepting in the commit message.
