# Pre-freeze GPU checks (Oct 16, 2026, new instance `sim2gate-arena-2`)

Run on a freshly built instance (Isaac Lab-Arena 2cc9e97, Isaac Lab 91a0906, Sim2Gate 1e44c74; image
`811979b7…`), from bundle `s2g-prefreeze-2026-10-16.tgz` (SHA-256 `d96a5703…`). Nothing was trained. Files here are
copied unmodified; the raw measurements stay in the bundle.

## 1. The planted tasks build (standing robot, zero actions, 100 steps)

| Task | PhysX base height (m) | Newton base height (m) |
|---|---|---|
| P0 v4 | 0.28575 | 0.26811 |
| P1, P2, P3, P4 | 0.28575 each | not run |
| P5 (stiffness 1000, damping 60) | 0.19903 | 0.25896 |

P1–P4 build and stand like P0 (their changes act only in training). P5's ground differs between the engines in the
intended direction (PhysX sinks 8.7 cm against normal ground, Newton 0.9 cm), but far more than the ~1 cm the
definition intended. VJ chose to stiffen it (option A): a pilot stand test picks the softest stiffness whose PhysX
sink is at most 2 cm, before any training (`../p5-pilot/` once run).

## 2. Known-answer dry run of the blinded pipeline

`private.json` lists the 4 policies; `key.json` is this dry run's key (not the test set's). Prepare → run → seal →
replay, all on the GPU.

| Policy | ID | Expected | Verdict |
|---|---|---|---|
| P0 v4 seed 1 | B9KTU | go | go |
| P0 v4 seed 2 | BV71C | go | go |
| P0 v4 seed 3 | BT7YY | go | go |
| P0 v2 seed 1 (no delay in training) | BSUBF | no-go, latency | no-go: latency cliff 20 ms, plus actuator limits (non-target) |

Replays (seeded draw B9KTU, BSUBF, BV71C): every verdict and risk list reproduced. `verdicts.sha256` matches the
verdict files. No blinded record (session log, manifest, aliases, verdicts) names a task or policy.

## 3. The rebuilt instance reproduces the calibration

For P0 v4 seeds 1–3: every check 2 metric, check 1's nominal success and every check 1 cliff equal the calibrated
values (`calibration/p0v4`). Check 5's Newton runs vary slightly between runs: one of 64 episodes can flip between
falling and not (fall-free 0.984 ↔ 1.0), and the error ratio moved by up to 0.02 (seed 2: 1.4461 against 1.4665);
the replay of seed 2 also had one Newton fall. PhysX results were identical. No verdict changed; the margins to the
check 5 flags (fall-free 0.8, ratio 2.0) are far larger than this run-to-run spread.
