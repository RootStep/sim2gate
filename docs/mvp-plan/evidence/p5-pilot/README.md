# P5 stiffness pilot and joint-sweep GPU test (Oct 16, 2026, `sim2gate-arena-2`)

Bundle `s2g-p5pilot-2026-10-16.tgz` (SHA-256 `cde6baec…`), Sim2Gate `4a558b0`. Nothing was trained.

## P5 stiffness pilot

A robot stands with zero actions for 100 control steps on P5's ground (PhysX compliant contact, acceleration
spring, damping 2√k) at each stiffness, on PhysX and on Newton (`stand_K<k>_<engine>.json`). Sink is measured against
P0's normal ground on the same engine (PhysX 0.28575 m, Newton 0.26811 m; `../pre-freeze/`).

| Stiffness | PhysX height (m) | PhysX sink | Newton height (m) | Newton sink | Gap, PhysX vs Newton |
|---|---|---|---|---|---|
| 1000 (Oct 16 morning) | 0.19903 | 8.67 cm | 0.25896 | 0.92 cm | 6.0 cm |
| 2000 | 0.23228 | 5.35 cm | 0.26364 | 0.45 cm | 3.1 cm |
| 3000 | 0.24356 | 4.22 cm | 0.26496 | 0.32 cm | 2.1 cm |
| 4000 | 0.25006 | 3.57 cm | 0.26545 | 0.27 cm | 1.5 cm |
| 5000 | 0.25596 | 2.98 cm | 0.26573 | 0.24 cm | 1.0 cm |
| 6000 | 0.26028 | 2.55 cm | 0.26589 | 0.22 cm | 0.6 cm |
| 8000 | 0.26343 | 2.23 cm | 0.26624 | 0.19 cm | 0.3 cm |
| 10000 | 0.26465 | 2.11 cm | 0.26657 | 0.15 cm | 0.2 cm |

Normal ground: PhysX 0.28575, Newton 0.26811, a 1.8 cm gap with Newton lower. The rule set before the pilot ("the
softest stiffness whose PhysX sink is at most 2 cm") could not be met in the range, and from about 4000 the engine gap
is smaller than normal ground's. VJ chose **2000**, fallback **3000** (`../../test-set.md`).

Four PhysX runs (2000, 5000, 6000, 10000) exited 1: each wrote its result first, then Isaac Sim crashed in
`SimulationApp.close()` (the shutdown crash seen since C5).

## Joint sweep, first GPU run

`joint_smoke.json`: P0 v4 seed 1 (`db9a4755…`), task P0 v4, PhysX, 2,944 episodes. Every episode's applied friction,
mass, motor strength, latency and noise read back exactly as drawn. Summary (`joint_smoke_report.json`, scored with
the frozen success reference): 74.4% success over the 504 episodes with every condition realistic at once (one at a
time, the same policy stays at 91–98% across every realistic range), 33.4% when any condition is beyond its range.
Worst pairs (both in the harder half of their realistic range, against the product expected if independent): mass +
motor strength 47% vs 60%, latency + noise 67% vs 82%, friction + latency 74% vs 83%. Report only; recorded as a
version 2 candidate (`docs/roadmap/v2-candidates.md`).
