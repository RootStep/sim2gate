# v2 phase 2, part A: arms, pilots and leak audit (frozen)

Approved by VJ on Oct 23, 2026: part A as drafted, revision 3 (R-hop via `feet_air_time`; controls 101–105 as the hop
reference), and the details he asked to be recorded before freezing. Those details are:

- hopping treated as a hypothesis;
- a policy-level hop reference;
- numerical acceptance rules;
- the wider leakage safeguard;
- dropped arms reported, with their effect on the analysis.

This file is frozen with its SHA-256 in `PHASE2_FREEZE.md` before any pilot robot is trained. Part B is drafted in
`PHASE2_PROTOCOL.md` and frozen separately. Sections are numbered as in that file.

## 2. Arms (part A) ★

All arms start from P0 v5b (2,000 rounds, everything as phase 1) and change one thing.

**Same reward terms everywhere (decision D1, approved in principle).** Every arm has the same reward-term names, in the
same order, logged in the same format:

- reward arms **weaken or strengthen** an existing P0 v5b term, and no arm adds or removes a term;
- arms differ only in reward weights or physics.

**Change in revision 3 (confirmed by VJ, Oct 23).** As approved, part A could not be
carried out:

- the controls are phase 1 robots, trained without a body-height term;
- the body-height term would be "present in every arm";
- so the controls' term list would differ from every other arm's, the same leak D1 closes.

The fix keeps the controls and the identical term set: **R-hop strengthens the existing air-time reward
(`feet_air_time`)** instead of adding a height term. A too-large air-time reward is a common real reward-design
mistake, and its expected effect is long flight phases, that is, hopping or bounding. The hypothesis is checked in the
pilots like every other.

No arm sets a weight to zero. The stock recipe's `dof_pos_limits` term has weight 0 in P0 v5b and therefore in every
arm, including the controls. Its features are the same in every robot and carry no arm information, and section 7's
zero-variance rule drops them.

| Arm | Change from P0 v5b | Planted cause (hypothesis) | Intended symptom | Candidate settings, tried in order |
|---|---|---|---|---|
| Control | None: phase 1 seeds 101–105 | none | none | — |
| R-slip | Foot-slip penalty (`feet_slide`) weight × k | Reward | Foot slip | k = 0.1, then 0.03 |
| Φ-slip | Training ground friction changed; slip penalty kept | Physics | Foot slip | low only (static 0.2–0.5), then high only (static 1.0–1.2); dynamic = 0.75 × static |
| R-jitter | Action-rate penalty (`action_rate_l2`) weight × k | Reward | Jitter | k = 0.1, then 0.03 |
| Φ-jitter | No command delay in training (as v1's P2) | Physics | Jitter | the single setting |
| R-hop | Air-time reward (`feet_air_time`) weight × m | Reward | Abnormal hopping (section 3) | m = 10, then 30 |

Base weights in P0 v5b (from its saved config):

| Term | Base weight |
|---|---|
| `feet_slide` | −0.25 |
| `action_rate_l2` | −0.01 |
| `feet_air_time` | +0.25 |

The friction change sets static friction as given and dynamic friction to 0.75 × static, as the task does.

The code is in `sim2gate.v2.phase2_arms`, with one registered task per candidate,
`Sim2Gate-Velocity-Flat-Go2-P2-<arm>-<candidate>`. Every pilot run's saved config is checked against its candidate,
and a mismatch is a problem, not a pilot result.

## 3. Pilots and leak audit (part A)

**Seeds.** Pilot robots are development robots: excluded from all scoring, and never study robots or test robots.
Each candidate has fixed seeds, whatever earlier pilots show:

| Arm | First candidate | Second candidate |
|---|---|---|
| R-slip | 201–202 | 203–204 |
| Φ-slip | 205–206 | 207–208 |
| R-jitter | 209–210 | 211–212 |
| Φ-jitter | 213–214 | — |
| R-hop | 215–216 | 217–218 |

A smoke test comes first. It trains each arm's first candidate for 2 rounds (seed 99) to check the saved config, and
re-measures one control on check 2 to check that the hop measures are written.

**Measured.** Pilots are judged on their own measurements only:

- nominal PhysX success over 256 episodes (v2 threshold);
- check 2;
- the hop measures below.

**Intended symptoms (numerical, fixed before any pilot).** Slip and jitter are tied to the frozen check 2 limit
(1.0541): the arm's own metric, on its own, must be large enough to be flagged.

| Symptom | Present when (per robot) |
|---|---|
| Slip | Foot slip p95 > 1.0541 × 0.1124 m/s = **0.1185 m/s** (the clean maximum times the frozen limit) |
| Jitter | Jitter share > 1.0541 × 0.00935 = **0.00986** |
| Abnormal hopping | **Both** hop measures must agree: flight share at low speed ≥ max(1.5 × the controls' maximum, **0.02**) **and** vertical-speed RMS at low speed ≥ **1.5 ×** the controls' maximum |

Hopping uses a factor of 1.5 rather than 1.0541 because its reference comes from 5 robots, not 20. The floor of 0.02
(2% of low-speed steps airborne) is there because a walking gait's low-speed flight share may be zero, which would make
any flight at all count.

**The hop reference, defined.**

- **Which robots:** the 5 control robots (101–105) are re-measured on check 2 with the hop measures.
- **Same conditions:** each is run under the same fixed command set as every pilot, and low speed means the same
  commands for every robot (commanded speed at most 0.5 m/s).
- **Policy-level statistics:** each control gives two numbers, its flight share and its vertical-speed RMS over all of
  its own low-speed steps.
- **The reference:** the maximum of those five numbers for each measure. It is never the largest single sample.

**The hop measures.**

- **Flight share at low commanded speed:** the share of control steps with no foot in contact (force above check 2's
  contact threshold), over environments still running, during commands whose speed is at most 0.5 m/s. At low speed a
  normal walking gait has no flight phase, so running gaits do not count.
- **Base vertical-speed RMS:** over the same steps.

Both are new check 2 outputs. In check 2's own verdict they are **reported only**: they do not change its four scored
metrics, its limit or its measurement version. Using them to confirm phase 2's hopping arm is a separate purpose,
defined here.

**Hopping is a hypothesis.** A stronger air-time reward may produce hopping, exaggerated steps or some other gait. The
pilot counts only the predefined symptom above, together with acceptable walking and speed tracking: qualification
requires at least 85% success over 256 episodes under the v2 threshold, and success includes tracking error. Any other
gait is reported as observed and does not count as the symptom.

**Stopping rules.**

- **A candidate passes** only when **both** of its pilot seeds show the intended symptom (thresholds above) **and**
  both qualify (at least 85% over 256 episodes, v2 threshold). Its saved config must also match the candidate, and its
  record must have no problems.
- **The arm's setting** is the first candidate in the table's order that passes. Later candidates are not run.
- **If a candidate fails,** the next candidate is tried on its own fixed seeds.
- **If all candidates fail,** the arm is dropped and its hypothesis is reported as not supported. No other arm changes
  because of it.
- **Every pilot is reported,** failures included.
- **A dropped arm is reported explicitly,** together with its effect on the acceptance analysis:
  - With one flawed arm dropped, attribution has 5 classes instead of 6. Chance accuracy becomes 1/5 (from 1/6), and
    C2 and C3 are scored on the remaining arms, with the reduced design stated beside each result.
  - With two or more flawed arms dropped, C2 and C3 are reported as not tested.

**Leak audit.** Matching term lists close only the missing-term leak. A 0.1×, 10× or 30× weight can still show
through weighted contributions. So the restrictions for B and C stay as agreed: no reward weights, no configs, no
weighted contributions in the primary inputs, and no directly identifying log fields. The audit checks them on the
pilot robots before part B is frozen, and passes only if all of these hold:

1. Every pilot's exported training record has identical reward-term names, order and format (an automated check).
2. No primary-input feature (section 5) is constant within every arm while differing between arms. Any such feature
   would be a configuration value, not a measurement. Any found is removed and reported.
3. A "trivial-clue" classifier is run (nearest centroid, as in section 7) on the per-term ratio of weighted to
   unweighted reward, which recovers the weights. Its accuracy is reported. This shows how much the weighted inputs
   would leak, and why they are kept out of the primary set.

**Pilot cost.** About 45 minutes per robot, plus about 30 minutes for the control re-measure. One pass is 10 robots,
about 8 GPU hours; each failed candidate adds about 1.5 hours. This is outside the plan's 24 hours for phase 2.
