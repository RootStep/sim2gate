# `truth_rules`: proposal to confirm, for the S1 gate (2026-10-30)

**Status: PROPOSAL.** The reviewer must confirm these values, or change them in a new manifest revision, before S1 is approved. Until then they stay provisional, and no value may change after S1 starts.

**Recommendation:** confirm all ten values unchanged.

**No run data was used to choose or check any value.** S0 output is never analysed. No fall, tracking or slip quantity has been computed from it, and none will be computed before S2 completes.

Each rationale below is a priori: it rests on the robot's size, the control rate (step_dt = 0.02 s), the command ranges, and the design of the comparison. The last column states what could make the rule a poor fit, so the reviewer can weigh it.

## Primary failure definition

An episode fails if the robot falls, or if it loses velocity tracking.

| Rule | Value | Definition (`independent_truth.py`) | A-priori rationale | Risk if wrong |
|---|---|---|---|---|
| `fall_height_m` | 0.20 | Base (trunk-origin) world height below this… | The Go2 stands about 40 cm tall overall and about 20 cm when crouched ([manufacturer specs][go2]). A walking trunk origin sits well above 0.20 m; a robot lying on the ground sits well below it. The ground is flat at z = 0. | A policy that walks very crouched could count as fallen. This is not expected for velocity-tracking policies, and a false fall would hit controls too, because the test is relative. |
| `fall_tilt_deg` | 60 | …or the angle between the base z-axis and vertical above this… | Normal gaits tilt a few degrees, and 60° is close to lying on a side. It catches rollovers whose trunk stays high. | Low. |
| `fall_persist_steps` | 5 | …for this many consecutive control steps (0.1 s) | Rejects single-step transients such as a stumble or contact spike; real falls persist. | Low. A fall that recovers within 0.1 s is not counted, which is intended. |
| `tracking_warmup_steps` | 50 | Steps excluded at the start (1 s) | Each episode starts from rest at the reset pose, so acceleration to the command is not counted as a tracking failure. | Low. |
| `tracking_loss_mps` | 0.25 | Fails if the mean xy velocity error, in the base frame after warm-up, exceeds this | Commanded planar speeds are 0.30–1.04 m/s. A mean error of 0.25 m/s means missing the command by 24–83%, a functional failure rather than noise. | **Main uncertainty.** If healthy controls often exceed 0.25 m/s, their failure rate is high: the test stays valid, because it is relative to controls, but loses power (see the manifest's power table). It cannot be tuned now without peeking at data. |

## Slip (secondary magnitude; never decides truth)

| Rule | Value | Rationale |
|---|---|---|
| `contact_force_n` | 10 | A foot is in stance when the contact force exceeds 10 N. The Go2 weighs about 15 kg ([manufacturer specs][go2]), about 147 N, so a stance foot carries tens of newtons; 10 N is below a supporting foot's load and above contact noise.<br><br>S0 finding: in this build the sensor returns the **normal** force. On flat ground its z component is the vertical load, so the rule is valid as written. |
| `min_stance_steps` | 3 | A stance must last at least 60 ms; shorter contacts are touch-and-go. |
| `stance_trim_steps` | 1 | One step is dropped at each end of a stance, so touchdown impact and lift-off sliding are not counted as stance slip. |

## Decision rule

| Rule | Value | Rationale |
|---|---|---|
| `alpha` | 0.01 | One-sided Fisher exact test. With Bonferroni across a policy's two primary cells, each cell is tested at 0.005. |
| `min_rate_difference` | 0.10 | The excess failure rate must also be at least 10 points. This guards against tiny but significant differences on large control pools. |

The manifest's power table shows the consequence: with 128 episodes, an excess of 0.15 or more is detected with power of at least 0.90, and one at the 0.10 floor about half the time.

## Optional check (not proposed by default; needs reviewer permission)

One sanity check would test the two height and tracking rules without looking at any flawed policy.

**The check:** use the S0 control run (p0v5b_s101, N080) and report only these two distributions:
- the base-height distribution;
- the per-episode mean tracking error.

**Why it would not tune anything:**
- it uses a control in the nominal cell;
- it does not count falls or failures;
- it cannot inform any arm-versus-control comparison.

**What it costs:** it is still a look at run data before the rules are frozen, so whether to allow it is the reviewer's call. If allowed, the result would be reported, and any rule change would need a new manifest revision before S1.

[go2]: https://www.robotlab.com/store/unitree-go2-robot-dog/
