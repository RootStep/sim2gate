# Evidence for plan change C6 (Oct 14, 2026)

P0 v2 seeds 1–3 under checks 1 and 5, plus the Newton diagnostics. Copied unmodified from the run bundle
`s2g-c15-2026-10-14.tgz` (SHA-256 `e0a1ade13ca75ef3caab30df87b4f4ef5a4686a98d4f0e9a942cbd30aef6c65c`, 50 files, all
matching its SHA256SUMS), which also holds the raw episode files and logs. Sim2Gate `9ca5494`.

| File | What it shows |
|---|---|
| `check1_s{1,2,3}.json` | Check 1 on P0 v2: latency cliffs at 20–25 ms (inside the plausible 0–30 ms) on every seed; friction success 0.81 at 0.4 |
| `check5_s{1,2,3}.json` | Check 5 with matching joint order, scored under the plan's original rule: flagged on Newton episode success 55–72%, with no falls and an error ratio of 1.34–1.48 |
| `success.json` | Episode-success reference from the three sweeps' nominal episodes (0.1078 m/s) |
| `robot_isaacsim_physx.json`, `robot_newton_mjwarp.json` | The robot read back from each engine: joint order by type on PhysX, leg by leg on MuJoCo-Warp |
| `diag_newton_{native,lab}_s1.json` | Before the ordering fix: P0 v2 seed 1 falls constantly on Newton with either motor path |
| `diag_armature_{default,newton_mjwarp}_s1.json` | Before the ordering fix: the joint armature preset does not explain it |
| `diag_newtontrained_on_{newton_mjwarp,isaacsim_physx}.json` | A policy trained on Newton walks on Newton (0 falls) and falls on PhysX (101) without the ordering fix |

Re-scored with C6's rule (Newton fall-free rate at least 80%, error ratio at most 2), check 5 passes all three
P0 v2 seeds: fall-free 1.0, ratios 1.476, 1.452 and 1.341.
