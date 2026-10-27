# Prior-art and benchmark-overlap watch

**First check: 2026-10-27. Status: working evidence register; NOT an exhaustive systematic literature review.** Check at every recipe, metric, claim, and protocol-freeze review. Prefer official docs, released code and peer-reviewed papers; distinguish published evidence from inferred gaps.

| Source | Verified overlapping capability | RootStep v0.1 implication | Source |
|---|---|---|---|
| NVIDIA Isaac Lab-Arena v0.3 | Composable evaluation, controlled variation sweeps, subtask predicates, success-conditioned joint-factor sensitivity analysis, parallel/multi-node execution | Do not claim novelty for perturbation sweeps, locating a failing task stage, or sensitivity/robustness analysis. Implement a faithful Arena-style sensitivity baseline; investigate integration rather than competing runtime. | https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/concepts/concept_sensitivity_analysis.html ; https://github.com/isaac-sim/IsaacLab-Arena |
| RoboGauge, Wu et al., RSS 2026 | Unitree Go2 sim-to-sim transfer assessment, multidimensional tracking/safety/quality metrics, multiple terrains and difficulty levels, policy ranking/selection | High direct overlap with Go2 robustness, cross-simulator checks and policy-quality scores. Include as named comparator or document infeasibility; do not claim Go2 sim-to-sim evaluation itself is novel. | https://www.roboticsproceedings.org/rss22/p156.html ; https://github.com/wty-yy/RoboGauge |
| RoboGauge Isaac Lab/RobotLab reproduction | Published RoboGauge score comparisons on Go2 models, including tracking/safety/quality | Consider portability and direct benchmark comparator before claiming unique evaluation coverage. | https://github.com/wertyuilife2/go2_rl_robotlab |
| Joint-failure adaptation, Wu et al., PLOS One 2026 | Quadruped actuator/joint failure perturbations, RL adaptation and robustness outcomes | Fault injection and actuator failure tests are not new in themselves; distinguish training-recipe diagnosis from controller fault adaptation. | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0361099 |

## Preliminary differentiation hypothesis — unproven

The **benchmark design** (not individual stress metrics) may add a controlled cohort of healthy and deliberately intervention-flawed policies with precommitted labels and independent behavioral manifestation truth, evaluating whether training-aware Sim2Gate diagnostics improve **flaw detection, intervention/cause attribution and failure-region localization** over nominal success, raw signals, Arena-style sensitivity and RoboGauge-style scores at comparable information/compute budgets. Do **not** assert no prior benchmark does this until systematic search/code review; do not equate intervention labels with causal ground truth.

## Immediate design corrections
1. Add RoboGauge as an explicit comparator candidate, not just Arena-style sensitivity; determine runnable version, license, Go2 policy adapter, terrain compatibility and evaluation budget.
2. Arena already advertises 'why policies fail' through sensitivity and 'where' through subtask predicates. Claiming unique failure diagnosis requires a **different operational endpoint** and head-to-head evidence.
3. Arena's joint-factor sensitivity analysis should not be weakened into one-factor-at-a-time sweeps as the sole comparator. Compare against the documented posterior workflow when feasible.
4. RoboGauge's sim-to-sim Go2 policy ranking is substantial prior art. PhysX/Newton discrepancy and locomotion score alone are not sufficient novelty claims.
5. Before freezing each flaw recipe, search for its equivalent in Arena, RoboGauge and papers; log source URL, version/commit, date, overlapping metrics, testable differentiator and baseline decision.
6. Recheck main versus release branches: Arena warns main is in-development and release/0.3.1 is more stable. Verify version-specific support rather than assuming Go2 locomotion availability or absence.

## Change-log procedure
For each meaningful discovery record: checked date; exact source and version; what is already implemented; what remains unknown; affected RootStep claim; baseline change; recipe change; reproducibility cost; reviewer signoff. Recheck before pilot, before freeze, before submission, and on new Arena/RoboGauge releases.

**No recurring alert is implied by this repository document.**
