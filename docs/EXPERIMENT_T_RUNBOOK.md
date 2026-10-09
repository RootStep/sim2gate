# Experiment T, step 1: Go2 smoke test on a GPU machine

Goal: confirm that Sim2Gate's training monitor and BSRS trainer run inside Isaac Lab's own Go2 training, log
sensible signals and cost almost nothing, before the 20-policy grid. This is not data for Experiment T; the
grid waits until the MVP checks pass validation (see the MVP design plan).

Status of the code: tested on CPU with rsl-rl-lib 5.0.1 and 5.5.1 (training is bit-identical with the monitor
on; ρ matches the RSL-RL acceptance gate to 1e-6). **Not yet run inside Isaac Lab.** Step 1 is that run.

## What you need

- Linux with an NVIDIA RTX GPU that Isaac Sim supports, and Isaac Lab 3.x installed (3.0 beta pins
  rsl-rl-lib 5.0.1, which is gated; Isaac Lab 2.x uses rsl-rl-lib 3.x and will not work)
- About an hour

## 1. Install Sim2Gate into Isaac Lab's Python

From the Isaac Lab folder:

```bash
./isaaclab.sh -p -m pip install "sim2gate[rslrl] @ git+https://github.com/RootStep/sim2gate@experiment-t-step1"
./isaaclab.sh -p -c "import sim2gate, rsl_rl, importlib.metadata as m; print(sim2gate.__version__, m.version('rsl-rl-lib'))"
```

The second line should print `0.1.0.dev0` and `5.0.1` (or `5.5.1`).

## 2. Baseline timing: stock PPO, monitor off and on

Isaac Lab 3.0 beta task ids end in `-v0`; on Isaac Lab main use `Isaac-Velocity-Flat-UnitreeGo2` and drop
`--isaaclab-script`.

```bash
T=scripts/reinforcement_learning/rsl_rl/train.py
ARGS="--task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --seed 1 --max_iterations 30"

./isaaclab.sh -p $T $ARGS                                                        # plain Isaac Lab
./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 0 --isaaclab-script $T -- $ARGS
```

Compare the iteration times Isaac Lab prints. The monitor adds one critic pass and three short recursions per
iteration; expect under 2% overhead. With `--eta 0`, the final rewards of the two runs should match closely
(GPU physics is not bit-deterministic, so exact equality is not expected here; the CPU tests check that).

## 3. BSRS run

```bash
./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 1 --isaaclab-script $T -- $ARGS
```

## 4. Check the signals

Each run writes `logs/rsl_rl/unitree_go2_flat/<run>/sim2gate_signals.jsonl`; the scalars also appear in
TensorBoard under `Sim2Gate/`.

```bash
./isaaclab.sh -p -m sim2gate.cli signals logs/rsl_rl/unitree_go2_flat/*/sim2gate_signals.jsonl
```

Pass conditions for step 1:

| Check | Expected |
|---|---|
| One JSON line per iteration | 30 lines |
| `signals.num_samples` | num_envs × 24 (98,304 at Isaac Lab's default 4,096 envs) |
| `signals.identity_residual` | tiny relative to the advantage scale (float32 rounding) |
| `signals.rho` and `predicted_rho` | equal to about 1e-6 |
| `eta_is_training_eta` | false for `--eta 0` (what-if at η = 1), true for `--eta 1` |
| `reward_terms` | Go2's reward terms (track_lin_vel_xy_exp, track_ang_vel_z_exp, ...) with shares summing to 1 |
| `signals.timeouts`, `terminations` | both above 0 over the run |

Send the two JSONL files and the console logs back; they decide whether the grid can start.

## 5. Before the grid (not part of step 1)

- Freeze the Experiment T plan (arms η = 0, 0.5, 1, 2 × seeds 1 to 5, iterations, outcome, analysis) in
  writing, dated, before the first grid run. If the same runs will also go into the BSRS paper, freeze the
  paper's plan at the same time.
- Grid: `for eta in 0 0.5 1 2; do for s in 1 2 3 4 5; do ... --eta $eta ... --seed $s; done; done`
- Summary per policy: `sim2gate signals --last 0.2` gives mean 1 − ρ and reversal share over the last 20%
  of training, the plan's training-signal summary.
