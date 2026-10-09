# Experiment T, step 1: Go2 smoke test on a GPU machine

Goal: confirm that Sim2Gate's training monitor and BSRS trainer run inside Isaac Lab's own Go2 training, log
sensible signals and cost almost nothing, before the 20-policy grid. This is not data for Experiment T; the
grid waits until the MVP checks pass validation (see the MVP design plan).

Status of the code: tested on CPU with rsl-rl-lib 5.0.1, 5.4.1 and 5.5.1 (training is bit-identical with the monitor
on; ρ matches the RSL-RL acceptance gate to 1e-6). **Not yet run inside Isaac Lab.** Step 1 is that run.

## What you need

- A Linux GPU machine Isaac Sim supports: an RTX-class GPU with RT cores and at least 16 GB of memory, Ubuntu
  22.04 or 24.04. Data-center GPUs without RT cores (A100, H100) are not supported by Isaac Sim.
- Isaac Lab 3.0.0, which pins rsl-rl-lib 5.5.1, the version the acceptance gate used. (Isaac Lab 3.0 beta pins
  5.0.1, also gated. Isaac Lab 2.x uses rsl-rl-lib 3.x and will not work.)
- About an hour of GPU time.

### Recommended: Isaac Lab-Arena on an NVIDIA Brev GPU instance

The MVP is built on Isaac Lab-Arena, so run step 1 inside Arena's own container: the same environment then
serves training now and Arena evaluation later. Arena's container installs Isaac Sim 6.0.1 and Isaac Lab
3.0.0 with RSL-RL (rsl-rl-lib 5.4.1, gated) and Isaac Lab's tasks, Go2 included.

1. On brev.nvidia.com, create a GPU instance with an RTX-class GPU (L40S, RTX PRO 6000 or similar; not A100 or
   H100) and Docker with the NVIDIA container toolkit. Brev shows the hourly price before you deploy.
2. On the instance:
   ```bash
   git clone --branch main --recurse-submodules https://github.com/isaac-sim/IsaacLab-Arena.git
   cd IsaacLab-Arena && ./docker/run_docker.sh        # first build takes a while
   ```
3. Inside the container, Isaac Lab is at `submodules/IsaacLab` and Python is `/isaac-sim/python.sh`.
   Use these in place of `./isaaclab.sh -p` below, e.g.
   `/isaac-sim/python.sh -m pip install "sim2gate[rslrl] @ git+https://github.com/RootStep/sim2gate@experiment-t-step1"`.
   The plain-Isaac-Lab baseline in step 2 becomes
   `/isaac-sim/python.sh submodules/IsaacLab/scripts/reinforcement_learning/train.py --rl_library rsl_rl $ARGS`.
4. Stop the instance when done; Brev bills running instances by the hour.

Quicker but without Arena: Brev's Isaac Launchable (VS Code in the browser, Isaac Lab 3.0.0-beta2,
rsl-rl-lib 5.0.1, gated). It works for step 1 using the 3.0 beta commands, but Arena would have to be added later.

### Other clouds

Isaac Lab's Isaac Automator deploys Isaac Lab to AWS, GCP, Azure or Alibaba Cloud. On AWS, a g6e instance
(one NVIDIA L40S, 48 GB) fits: g6e.xlarge lists at about $1.86/hour on demand in us-east-1 (Oct 2026), so the
smoke test costs a few dollars. Stop the instance when done; a forgotten one costs about $1,360 a month.

## 1. Install Sim2Gate into Isaac Lab's Python

From the Isaac Lab folder:

```bash
./isaaclab.sh -p -m pip install "sim2gate[rslrl] @ git+https://github.com/RootStep/sim2gate@experiment-t-step1"
./isaaclab.sh -p -c "import sim2gate, rsl_rl, importlib.metadata as m; print(sim2gate.__version__, m.version('rsl-rl-lib'))"
```

The second line should print `0.1.0.dev0` and the rsl-rl-lib version (`5.4.1` in Arena's container, `5.5.1` on Isaac Lab 3.0.0). If pip tries to change rsl-rl-lib, stop and send the
output back.

## 2. Baseline timing: stock PPO, monitor off and on

Commands for Isaac Lab 3.0.0. On 3.0 beta, the task id is `Isaac-Velocity-Flat-Unitree-Go2-v0` and the launcher
needs `--isaaclab-script scripts/reinforcement_learning/rsl_rl/train.py`.

```bash
ARGS="--task Isaac-Velocity-Flat-UnitreeGo2 --headless --seed 1 --max_iterations 30"

./isaaclab.sh train --rl_library rsl_rl $ARGS                        # plain Isaac Lab
./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 0 -- $ARGS
```

Compare the iteration times Isaac Lab prints. The monitor adds one critic pass and three short recursions per
iteration; expect under 2% overhead. With `--eta 0`, the final rewards of the two runs should match closely
(GPU physics is not bit-deterministic, so exact equality is not expected here; the CPU tests check that).

## 3. BSRS run

```bash
./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 1 -- $ARGS
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
