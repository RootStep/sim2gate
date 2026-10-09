# Sim2Gate

Training-aware diagnostics for sim-trained robot policies.

Sim2Gate looks at a reinforcement-learning policy trained in simulation, together with the
training setup that produced it, and reports why it is likely to fail on real hardware and what
to change in training. Every finding comes with the evidence that triggered it, a likely cause
and a concrete change to try.

> **Status: pre-alpha.** The training-signal logger below works; the hardware-readiness checks are not
> implemented yet, and nothing here has been validated against real hardware.

## What it will do

The first release targets quadruped locomotion policies trained with Isaac Lab and RSL-RL. It
reads evaluation results from [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) and
adds checks on the training recipe itself:

| Check | Looks for |
| --- | --- |
| Simulator exploits | Behavior that only works because of simulator artifacts: foot sliding, ground penetration, contact-force spikes, torques or velocities past actuator limits |
| Observation reliance | Dependence on signals that are exact in simulation but noisy or missing on hardware, such as ground-truth base velocity |

Reward attribution and training-history checks come later.

## What a report says

Each run ends in one of three findings, not a single score:

- **Risk found:** a specific failure risk, with evidence, likely cause and suggested change
- **Nothing found:** no failure detected within the tested conditions
- **Insufficient evidence:** the inputs can't support a call

A report informs a hardware decision. It does not replace a team's own safety checks.

## Training signals (experimental)

`sim2gate.training` logs signals from inside RSL-RL training, the part of the recipe that evaluation alone
can't see. For each PPO iteration it records, from the rollout PPO is about to train on:

- **ρ**, the correlation between the raw and the value-shaped advantage over the normalization group, and its
  regime (cancelled, partial, reversed). From the BSRS-in-PPO research behind Sim2Gate: ρ, computed before the
  update, predicted how much same-critic shaping changed PPO's gradient, optimizer step and policy.
- the zero-tail fraction (how much of the rollout saw no reward ahead of it)
- each reward term's share of the episode return, from Isaac Lab's `Episode_Reward/*` logs
- action std and learning rate

```python
from sim2gate.training import TrainingMonitor
monitor = TrainingMonitor(runner, "sim2gate_signals.jsonl")   # runner: rsl_rl OnPolicyRunner
runner.learn(1500)
```

The monitor is read-only: training with and without it gives bit-identical parameters (tested). Its math
matches a reference implementation that passed an acceptance gate against rsl-rl-lib 5.0.1 and 5.5.1, down
to the actor gradient. With Isaac Lab 3.x, use the launcher, which leaves Isaac Lab untouched:

```bash
pip install "sim2gate[rslrl]"
./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 0 -- \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless
```

`--eta` above 0 trains with same-critic BSRS (`sim2gate.training.BSRSPPO`), for research runs.
See [docs/EXPERIMENT_T_RUNBOOK.md](docs/EXPERIMENT_T_RUNBOOK.md).

## Usage

```bash
pip install sim2gate
sim2gate preflight --help
```

The `preflight` command is a placeholder in this release.

## License

Apache 2.0. See [LICENSE](LICENSE).
