# Sim2Gate

Training-aware diagnostics for sim-trained robot policies.

Sim2Gate looks at a reinforcement-learning policy trained in simulation, together with the
training setup that produced it, and reports why it is likely to fail on real hardware and what
to change in training. Every finding comes with the evidence that triggered it, a likely cause
and a concrete change to try.

> **Status: pre-alpha.** This repository reserves the project name. No diagnostics are
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

## Usage

```bash
pip install sim2gate
sim2gate preflight --help
```

The `preflight` command is a placeholder in this release.

## License

Apache 2.0. See [LICENSE](LICENSE).
