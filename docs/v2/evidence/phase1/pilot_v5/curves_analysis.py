"""Exploratory (post hoc): P0 v5 pilot seeds 1-2 against two P0 v4 runs, from TensorBoard event files.

Input: p0v5-training-curves.tgz (sha256 a49936328a1a558b2a2d8de068484b88f037cf251506c92fe728d40bd94bbac6), made by
`python -m sim2gate.v2.phase1 curves` on Oct 21, 2026. Usage: python curves_analysis.py <unpacked folder>
All curves are smoothed with a trailing 20-round mean.
"""
import glob
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TRACK = "Episode_Reward/track_lin_vel_xy_exp"
ERROR = "Metrics/base_velocity/error_vel_xy"
ROUNDS = [50, 100, 150, 200, 300, 500, 800, 1000, 1500, 1999]
SHOW = [TRACK, ERROR, "Train/mean_episode_length", "Episode_Termination/base_contact",
        "Episode_Reward/dof_torques_l2", "Episode_Reward/dof_acc_l2", "Episode_Reward/action_rate_l2",
        "Episode_Reward/track_ang_vel_z_exp", "Policy/mean_std", "Sim2Gate/rho"]


def load(root):
    runs = {}
    for ev in sorted(glob.glob(os.path.join(root, "**", "events.out.tfevents*"), recursive=True)):
        rel = os.path.relpath(ev, root).split(os.sep)
        name = rel[0] if rel[0] != "p0v4_reference" else "p0v4_" + rel[1]
        ea = EventAccumulator(ev, size_guidance={"scalars": 0})
        ea.Reload()
        runs[name] = {t: dict((e.step, e.value) for e in ea.Scalars(t)) for t in ea.Tags()["scalars"]}
    return runs


def curve(run, tag, w=20):
    d = run.get(tag, {})
    x = np.array([d.get(i, np.nan) for i in range(2000)])
    with np.errstate(all="ignore"):
        return np.array([np.nanmean(x[max(0, i - w + 1): i + 1]) if np.isfinite(x[max(0, i - w + 1): i + 1]).any()
                         else np.nan for i in range(2000)])


def main(root):
    runs = load(root)
    names = sorted(runs)
    ref = [n for n in names if n.startswith("p0v4_")]
    refm = np.mean([curve(runs[n], TRACK) for n in ref], axis=0)
    for tag in SHOW:
        print(f"\n{tag}\nround " + " ".join(f"{n[-8:]:>9}" for n in names))
        cs = {n: curve(runs[n], tag) for n in names}
        for r in ROUNDS:
            print(f"{r:5d} " + " ".join(f"{cs[n][r]:9.3f}" for n in names))
    print()
    for n in names:
        tr, er = curve(runs[n], TRACK), curve(runs[n], ERROR)
        ratio = tr / refm
        f = lambda cond, lo=50: next((i for i in range(lo, 2000) if cond(i)), None)  # noqa: E731
        print(f"{n}: tracking below 70% of P0 v4 from round {f(lambda i: ratio[i] < 0.7)}, below 50% from "
              f"{f(lambda i: ratio[i] < 0.5)}; back to 90% at {f(lambda i: ratio[i] > 0.9, 300)}; "
              f"speed error first below 0.5 m/s at {f(lambda i: er[i] < 0.5, 0)}")


if __name__ == "__main__":
    main(sys.argv[1])
