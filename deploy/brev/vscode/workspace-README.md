# Sim2Gate on Isaac Lab Arena (RootStep launchable)

## Sim2Gate: Go2 smoke test in one command

```console
cd /workspaces/isaaclab_arena
python -m sim2gate.training.smoke_go2
```

It trains Isaac Lab's Go2 flat-ground task three times (stock Isaac Lab, Sim2Gate attached at eta 0, and BSRS at
eta 1; about 30 iterations each, a few minutes in total after the first launch), checks the runbook's pass
conditions and prints PASS or FAIL. Results go to `/workspaces/isaaclab_arena/sim2gate_smoke/<time>/`.
Versions installed on this instance: `cat ~/sim2gate-versions.txt` on the host.

---

# Isaac Lab Arena Launchable

Isaac Sim 6.0.1 + Isaac Lab 3.0 + **Isaac Lab Arena**, with browser-based VS Code and a
streaming Isaac Sim viewport. Everything is already installed — nothing to build here.

The Arena repository is mounted at `/workspaces/isaaclab_arena`. Edits you make in this
editor are edits to the clone on the host, and they take effect immediately (Arena is
installed in editable mode).

Inside this container `python` is an alias for `/isaac-sim/python.sh`, so use `python`
(not `python3`) for everything below.

## Web Viewer for the Isaac Sim UI

The viewer is always running, but it shows "Waiting for stream..." until you start a
simulation. Nothing auto-starts — run one of the commands below first.

### To view in a separate tab
1. Run a command that starts Isaac Sim (see below).
2. Copy your current URL (ie `https://isaac.brevlab-1234`).
3. Open a new tab and paste it with `/viewer/` appended (ie `https://isaac.brevlab-1234/viewer/`).
4. The Isaac Sim UI appears once the app is ready.

### To view inside VS Code
1. Run a command that starts Isaac Sim (see below).
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac).
3. Type "Simple Browser: Show".
4. Enter URL: `/viewer/`

> Only one viewer tab at a time is supported.

## Running an evaluation

```console
cd /workspaces/isaaclab_arena
```

### Headless smoke test (no viewer, fastest way to confirm the stack works)

```console
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps 20 cube_goal_pose
```

### Streamed to the web viewer

```console
python isaaclab_arena/evaluation/policy_runner.py \
    --livestream 2 --viz kit \
    --policy_type zero_action --num_steps 200 cube_goal_pose
```

`--livestream 2` selects WebRTC streaming and `--viz kit` selects the Kit visualizer.
You **must** pass both to get a picture in the viewer. Append them to any Arena command.

### A scene with objects and a background

```console
python isaaclab_arena/evaluation/policy_runner.py \
    --livestream 2 --viz kit \
    --policy_type zero_action \
    --num_steps 2000 \
    pick_and_place_maple_table \
    --embodiment droid_rel_joint_pos \
    --pick_up_object rubiks_cube_hot3d_robolab \
    --destination_location bowl_ycb_robolab \
    --hdr home_office_robolab
```

Note the argument order: runner and policy flags come *before* the environment name,
environment variations come *after* it. `--list_variations` prints what a given
environment accepts.

Keep `--num_steps` generous when you want to watch. Kit takes a minute or two to start
and the WebRTC stream only becomes available after that; a short run can finish before
you have connected. `zero_action` sends no motion, so the scene will be static — it is
for confirming the scene and the stream, not the robot.

### Experiment runner (the recommended entry point)

Runs a YAML-defined sweep — several configurations of the same task — and writes a
report you can browse.

```console
python isaaclab_arena/evaluation/experiment_runner.py \
    --livestream 2 --viz kit \
    --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml
```

Results land in `outputs/<timestamp>/`, with `arena_experiment_result.json` and an
`index.html` report. Add `--record_camera_video --serve_evaluation_report` to capture
video and serve the report.

Override any config value on the command line:

```console
python isaaclab_arena/evaluation/experiment_runner.py \
    --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml \
    shared.rollout_limit.num_episodes=4 \
    runs.parallel_envs.environment_builder.num_envs=8
```

### Explore an environment interactively (no policy)

```console
python isaaclab_arena/scripts/environment_runner.py \
    --viz kit --livestream 2 \
    pick_and_place_maple_table --embodiment droid_rel_joint_pos
```

The interactive runner requires `--viz kit`, `--num_envs 1` and `--device cpu`.

> **Argument order matters**: policy flags (`--policy_type`, `--checkpoint_path`, ...)
> must come *before* the positional environment name.

## Verifying the installation

```console
cd /workspaces/isaaclab_arena
pytest -sv -m "not with_cameras and not with_subprocess" isaaclab_arena/tests/
pytest -sv -m "with_cameras and not with_subprocess" isaaclab_arena/tests/
pytest -sv -m with_subprocess isaaclab_arena/tests/
```

## Next steps

- **Trained RL policies** — `--policy_type rsl_rl --checkpoint_path /models/<run>/model_1999.pt`.
  Put checkpoints in `/models` (mounted from `~/models` on the host); `/datasets` and
  `/eval` are mounted the same way. Never commit models or datasets to the repo.
- **GR00T policies** — start a server, then evaluate against it:
  `python isaaclab_arena/evaluation/policy_runner.py --policy_type gr00t_remote_closedloop ...`
  See `docs/` and `skills/user/serve-gr00t-policy/`.
- **OpenPI (pi0 / pi05) policies** — `./isaaclab_arena_openpi/docker/run_openpi_server.sh`,
  then `--policy_type pi0_remote`. See `skills/user/serve-openpi-policy/`.
- **Full documentation** — `docs/pages/quickstart/` in this repo, and the skills under
  `skills/user/` which are the most operationally precise guides available.

## If the viewer says "Waiting for stream..."

That is the normal idle state - it clears once an app is streaming. If it persists after
Isaac Sim has fully started, check that Kit bound its signaling port:

```console
ss -tulpn | grep kit
```

Expect a listener on **8011**, which nginx proxies at `/sign_in`. A different port there
means this launchable's `nginx.conf` needs updating to match.

## Notes and limits

- **cuRobo** (the `ik_reachable` reachability check) is only present if this launchable
  was built with `INSTALL_CUROBO=true`. Check with
  `python -c "import curobo"`.
- The first Isaac Sim launch compiles shaders and takes several minutes. Later launches
  are fast — the caches are on Docker volumes and survive container restarts.
- Isaac Sim holds the GPU while it runs. Stop one evaluation before starting another.
