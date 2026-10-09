# Sim2Gate on Isaac Lab-Arena: RootStep's NVIDIA Brev launchable

One-click GPU workspace with Isaac Sim 6.0.1, Isaac Lab 3.0.0, Isaac Lab-Arena (pinned commit 2cc9e97, rsl-rl-lib
5.4.1) and Sim2Gate preinstalled, plus VS Code in the browser. Adapted from the community
[IsaacLab-Arena-launchable](https://github.com/dorperetz/IsaacLab-Arena-launchable) (see NOTICE), with the fixes
found during the Experiment T step 1 run on Oct 9, 2026.

## Create the launchable (once, in the Brev console)

1. brev.nvidia.com → Launchables → Create.
2. **Compute:** AWS g6e.2xlarge (1× L40S, 8 vCPU, 64 GiB). Isaac Sim needs a GPU with RT cores: L40S, L4 or
   A10G class work; A100 and H100 do not. AWS is the tested provider.
3. **Storage:** 300 GiB (the Arena image alone needs most of 250).
4. **Container:** VM Mode → "Basic VM with Python installed", "I don't have any code files", no Jupyter.
5. **Setup script:** paste `setup.sh`. Until the sim2gate pull requests are merged, change
   `SIM2GATE_REF="${SIM2GATE_REF:-main}"` to `SIM2GATE_REF="${SIM2GATE_REF:-brev-launchable}"`.
6. **Expose ports:** a Secure Link named `isaac` on port 80 (VS Code). Only if you need the streamed Isaac Sim
   viewer, also TCP/UDP 1024, 47998 and 49100; training and the smoke test don't.
7. Keep the launchable private to your Brev account or team.

## Use it

Deploy, wait for the build (45 to 90 minutes the first time), open the `isaac` Secure Link, then in a VS Code
terminal:

```bash
cd /workspaces/isaaclab_arena
python -m sim2gate.training.smoke_go2
```

That runs the whole Go2 smoke test and prints PASS or FAIL. `~/sim2gate-versions.txt` on the host records the
Arena, Isaac Lab and Sim2Gate commits and the rsl-rl-lib version actually installed.

No Secure Link? From WSL or a Mac/Linux terminal: `brev shell <instance>`, then `tmux new -s s2g` and
`docker exec -it isaac-arena-vscode su ubuntu`; or `brev port-forward <instance> -p 8080:80` and open
http://localhost:8080.

## Skip the hour-long build next time

After one successful setup, save the Arena image to a **private** registry (the image contains Isaac Sim, whose
license terms govern redistribution):

```bash
docker login nvcr.io          # user: $oauthtoken, password: your NGC API key (or use AWS ECR)
~/sim2gate/deploy/brev/push_arena_image.sh nvcr.io/<your-ngc-org>/sim2gate-arena:2cc9e97
```

Then set `ARENA_IMAGE` (and `REGISTRY`, `REGISTRY_USER`, `REGISTRY_TOKEN`) at the top of the pasted setup script.
New instances pull the image in minutes instead of building it.

## Changing versions

- Newer Arena: change `ARENA_REF` in `setup.sh`, then re-run the smoke test before trusting results. If rsl-rl-lib
  changes, the version has to pass Sim2Gate's RSL-RL acceptance gate first.
- Newer Sim2Gate on an existing instance: `cd ~/sim2gate/deploy/brev && git pull && ./build.sh -s -R &&
  docker compose up -d` (`-R` skips Docker's cache, so pip fetches the new code).

## Costs

Brev bills running instances by the hour (about $2.73/hour for a 1× L40S instance when we checked, Oct 2026) and stopped
instances for their disk. Stop the instance after each session; delete it when the image is in a registry.
