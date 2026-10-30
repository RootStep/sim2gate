#!/usr/bin/env bash
# G2 stage S0 smoke run (authorized 2026-10-30, one run, cap 0.5 GPU-hours / 1.75 USD).
# Run on the Brev host sim2gate-arena-2 as the user who owns /home/ubuntu/IsaacLab-Arena:
#     bash run_s0_smoke.sh
# It fetches the tier-R tools at a pinned commit, checks their digests, runs ONE collector run
# (p0v5b_s101 x N080) in the isaac-arena-vscode container under a 25-minute timeout, and writes
# v2_runs/phase2_tier_r_smoke/s0_evidence.txt. It runs nothing else: no other policy, cell or stage.
set -euo pipefail

COMMIT=7881a3e330c117a82833e23b07fb1a7c9a0c208a
HOST_ROOT=/home/ubuntu/IsaacLab-Arena            # = /workspaces/isaaclab_arena in the container
CONTAINER=isaac-arena-vscode
TOOLS=v2_runs/phase2_eval_tools
SMOKE=v2_runs/phase2_tier_r_smoke
OUT=$SMOKE/p0v5b_s101__N080.jsonl.gz

cd "$HOST_ROOT"
if [ -e "$OUT" ]; then echo "STOP: $OUT already exists; runs are written once"; exit 1; fi
mkdir -p "$TOOLS/configs" "$SMOKE"

echo "== fetch tools at $COMMIT"
BASE=https://raw.githubusercontent.com/RootStep/sim2gate/$COMMIT/benchmark/v0.1
for f in independent_collector.py independent_truth.py independent_analysis.py G2_TIER_R_MANIFEST.json \
         configs/tier_r_commands_64.json; do
  curl -fsSL "$BASE/$f" -o "$TOOLS/$f"
done
( cd "$TOOLS" && sha256sum -c - ) <<'SUMS'
cb6671841a3ece66717630a99fea494a50161a4e648e030ac4b8a61a91e6aa8c  independent_collector.py
19f7e3d222151e4fb4c7ec061acc655d2ae47c394873a3bd60667363e626d0a8  independent_truth.py
cc765b8695f300cf40ff6a2bd0b3996d57b563c554cdbaacb78df777ea12231d  independent_analysis.py
decd3c6102db9ff55dcdb2772d320ef4074e38f4d38dbb7109f7f7c0d1e2d7b8  G2_TIER_R_MANIFEST.json
3e0f2053d9ae6d644c392c14dc49012de8f5065373e7be88fa8e95765ec79440  configs/tier_r_commands_64.json
SUMS

echo "== run S0 in $CONTAINER"
docker exec -i -w /workspaces/isaaclab_arena "$CONTAINER" bash -s <<'IN_CONTAINER'
set -uo pipefail
TOOLS=v2_runs/phase2_eval_tools
SMOKE=v2_runs/phase2_tier_r_smoke
OUT=$SMOKE/p0v5b_s101__N080.jsonl.gz
CKPT=/workspaces/isaaclab_arena/logs/rsl_rl/sim2gate_go2_flat_p0_v5b/2026-10-21_20-28-49/model_1999.pt
EV=$SMOKE/s0_evidence.txt
PY=/isaac-sim/python.sh; [ -x "$PY" ] || PY=python
{
  echo "S0 evidence  $(date -u +%FT%TZ)  python=$PY"
  echo "== checkpoint (manifest: 14e03fab96f671e5b45c43ffefe947ea11754c6a07634d232c58b67745befe62)"
  sha256sum "$CKPT"
  echo "== installed sim2gate dependencies"
  "$PY" - <<'PYDEP' 2>&1 | grep -E "^(sha|IMPORT|attrs)" || true
import hashlib, importlib
for m in ("sim2gate.checks.isaaclab_conditions", "sim2gate.checks.commands"):
    try:
        mod = importlib.import_module(m)
        print("sha", hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest(), mod.__file__)
    except Exception as e:
        print("IMPORT FAIL", m, type(e).__name__, e)
try:
    import sim2gate.checks.isaaclab_conditions as c
    print("attrs REMOVED_EVENTS", getattr(c, "REMOVED_EVENTS", None), "install", hasattr(c, "install"),
          "apply_profile", hasattr(c, "apply_profile"))
except Exception as e:
    print("IMPORT FAIL attrs", e)
PYDEP
} > "$EV" 2>&1

START=$(date -u +%s)
echo "== collector start $(date -u +%FT%TZ)" >> "$EV"
timeout 1500 "$PY" "$TOOLS/independent_collector.py" --manifest "$TOOLS/G2_TIER_R_MANIFEST.json" \
  --stage S0 --policy p0v5b_s101 --cell N080 --out "$OUT" -- \
  --task Sim2Gate-Velocity-Flat-Go2-P0-v5b --checkpoint "$CKPT" \
  physics=isaacsim_physx env.sim.use_newton_actuators=false env.scene.robot.joint_ordering=physx \
  env.scene.robot.body_ordering=physx --external_callback sim2gate.v2.tasks.register --seed 20261031 \
  < /dev/null > "$SMOKE/s0_run.log" 2>&1        # stdin closed: this script itself arrives on stdin
RC=$?
END=$(date -u +%s)
echo "== collector exit $RC  wall_seconds $((END-START))  end $(date -u +%FT%TZ)  (124 = 25-min timeout)" >> "$EV"

"$PY" - "$TOOLS" "$OUT" >> "$EV" 2>&1 <<'PYEV'
import json, os, sys
tools, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools)
import independent_analysis as ia
man = os.path.join(tools, "G2_TIER_R_MANIFEST.json")
m = json.load(open(man))
if os.path.exists(out):
    h, eps, probs = ia.read_run(out, m, ia._sha(man), "p0v5b_s101", "N080")
    print("== run file", out, os.path.getsize(out), "bytes;", len(eps), "episodes")
    print("== read_run problems:", probs or "none")
    keep = ("completed", "stage", "checkpoint", "checkpoint_matches_manifest", "num_envs", "step_dt", "episode_steps",
            "steps_run", "episode_lengths", "unexpected_early_dones", "foot_names", "task", "play_args",
            "manifest_sha256", "tool_sha256", "truth_lib_sha256", "dependency_files", "cfg_changes",
            "sync_checks", "runtime_checks")
    print(json.dumps({k: (h or {}).get(k) for k in keep}, indent=1, default=str))
else:
    print("== no run file written")
    d = out + ".diagnostic.json"
    if os.path.exists(d):
        print("== diagnostic", d)
        print(open(d).read())
PYEV
if [ "$RC" -ne 0 ]; then echo "== last 80 log lines" >> "$EV"; tail -n 80 "$SMOKE/s0_run.log" >> "$EV"; fi
cat "$EV"
IN_CONTAINER
echo
echo "Done. Evidence: $HOST_ROOT/$SMOKE/s0_evidence.txt   Run log: $HOST_ROOT/$SMOKE/s0_run.log"
