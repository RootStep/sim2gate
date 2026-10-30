#!/usr/bin/env bash
# Tier-R stage wrapper, run on the Brev HOST as ubuntu:
#     bash run_stage.sh S1 --preflight      # no GPU: checkpoint bytes on the instance vs the manifest
#     bash run_stage.sh S1 --run            # only once S1 is APPROVED in the manifest
# Fetches the tools at a pinned commit, checks their digests, then runs run_stage.py.
set -euo pipefail
COMMIT=083db893b61d9a6766e62ba0fa037fe22b54c031
HOST_ROOT=/home/ubuntu/IsaacLab-Arena
TOOLS=$HOST_ROOT/v2_runs/phase2_eval_tools
STAGE=${1:?stage id, e.g. S1}; MODE=${2:?--preflight or --run}
mkdir -p "$TOOLS/configs"
BASE=https://raw.githubusercontent.com/RootStep/sim2gate/$COMMIT/benchmark/v0.1
for f in run_stage.py independent_collector.py independent_truth.py independent_analysis.py G2_TIER_R_MANIFEST.json \
         configs/tier_r_commands_64.json; do
  curl -fsSL "$BASE/$f" -o "$TOOLS/$f"
done
( cd "$TOOLS" && sha256sum -c - ) <<'SUMS'
cc3d4b6d99ff75043551859fe80fc1d1efe73ad70b43ea3aaa43ca31bcaa0fc9  run_stage.py
e10b20ccbeca92992f48c82dff13901662c7d689885e608caa57a33b0be8c20f  independent_collector.py
19f7e3d222151e4fb4c7ec061acc655d2ae47c394873a3bd60667363e626d0a8  independent_truth.py
cc765b8695f300cf40ff6a2bd0b3996d57b563c554cdbaacb78df777ea12231d  independent_analysis.py
c4139f5c9e93f1a3b0bcc87df11f33229896efcb7e81ece512517e168e27251d  G2_TIER_R_MANIFEST.json
3e0f2053d9ae6d644c392c14dc49012de8f5065373e7be88fa8e95765ec79440  configs/tier_r_commands_64.json
SUMS
cd "$HOST_ROOT"
python3 -I "$TOOLS/run_stage.py" --stage "$STAGE" "$MODE"
