#!/usr/bin/env bash
# Tier-R stage wrapper, run on the Brev HOST as ubuntu:
#     bash run_stage.sh S1 --preflight      # no GPU: checkpoint bytes on the instance vs the manifest
#     bash run_stage.sh S1 --run            # only once S1 is APPROVED in the manifest
# Fetches the tools at a pinned commit, checks their digests, then runs run_stage.py.
set -euo pipefail
COMMIT=__COMMIT__
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
__SUMS__
SUMS
cd "$HOST_ROOT"
python3 -I "$TOOLS/run_stage.py" --stage "$STAGE" "$MODE"
