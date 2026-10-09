#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 RootStep
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# RootStep Sim2Gate launchable: NVIDIA Brev setup script. Paste this file into the "Paste Script" tab when
# creating the Launchable (see deploy/brev/README.md). Also runs by hand on any Linux GPU host with Docker and
# the NVIDIA Container Toolkit.
#
# Builds Isaac Lab-Arena's own, unmodified image at a pinned commit (45-90 minutes, >= 250 GB disk), or pulls a
# prebuilt copy if ARENA_IMAGE names one, then layers VS Code and Sim2Gate on top.

set -euxo pipefail

# ---- settings (edit here, or export before running) -----------------------------------------------------
# Sim2Gate branch or tag: provides this launchable's files and the sim2gate package installed in the image.
SIM2GATE_REF="${SIM2GATE_REF:-main}"
SIM2GATE_REPO_URL="${SIM2GATE_REPO_URL:-https://github.com/RootStep/sim2gate}"
# Isaac Lab-Arena commit. 2cc9e97 (Oct 8, 2026) = Isaac Sim 6.0.1, Isaac Lab 3.0.0, rsl-rl-lib 5.4.1; the
# Experiment T step 1 smoke test passed on it.
ARENA_REF="${ARENA_REF:-2cc9e9766f1f62a500b85c92207192d8c60fcce5}"
ARENA_REPO_URL="${ARENA_REPO_URL:-https://github.com/isaac-sim/IsaacLab-Arena}"
# Optional: a prebuilt Arena image in a PRIVATE registry (see push_arena_image.sh). When set, the instance
# pulls it instead of building for an hour. If the registry needs a login, also set REGISTRY, REGISTRY_USER
# and REGISTRY_TOKEN (for NGC: REGISTRY=nvcr.io, REGISTRY_USER='$oauthtoken', REGISTRY_TOKEN=<NGC API key>).
ARENA_IMAGE="${ARENA_IMAGE:-}"
# --------------------------------------------------------------------------------------------------------

ARENA_REPO="${ARENA_REPO:-$HOME/IsaacLab-Arena}"
SIM2GATE_DIR="${SIM2GATE_DIR:-$HOME/sim2gate}"
LAUNCHABLE_DIR="${SIM2GATE_DIR}/deploy/brev"

export GIT_TERMINAL_PROMPT=0
for url in "${ARENA_REPO_URL}" "${SIM2GATE_REPO_URL}"; do
    if ! git ls-remote "${url}" HEAD >/dev/null 2>&1; then
        echo "ERROR: cannot read git repository: ${url}" >&2
        exit 1
    fi
done

# Arena's .gitmodules uses SSH URLs; a fresh instance has no SSH key.
git config --global url."https://github.com/".insteadOf "git@github.com:"

if [ ! -d "${ARENA_REPO}/.git" ]; then
    git clone "${ARENA_REPO_URL}" "${ARENA_REPO}"
fi
git -C "${ARENA_REPO}" fetch --quiet origin
git -C "${ARENA_REPO}" checkout --quiet "${ARENA_REF}"
git -C "${ARENA_REPO}" submodule update --init --recursive

if [ ! -d "${SIM2GATE_DIR}/.git" ]; then
    git clone --branch "${SIM2GATE_REF}" "${SIM2GATE_REPO_URL}" "${SIM2GATE_DIR}"
else
    git -C "${SIM2GATE_DIR}" fetch --quiet origin "${SIM2GATE_REF}"
    git -C "${SIM2GATE_DIR}" checkout --quiet FETCH_HEAD
fi
cd "${LAUNCHABLE_DIR}"

# Preflight: Isaac Sim renders through Vulkan and streams through NVENC; compute-only driver images lack them.
if command -v nvidia-container-cli >/dev/null 2>&1; then
    DRIVER_LIBS=$(sudo nvidia-container-cli list --libraries 2>/dev/null || true)
    MISSING=""
    echo "${DRIVER_LIBS}" | grep -qi 'libGLX_nvidia' || MISSING="${MISSING} rendering (libGLX_nvidia)"
    echo "${DRIVER_LIBS}" | grep -qi 'libnvidia-encode' || MISSING="${MISSING} encoding (libnvidia-encode/NVENC)"
    if [ -n "${MISSING}" ]; then
        echo "WARNING: this host's NVIDIA driver cannot stream (missing:${MISSING})." >&2
        echo "         Headless training and evaluation still work; the browser viewer will not." >&2
    fi
fi

mkdir -p "$HOME/datasets" "$HOME/models" "$HOME/eval"

cat > .env <<ENVEOF
ARENA_REPO=${ARENA_REPO}
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)
HOST_GROUP=$(id -gn)
DATASETS_DIR=$HOME/datasets
MODELS_DIR=$HOME/models
EVAL_DIR=$HOME/eval
VIEWER_ENV=brev
SIM2GATE_REF=${SIM2GATE_REF}
ENVEOF

BUILD_ARGS=""
if [ -n "${ARENA_IMAGE}" ]; then
    if [ -n "${REGISTRY:-}" ] && [ -n "${REGISTRY_TOKEN:-}" ]; then
        set +x
        echo "${REGISTRY_TOKEN}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
        set -x
    fi
    if docker pull "${ARENA_IMAGE}"; then
        echo "ARENA_IMAGE=${ARENA_IMAGE}" >> .env
        BUILD_ARGS="-s"
    else
        echo "WARNING: could not pull ${ARENA_IMAGE}; building Arena from source instead." >&2
    fi
fi

./build.sh ${BUILD_ARGS} 2>&1 | tee "$HOME/isaac-arena-build.log"
docker compose up -d

# Record exactly what this instance runs.
sleep 5
{
    echo "arena_commit=$(git -C "${ARENA_REPO}" rev-parse HEAD)"
    echo "isaaclab_commit=$(git -C "${ARENA_REPO}/submodules/IsaacLab" rev-parse HEAD)"
    echo "sim2gate_commit=$(git -C "${SIM2GATE_DIR}" rev-parse HEAD)"
    docker exec isaac-arena-vscode /isaac-sim/python.sh -c \
        "import importlib.metadata as m; print('sim2gate', m.version('sim2gate')); print('rsl-rl-lib', m.version('rsl-rl-lib'))"
} | tee "$HOME/sim2gate-versions.txt"
echo "Setup complete. Open the 'isaac' Secure Link for VS Code."
