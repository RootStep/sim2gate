#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Builds the launchable in two steps:
#   1. isaaclab_arena:<tag>   from IsaacLab-Arena's own, unmodified Dockerfile
#   2. the compose images     (vscode layers code-server onto step 1)
#
# Step 1 mirrors the flags in IsaacLab-Arena/docker/run_docker.sh so the image is
# identical to the one Arena developers use locally.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"

# Load .env so ARENA_REPO and friends are available to this script as well as to
# docker compose (compose reads .env itself; a plain script does not).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Path to the IsaacLab-Arena clone on the host. Defaults to a sibling checkout,
# which is the layout in this workspace.
ARENA_REPO="${ARENA_REPO:-$(cd "${SCRIPT_DIR}/../IsaacLab-Arena" 2>/dev/null && pwd || true)}"
ARENA_TAG="${ARENA_TAG:-latest}"
ARENA_IMAGE="${ARENA_IMAGE:-isaaclab_arena:${ARENA_TAG}}"
# Set to true to also build cuRobo, enabling the `ik_reachable` reachability
# check. Equivalent to `run_docker.sh -c`. Adds roughly 10 minutes.
INSTALL_CUROBO="${INSTALL_CUROBO:-false}"
# This MUST match the bind-mount target in docker-compose.yml: the editable
# installs inside the image resolve against this absolute path.
WORKDIR="/workspaces/isaaclab_arena"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [-R] [-s] [-h]

  -R  Rebuild without the Docker cache.
  -s  Skip step 1 (reuse an existing ${ARENA_IMAGE}) and only build the
      compose images. Useful when iterating on the code-server layer.
  -h  Show this help.

Environment:
  ARENA_REPO       Path to the IsaacLab-Arena clone (default: ../IsaacLab-Arena)
  ARENA_TAG        Tag for the Arena image (default: latest)
  INSTALL_CUROBO   true to build cuRobo as well (default: false)
USAGE
}

NO_CACHE=""
SKIP_ARENA=false
while getopts ":Rsh" OPTION; do
    case $OPTION in
        R) NO_CACHE="--no-cache" ;;
        s) SKIP_ARENA=true ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

if [ -z "${ARENA_REPO}" ] || [ ! -f "${ARENA_REPO}/docker/Dockerfile.isaaclab_arena" ]; then
    echo "ERROR: ARENA_REPO does not point at an IsaacLab-Arena clone: '${ARENA_REPO:-<unset>}'" >&2
    echo "       Set ARENA_REPO in .env (see .env.example)." >&2
    exit 1
fi

# The Arena Dockerfile COPYs submodules/IsaacLab and submodules/Isaac-GR00T, so an
# uninitialised clone fails deep into a very long build. Catch it up front.
for sub in submodules/IsaacLab submodules/Isaac-GR00T; do
    if [ ! -e "${ARENA_REPO}/${sub}/.git" ]; then
        echo "ERROR: ${ARENA_REPO}/${sub} is not initialised." >&2
        echo "       Run: git -C '${ARENA_REPO}' submodule update --init --recursive" >&2
        echo "       (Arena's .gitmodules uses SSH URLs; if you have no key, first run:" >&2
        echo "        git config --global url.\"https://github.com/\".insteadOf \"git@github.com:\")" >&2
        exit 1
    fi
done

if [ "${SKIP_ARENA}" = false ]; then
    echo "==> [1/2] Building ${ARENA_IMAGE} from ${ARENA_REPO} (INSTALL_CUROBO=${INSTALL_CUROBO})"
    echo "    This pulls the ~20 GB Isaac Sim base image and can take well over an hour."
    docker build --pull \
        ${NO_CACHE} \
        --progress=plain \
        --build-arg WORKDIR="${WORKDIR}" \
        --build-arg INSTALL_CUROBO="${INSTALL_CUROBO}" \
        -t "${ARENA_IMAGE}" \
        --file "${ARENA_REPO}/docker/Dockerfile.isaaclab_arena" \
        "${ARENA_REPO}"
else
    echo "==> [1/2] Skipped; reusing ${ARENA_IMAGE}"
    if [ -z "$(docker images -q "${ARENA_IMAGE}" 2>/dev/null)" ]; then
        echo "ERROR: ${ARENA_IMAGE} does not exist. Drop -s to build it." >&2
        exit 1
    fi
fi

echo "==> [2/2] Building the launchable images (vscode, nginx, web-viewer)"
ARENA_IMAGE="${ARENA_IMAGE}" docker compose build ${NO_CACHE}

echo "==> Build complete. Start the stack with: docker compose up -d"
