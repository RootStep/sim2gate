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

# Layer code-server on top of the IsaacLab Arena image, which is built by
# ../build.sh from IsaacLab-Arena/docker/Dockerfile.isaaclab_arena (unmodified).
# That image already carries Isaac Sim 6.0.1, Isaac Lab 3.0 (editable, from the
# submodule) and IsaacLab Arena, so nothing sim-related is installed here.
ARG ARENA_IMAGE=isaaclab_arena:latest
FROM ${ARENA_IMAGE}

USER root

# The Arena image already provides git, cmake, curl, jq, ffmpeg and build tooling.
# dnsutils/iproute2 mirror the upstream launchable image (handy for debugging the
# host-networked stack from inside the container); unzip is needed by code-server
# to unpack extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip \
        dnsutils \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.96.4

# Arena's entrypoint creates the user account at *runtime* (to match the host
# uid/gid), so anything baked into a home directory at build time would be
# invisible. Keep extensions in a world-readable shared location instead.
ENV CODE_SERVER_EXTENSIONS_DIR=/opt/code-server/extensions

# code-server opens the parent directory, not the repo: the Arena clone is
# bind-mounted at /workspaces/isaaclab_arena and would shadow a README baked
# there. This way the launchable quickstart survives the mount.
ENV WORKSPACE_DIR=/workspaces

COPY workspace-README.md ${WORKSPACE_DIR}/README.md
COPY settings.json ${WORKSPACE_DIR}/.vscode/settings.json

# code-server does not create --extensions-dir itself; it errors out if missing.
RUN mkdir -p ${CODE_SERVER_EXTENSIONS_DIR} \
    && code-server \
        --extensions-dir ${CODE_SERVER_EXTENSIONS_DIR} \
        --install-extension ms-python.python \
        --install-extension ms-toolsai.jupyter \
        --force ${WORKSPACE_DIR} \
    && chmod -R a+rX /opt/code-server

# The runtime user owns neither of these yet (it does not exist at build time),
# so make them writable for whoever the entrypoint creates.
RUN mkdir -p ${WORKSPACE_DIR}/isaaclab_arena && chmod -R a+rwX ${WORKSPACE_DIR}

# RootStep: Sim2Gate's training monitor, installed into Isaac Sim's Python. The install must
# not change rsl-rl-lib (Sim2Gate's math is gated on specific versions), so compare before/after.
ARG SIM2GATE_REF=main
RUN before=$(/isaac-sim/python.sh -c "import importlib.metadata as m; print(m.version('rsl-rl-lib'))") \
    && /isaac-sim/python.sh -m pip install --no-cache-dir \
        "sim2gate[rslrl] @ git+https://github.com/RootStep/sim2gate@${SIM2GATE_REF}" \
    && after=$(/isaac-sim/python.sh -c "import importlib.metadata as m; print(m.version('rsl-rl-lib'))") \
    && echo "rsl-rl-lib before=${before} after=${after}" \
    && test "${before}" = "${after}" \
    && /isaac-sim/python.sh -c "import sim2gate.training; print('sim2gate', __import__('sim2gate').__version__)"

COPY launch-code-server.sh /launch-code-server.sh
COPY entrypoint.sh /launchable-entrypoint.sh
RUN chmod +x /launch-code-server.sh /launchable-entrypoint.sh

# The shim prepares the runtime user's home directory, then hands off to Arena's
# own /entrypoint.sh (inherited from the base image, unmodified). That entrypoint
# creates the host-matched user, grants passwordless sudo, creates /datasets,
# /models and /eval, restores the _isaac_sim symlink, and finally runs CMD as that
# user through `bash -ic`, so the image aliases (python -> /isaac-sim/python.sh) apply.
ENTRYPOINT ["/launchable-entrypoint.sh"]
CMD ["/launch-code-server.sh"]
