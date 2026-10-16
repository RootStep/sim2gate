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
# A thin shim in front of Arena's own entrypoint (/entrypoint.sh), which is left
# untouched so that upstream changes to it keep flowing through.
#
# Arena's entrypoint runs `useradd` without -m and then chowns /home/$USER. The
# Isaac Sim base image only ships /home/ubuntu, so on a host whose user is named
# anything else that chown fails and, under `set -e`, takes the container down.
# run_docker.sh never hits this because it also mounts paths under that home.
# Create the directory first, then hand off unchanged.

set -euo pipefail

HOME_DIR="/home/${DOCKER_RUN_USER_NAME:-ubuntu}"
mkdir -p "${HOME_DIR}"

# The Omniverse caches are mounted here as volumes owned by root; Arena's
# entrypoint only chowns the home directory itself, not what is mounted inside it.
chown -R "${DOCKER_RUN_USER_ID:-1000}:${DOCKER_RUN_GROUP_ID:-1000}" "${HOME_DIR}" || true

exec /entrypoint.sh "$@"
