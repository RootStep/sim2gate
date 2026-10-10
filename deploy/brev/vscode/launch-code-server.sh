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
set -eu
# No `set -x`: PASSWORD is in the environment and must never reach a log.

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspaces}
CODE_SERVER_EXTENSIONS_DIR=${CODE_SERVER_EXTENSIONS_DIR:-/opt/code-server/extensions}

# RootStep: fail closed. code-server requires a password unless ALLOW_NO_AUTH=1 is set explicitly; setup.sh
# generates one. nginx listens on the host network, so a Secure Link alone does not protect the editor.
if [ -n "${PASSWORD:-}" ]; then
    AUTH_MODE="password"
elif [ "${ALLOW_NO_AUTH:-0}" = "1" ]; then
    echo "WARNING: code-server running WITHOUT authentication (ALLOW_NO_AUTH=1)" >&2
    AUTH_MODE="none"
else
    echo "ERROR: no PASSWORD set for code-server; refusing to start without authentication." >&2
    echo "       Set VSCODE_PASSWORD in .env (setup.sh generates one), or ALLOW_NO_AUTH=1 to override." >&2
    exit 1
fi

# Bind to loopback only: nginx is the sole front door for this stack.
exec code-server \
    --bind-addr=127.0.0.1:8080 \
    --auth="${AUTH_MODE}" \
    --extensions-dir "${CODE_SERVER_EXTENSIONS_DIR}" \
    --user-data-dir "${HOME}/.local/share/code-server" \
    "${WORKSPACE_DIR}"
