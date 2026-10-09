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
set -eux

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspaces}
CODE_SERVER_EXTENSIONS_DIR=${CODE_SERVER_EXTENSIONS_DIR:-/opt/code-server/extensions}

# Set auth mode based on PASSWORD env var. With a Brev Secure Link the link
# itself is authenticated, so an unset password is the normal case.
if [ -z "${PASSWORD:-}" ]; then
    AUTH_MODE="none"
else
    AUTH_MODE="password"
fi

# Bind to loopback only: nginx is the sole front door for this stack.
exec code-server \
    --bind-addr=127.0.0.1:8080 \
    --auth="${AUTH_MODE}" \
    --extensions-dir "${CODE_SERVER_EXTENSIONS_DIR}" \
    --user-data-dir "${HOME}/.local/share/code-server" \
    "${WORKSPACE_DIR}"
