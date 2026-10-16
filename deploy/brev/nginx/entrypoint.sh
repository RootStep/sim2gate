#!/bin/sh
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
# RootStep: no TLS listener, so no self-signed certificate is generated.
# The streamed-viewer routes get HTTP basic auth (user "sim2gate", the VS Code password); without a password they
# are denied unless ALLOW_NO_AUTH=1. Never echo the password.
if [ -n "${VIEWER_PASSWORD:-}" ]; then
    printf 'sim2gate:%s\n' "$(printf '%s' "${VIEWER_PASSWORD}" | openssl passwd -apr1 -stdin)" > /etc/nginx/viewer.htpasswd
    printf 'auth_basic "Sim2Gate viewer";\nauth_basic_user_file /etc/nginx/viewer.htpasswd;\n' > /etc/nginx/viewer_auth.conf
elif [ "${ALLOW_NO_AUTH:-0}" = "1" ]; then
    printf 'auth_basic off;\n' > /etc/nginx/viewer_auth.conf
else
    printf 'deny all;\n' > /etc/nginx/viewer_auth.conf
fi
chmod 644 /etc/nginx/viewer.htpasswd 2>/dev/null || true   # workers run unprivileged; file holds only a hash
exec nginx -g "daemon off;"
