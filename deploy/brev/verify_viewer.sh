#!/usr/bin/env bash
# Verify the web viewer's exposure after a rebuild (fourth review): run on the instance, outside the container.
#
#   bash deploy/brev/verify_viewer.sh | tee ~/viewer_check.txt
#
# Checks, and prints one PASS/FAIL line each:
#   1. the viewer dev server (port 5173) listens on loopback only (127.0.0.1 or [::1]), nowhere else;
#   2. through nginx, /viewer/ without credentials is refused (401);
#   3. through nginx, /viewer/ with the workspace password is served (200).
# The password is read from ~/sim2gate-vscode-password.txt (written by setup.sh) and is never printed.
# Exit code 0 only if all three pass; the last line repeats it ("exit code N") so saved output carries the verdict.
set -uo pipefail

fails=0
check() { if [ "$2" = ok ]; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$((fails + 1)); fi; }

echo "# viewer check on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
listeners=$( (sudo -n ss -Hltnp 2>/dev/null || ss -Hltnp) | awk '{print $4}' | grep -E ':5173$' || true)
echo "# port 5173 listeners: ${listeners:-none}"
if [ -z "$listeners" ]; then
    check "viewer listening on 5173 (is the viewer container running?)" fail
elif echo "$listeners" | grep -vqE '^(127\.0\.0\.1|\[::1\]):5173$'; then
    check "viewer listens on loopback only" fail
else
    check "viewer listens on loopback only" ok
fi

code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' http://localhost/viewer/ 2>/dev/null); code=${code:-000}
[ "$code" = 401 ] && check "nginx /viewer/ without credentials -> 401 (got $code)" ok \
                  || check "nginx /viewer/ without credentials -> 401 (got $code)" fail

pwfile="$HOME/sim2gate-vscode-password.txt"
if [ -r "$pwfile" ]; then
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' -K - http://localhost/viewer/ 2>/dev/null <<EOF
user = "sim2gate:$(tr -d '\n' < "$pwfile")"
EOF
); code=${code:-000}
    [ "$code" = 200 ] && check "nginx /viewer/ with the password -> 200 (got $code)" ok \
                      || check "nginx /viewer/ with the password -> 200 (got $code)" fail
else
    check "password file $pwfile readable" fail
fi

rc=$([ "$fails" -eq 0 ] && echo 0 || echo 1)
echo "exit code $rc"
exit "$rc"
