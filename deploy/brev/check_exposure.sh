#!/usr/bin/env bash
# Ingress evidence for the Brev workspace (review finding 1).
#
# The services run with host networking, so every socket bound to a non-loopback address is reachable from the
# internet unless the cloud firewall blocks it. This script records both halves:
#
#   On the instance:       bash deploy/brev/check_exposure.sh > exposure_inside.txt
#     lists every listening TCP and UDP socket, which process owns it, and whether it is loopback-only, and prints
#     the command to run from outside.
#
#   From another machine:  bash check_exposure.sh --probe <public-ip> <port> [<port> ...] > exposure_outside.txt
#     tries a TCP connection to each port (3 s timeout) and reports open / closed-or-filtered, then prints its own
#     exit code as the last line ("exit code 0" = nothing reachable). Needs only bash.
#     UDP cannot be probed this way; the inside listing shows which UDP ports are bound.
#
# Expected for a correctly locked-down instance: from outside, every port is closed-or-filtered; access goes through
# Brev's Secure Link only.
set -euo pipefail

probe() {
    local ip=$1; shift
    [ $# -gt 0 ] || { echo "usage: $0 --probe <public-ip> <port> [<port> ...]" >&2; exit 2; }
    echo "# exposure probe from $(hostname) to $ip at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local open=0 p
    for p in "$@"; do
        if timeout 3 bash -c "exec 3<>/dev/tcp/$ip/$p" 2>/dev/null; then
            echo "tcp $p OPEN"; open=$((open + 1))
        else
            echo "tcp $p closed-or-filtered"
        fi
    done
    echo "# $open of $# probed TCP ports reachable"
    local rc=0
    [ "$open" -eq 0 ] || rc=1
    echo "exit code $rc"            # saved with the output, so the evidence file carries the verdict
    return "$rc"
}

if [ "${1:-}" = "--probe" ]; then
    shift; probe "$@"; exit $?
fi

command -v ss >/dev/null || { echo "ss not found (iproute2)" >&2; exit 2; }
echo "# listening sockets on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
pub=$(curl -fsS -m 3 http://checkip.amazonaws.com 2>/dev/null || true)
echo "# public IP: ${pub:-unknown}"
echo "# proto  local-address            exposure        process"
ports=()
while read -r proto _ _ _ local _ proc; do
    addr=${local%:*}; port=${local##*:}
    case "$addr" in
        127.*|"[::1]"|"::1"|"[::ffff:127."*) scope="loopback-only" ;;
        *) scope="ALL-INTERFACES"; [ "$proto" = tcp ] && ports+=("$port") ;;
    esac
    printf '%-6s %-26s %-15s %s\n' "$proto" "$local" "$scope" "${proc:--}"
done < <(sudo -n ss -Hltunp 2>/dev/null || ss -Hltunp)
mapfile -t ports < <(printf '%s\n' "${ports[@]}" | grep -v "^$" | sort -un || true)
echo
echo "# TCP ports bound beyond loopback: ${ports[*]:-none}"
echo "# From a machine outside this cloud, run:"
echo "#   bash check_exposure.sh --probe ${pub:-<public-ip>} ${ports[*]:-}"
