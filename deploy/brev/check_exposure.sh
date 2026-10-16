#!/usr/bin/env bash
# Ingress evidence for the Brev workspace (review finding 1).
#
# The services run with host networking, so every socket bound to a non-loopback address is reachable from the
# internet unless the cloud firewall blocks it. This script records both halves:
#
#   On the instance:       bash deploy/brev/check_exposure.sh > exposure_inside.txt
#     lists every listening TCP and UDP socket, which process owns it, and what it is bound to: loopback-only,
#     ALL-INTERFACES (0.0.0.0 / [::]), private-overlay (100.64.0.0/10, e.g. Brev's NetBird network) or one
#     specific address (this-address); then prints the command to run from outside, covering every TCP port
#     that is not loopback-only.
#
#   From another machine:  bash check_exposure.sh --probe <public-ip> <port> [<port> ...] > exposure_outside.txt
#     tries a TCP connection to each port (3 s timeout) and reports open / closed-or-filtered, then prints its own
#     exit code as the last line ("exit code 0" = nothing reachable). Each port is first tried at 192.0.2.1, an
#     address that cannot answer: if it connects, the probing network intercepts that port (home routers and WSL
#     often answer DNS on port 53 themselves), so the port is reported INTERCEPTED-LOCALLY and the run is
#     inconclusive for it (exit code 3). Needs only bash.
#     UDP cannot be probed this way; the inside listing shows which UDP ports are bound.
#
# Expected for a correctly locked-down instance: from outside, every port is closed-or-filtered; access goes through
# Brev's Secure Link only.
set -euo pipefail

CONTROL_IP=192.0.2.1    # TEST-NET-1 (RFC 5737): never routed on the internet, so nothing there can answer

connects() { timeout 3 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; }

probe() {
    local ip=$1; shift
    [ $# -gt 0 ] || { echo "usage: $0 --probe <public-ip> <port> [<port> ...]" >&2; exit 2; }
    echo "# exposure probe from $(hostname) to $ip at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# control: each port is also tried at $CONTROL_IP, which cannot answer; if it does, the probing"
    echo "#          network answers that port itself (e.g. DNS interception), and the result is inconclusive"
    local open=0 intercepted=0 p
    for p in "$@"; do
        if connects "$CONTROL_IP" "$p"; then
            echo "tcp $p INTERCEPTED-LOCALLY"; intercepted=$((intercepted + 1))
        elif connects "$ip" "$p"; then
            echo "tcp $p OPEN"; open=$((open + 1))
        else
            echo "tcp $p closed-or-filtered"
        fi
    done
    echo "# $open of $# probed TCP ports reachable"
    echo "# $intercepted of $# probed TCP ports intercepted by the probing network (not measurable from here)"
    local rc=0
    if [ "$open" -gt 0 ]; then rc=1; elif [ "$intercepted" -gt 0 ]; then rc=3; fi
    echo "exit code $rc"            # saved with the output: 0 nothing reachable, 1 reachable, 3 inconclusive
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
    addr=${local%:*}; port=${local##*:}; addr=${addr%%%*}     # drop an interface suffix (127.0.0.53%lo)
    case "$addr" in
        127.*|"[::1]"|"::1"|"[::ffff:127."*) scope="loopback-only" ;;
        0.0.0.0|"*"|"[::]"|"::") scope="ALL-INTERFACES" ;;
        100.6[4-9].*|100.[7-9][0-9].*|100.1[01][0-9].*|100.12[0-7].*) scope="private-overlay" ;;   # 100.64.0.0/10
        *) scope="this-address" ;;
    esac
    case "$scope" in loopback-only) ;; *) [ "$proto" = tcp ] && ports+=("$port") ;; esac
    printf '%-6s %-26s %-15s %s\n' "$proto" "$local" "$scope" "${proc:--}"
done < <(sudo -n ss -Hltunp 2>/dev/null || ss -Hltunp)
mapfile -t ports < <(printf '%s\n' "${ports[@]}" | grep -v "^$" | sort -un || true)
echo
echo "# TCP ports bound beyond loopback: ${ports[*]:-none}"
echo "# From a machine outside this cloud, run:"
echo "#   bash check_exposure.sh --probe ${pub:-<public-ip>} ${ports[*]:-}"
