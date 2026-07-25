#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

PID_FILE="${PROJECT_DIR}/results/server.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    log "No server PID file found"
    exit 0
fi

if ! pid_is_ours "${PID_FILE}"; then
    warn "PID file exists but process is not a live project server; removing stale PID file"
    rm -f "${PID_FILE}"
    exit 0
fi

PID="$(<"${PID_FILE}")"
log "Stopping project server PID ${PID}"
kill "${PID}" 2>/dev/null || true
for _ in {1..30}; do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 1
done
if kill -0 "${PID}" 2>/dev/null; then
    warn "server did not stop after 30 seconds; sending TERM again"
    kill -TERM "${PID}" 2>/dev/null || true
fi
rm -f "${PID_FILE}"
record_log "stopped project server PID ${PID}"
