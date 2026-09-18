#!/usr/bin/env bash
# Appends current llama-server token counters to usage.md (counters reset on server restart).
set -euo pipefail
cd "$(dirname "$0")/.."
m=$(curl -s -m 5 http://127.0.0.1:8000/metrics) || exit 0   # server down: skip silently
p=$(echo "$m" | awk "/^llamacpp:prompt_tokens_total/{print \$2}")
g=$(echo "$m" | awk "/^llamacpp:tokens_predicted_total/{print \$2}")
[ -n "$p" ] && [ -n "$g" ] || exit 0
echo "| $(date +%F\ %H:%M) | $p | $g | $((p+g)) |" >> usage.md
