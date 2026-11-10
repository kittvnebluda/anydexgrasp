#!/bin/bash
# Send one command to adg_client and print its output when done: cmd.sh plan r ...
cd "$(dirname "$0")" || exit
until grep -q "ready for commands" client.log 2>/dev/null; do sleep 1; done
n=$(grep -c "<done>" client.log)
echo "$*" >/tmp/adg/cmd.fifo
timeout "${CMD_TIMEOUT:-400}" bash -c "until [ \$(grep -c '<done>' client.log) -gt $n ]; do sleep 1; done"
awk -v n="$n" '/<done>/{c++} c>=n' client.log | grep -v "^\[INFO\].*cmd_vel" | tail -n +2
