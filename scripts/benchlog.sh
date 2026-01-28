#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <log-file> <command...>" >&2
  exit 2
fi

log_file=$1
shift

stamp=$(date +"%Y-%m-%d %H:%M:%S %Z")

mkdir -p "$(dirname "$log_file")"

{
  echo "## $stamp"
  echo "Command: $*"
  echo
  "$@"
  echo
} | tee -a "$log_file"
