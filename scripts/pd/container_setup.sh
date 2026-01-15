#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y vim net-tools

scripts_dir="${PD_SCRIPTS_DIR:-}"

if [[ -z "$scripts_dir" ]]; then
  echo "PD_SCRIPTS_DIR is not set; skipping copy of run_*.sh scripts."
else
  # Copy the PD runner scripts into the container for easy access.
  if compgen -G "${scripts_dir}/run_*.sh" >/dev/null; then
    for script in "${scripts_dir}"/run_*.sh; do
      cp "$script" /usr/local/bin/
      chmod +x "/usr/local/bin/$(basename "$script")"
    done
  else
    echo "No run_*.sh scripts found under ${scripts_dir}, skipping copy."
  fi
fi