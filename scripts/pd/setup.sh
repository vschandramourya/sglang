#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [hugging_face_hub_token] [pd_image]"
  echo "  user_name: derived from \$(whoami)."
  echo "  hugging_face_hub_token: optional. Passed through to containers if provided."
  echo "  pd_image: optional. Overrides PD_IMAGE; defaults to compose file value if unset."
}

# Expect args: optional HF token, optional image override.
if [[ ${1-} == "-h" || ${1-} == "--help" ]]; then
  usage
  exit 0
fi

USER_NAME="$(whoami)"
HUGGING_FACE_HUB_TOKEN="${1-}"
PD_IMAGE="${2-}"

# Sanity check for all required files to be present.
files=(/data/setup-nixl.sh /data/unset-nixl.sh)
checks_passed=true

for f in "${files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "$f is missing, exiting."
    checks_passed=false
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "Docker access check failed, exiting."
  exit 1
fi

if ! $checks_passed; then
  echo "Sanity checks failed, exiting."
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PD_SCRIPTS_DIR="${PD_SCRIPTS_DIR:-$script_dir}"
echo "Your scripts can be found in ${PD_SCRIPTS_DIR}"

export USER_NAME

if [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
  export HUGGING_FACE_HUB_TOKEN
else
  echo "HUGGING_FACE_HUB_TOKEN not provided, leaving unset."
fi

# Define dx helper function immediately for current session
dx() {
  docker exec -it "$1" bash
}
export -f dx

# Ensure dx helper function exists in ~/.bashrc for persistence in future SSH sessions
bashrc_path="${HOME}/.bashrc"
if ! grep -q "dx()" "$bashrc_path" 2>/dev/null; then
  echo "Adding dx function to $bashrc_path"
  printf '\ndx() { docker exec -it "$1" bash; }\n' >> "$bashrc_path"
fi

# Only export PD_IMAGE if provided to override compose default.
if [[ -n "$PD_IMAGE" ]]; then
  export PD_IMAGE
fi

if ! docker compose -f "${script_dir}/docker-compose.yml" up -d >/dev/null; then
  echo "Failed to start docker compose, exiting."
  exit 1
fi
