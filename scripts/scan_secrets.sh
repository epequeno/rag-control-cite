#!/usr/bin/env bash
# Fail if a key-shaped string is anywhere in the tree. Wire as a pre-commit
# hook once the repo has git:  ln -s ../../scripts/scan_secrets.sh .git/hooks/pre-commit
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

matches="$(grep -rIlE "sk-or-v1-[A-Za-z0-9]{20,}" . 2>/dev/null || true)"
if [[ -n "${matches}" ]]; then
  echo "secret-shaped string found in:" >&2
  echo "${matches}" | sed 's/^/  /' >&2
  exit 1
fi
echo "secret scan: clean"
