#!/usr/bin/env bash
# Fetch OSCAL content from usnistgov/oscal-content (public domain).
# Usage: bash scripts/fetch_corpus.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS_DIR="${REPO_DIR}/data/corpus/oscal"

echo "Fetching OSCAL content into ${CORPUS_DIR}..."

# Clone the OSCAL content repository (shallow, content only)
if [ -d "${CORPUS_DIR}/.git" ]; then
  echo "OSCAL content already present — pull to update."
  git -C "${CORPUS_DIR}" pull --ff-only
else
  git clone --depth 1 \
    https://github.com/usnistgov/oscal-content.git \
    "${CORPUS_DIR}"
fi

echo "Done."
echo "Primary catalogs:"
echo "  NIST SP 800-53 rev5      $(find "${CORPUS_DIR}" -path '*800-53*' -name '*.json' | head -1)"
echo "  NIST SP 800-171 rev3     $(find "${CORPUS_DIR}" -path '*800-171*' -name '*.json' | head -1)"
echo "  NIST CSF v2.0            $(find "${CORPUS_DIR}" -path '*csf*' -name '*.json' | head -1)"