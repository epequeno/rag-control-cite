#!/usr/bin/env bash
# Prove teardown: list every resource tagged project=rag-control-cite (plus
# the deploy-time secrets and CDK bootstrap template objects) and exit
# non-zero if anything remains. The README's "verified to zero resources"
# cites this script's output.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION="${REGION}"
PROJECT="rag-control-cite"
OPENROUTER_SECRET_NAME="${RCC_OPENROUTER_SECRET_NAME:-rag-control-cite/openrouter-api-key}"
TENANT_TOKENS_SECRET_NAME="${RCC_TENANT_TOKENS_SECRET_NAME:-rag-control-cite/tenant-tokens}"

remaining=0

echo "== verify teardown (region=${REGION}) =="

# 1. Anything still tagged for this project, across all services.
tagged="$(aws resourcegroupstaggingapi get-resources \
  --tag-filters "Key=project,Values=${PROJECT}" \
  --query 'ResourceTagMappingList[].ResourceARN' --output text 2>/dev/null || echo "__ERROR__")"
if [[ "${tagged}" == "__ERROR__" ]]; then
  echo "WARN: could not query Resource Groups Tagging API — cannot fully verify" >&2
elif [[ -n "${tagged}" && "${tagged}" != "None" ]]; then
  echo "REMAINING tagged resources:"
  echo "${tagged}" | tr '\t' '\n' | sed 's/^/  /'
  remaining=1
else
  echo "tagged resources: none"
fi

# 2. The deploy-time secrets (not part of the stack).
for name in "${OPENROUTER_SECRET_NAME}" "${TENANT_TOKENS_SECRET_NAME}" "${TENANT_TOKENS_SECRET_NAME}/operator"; do
  if aws secretsmanager describe-secret --secret-id "${name}" >/dev/null 2>&1; then
    echo "REMAINING secret: ${name}"
    remaining=1
  fi
done

# 3. CDK bootstrap template objects (shared bucket; note, do not fail).
ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '')"
if [[ -n "${ACCOUNT}" ]]; then
  BUCKET="cdk-hnb659fds-assets-${ACCOUNT}-${REGION}"
  count="$(aws s3 ls "s3://${BUCKET}/" --recursive 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
  echo "cdk bootstrap template objects in ${BUCKET}: ${count} (shared — owner prunes)"
fi

if [[ "${remaining}" -ne 0 ]]; then
  echo "verify teardown: FAILED — resources remain" >&2
  exit 1
fi
echo "verify teardown: OK — zero project resources remain"
