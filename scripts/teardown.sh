#!/usr/bin/env bash
# Tear down the CDK stack, the out-of-band secrets, and local state.
# Idempotent: safe to run when nothing is deployed. `just down` always calls it.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

: "${AWS_PROFILE:?set AWS_PROFILE (e.g. arch)}"
REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION="${REGION}"

OPENROUTER_SECRET_NAME="${RCC_OPENROUTER_SECRET_NAME:-rag-control-cite/openrouter-api-key}"
TENANT_TOKENS_SECRET_NAME="${RCC_TENANT_TOKENS_SECRET_NAME:-rag-control-cite/tenant-tokens}"

echo "== rag-control-cite teardown (profile=${AWS_PROFILE} region=${REGION}) =="

STACK="rag-control-cite"
if aws cloudformation describe-stacks --stack-name "${STACK}" >/dev/null 2>&1; then
  (cd "${REPO_DIR}/infra/cdk" && uv run npx -y aws-cdk destroy --force)
else
  echo "no stack found — nothing to destroy"
fi

# Out-of-band secrets created by deploy.sh (not part of the stack)
for name in "${OPENROUTER_SECRET_NAME}" "${TENANT_TOKENS_SECRET_NAME}" "${TENANT_TOKENS_SECRET_NAME}/operator"; do
  if aws secretsmanager describe-secret --secret-id "${name}" >/dev/null 2>&1; then
    aws secretsmanager delete-secret --secret-id "${name}" --force-delete-without-recovery >/dev/null
    echo "deleted secret: ${name}"
  fi
done

# Local deploy state — paths relative to the repo root, not the cwd
rm -f "${REPO_DIR}/.task_ip" "${REPO_DIR}/.cluster" "${REPO_DIR}/.service"
rm -rf "${REPO_DIR}/infra/cdk/cdk.out" "${REPO_DIR}/infra/cdk/cdk-outputs.json"

# Prove it: exit non-zero if anything tagged project=rag-control-cite remains
bash "${REPO_DIR}/scripts/verify_teardown.sh"

echo "teardown complete."