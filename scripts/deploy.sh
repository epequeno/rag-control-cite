#!/usr/bin/env bash
# Deploy to AWS: create secrets out of band, cdk deploy, resolve task IP.
# Requires: AWS_PROFILE, OPENROUTER_API_KEY, region us-east-1.
# Secrets never enter a CloudFormation template — the stack references them
# by name (see infra/cdk/cdk_stack.py).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

: "${AWS_PROFILE:?set AWS_PROFILE (e.g. arch)}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION="${REGION}"

OPENROUTER_SECRET_NAME="${RCC_OPENROUTER_SECRET_NAME:-rag-control-cite/openrouter-api-key}"
TENANT_TOKENS_SECRET_NAME="${RCC_TENANT_TOKENS_SECRET_NAME:-rag-control-cite/tenant-tokens}"

echo "== rag-control-cite deploy (profile=${AWS_PROFILE} region=${REGION}) =="

# Ingress from the deployer's current IP only (required by the stack)
DEPLOYER_IP="$(curl -fsS https://checkip.amazonaws.com)/32"
export RCC_DEPLOYER_IP="${DEPLOYER_IP}"
echo "deployer IP: ${DEPLOYER_IP}"

put_secret() {
  local name="$1" value="$2"
  if aws secretsmanager describe-secret --secret-id "${name}" >/dev/null 2>&1; then
    aws secretsmanager put-secret-value --secret-id "${name}" --secret-string "${value}" >/dev/null
  else
    aws secretsmanager create-secret --name "${name}" --secret-string "${value}" >/dev/null
  fi
}

# OpenRouter key — created here, referenced by name from the stack
put_secret "${OPENROUTER_SECRET_NAME}" "${OPENROUTER_API_KEY}"

# Tenant bearer tokens — generate if the deployer did not supply them.
if [[ -z "${RCC_TENANT_TOKENS:-}" ]]; then
  RCC_TENANT_TOKENS="$(python3 - <<'PY'
import json, secrets
tenants = ["eng-low", "sec-moderate", "priv-high", "sec-high"]
print(json.dumps({t: secrets.token_urlsafe(24) for t in tenants}))
PY
)"
  echo "generated tenant tokens (save these):"
  echo "${RCC_TENANT_TOKENS}" | python3 -m json.tool
fi
put_secret "${TENANT_TOKENS_SECRET_NAME}" "${RCC_TENANT_TOKENS}"
if [[ -n "${RCC_OPERATOR_TOKEN:-}" ]]; then
  put_secret "${TENANT_TOKENS_SECRET_NAME}/operator" "${RCC_OPERATOR_TOKEN}"
fi

cd infra/cdk
uv run npx -y aws-cdk bootstrap --require-approval never 2>/dev/null || true
uv run npx -y aws-cdk deploy --require-approval never --outputs-file cdk-outputs.json

echo "== resolving task public IP =="
CLUSTER="$(python3 -c "import json;d=json.load(open('cdk-outputs.json'));print(list(d.values())[0].get('ClusterName',''))" 2>/dev/null || true)"
SERVICE="$(python3 -c "import json;d=json.load(open('cdk-outputs.json'));print(list(d.values())[0].get('ServiceName',''))" 2>/dev/null || true)"

if [[ -z "${CLUSTER}" || -z "${SERVICE}" ]]; then
  echo "could not resolve cluster/service from cdk-outputs.json" >&2
  exit 1
fi

PUBLIC_IP=""
for i in $(seq 1 60); do
  TASK_ARN="$(aws ecs list-tasks --cluster "${CLUSTER}" --service-name "${SERVICE}" \
    --query 'taskArns[0]' --output text 2>/dev/null || true)"
  if [[ -n "${TASK_ARN}" && "${TASK_ARN}" != "None" ]]; then
    ENI="$(aws ecs describe-tasks --cluster "${CLUSTER}" --tasks "${TASK_ARN}" \
      --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
      --output text 2>/dev/null || true)"
    if [[ -n "${ENI}" && "${ENI}" != "None" ]]; then
      PUBLIC_IP="$(aws ec2 describe-network-interfaces --network-interface-ids "${ENI}" \
        --query 'NetworkInterfaces[0].Association.PublicIp' --output text 2>/dev/null || true)"
    fi
  fi
  if [[ -n "${PUBLIC_IP}" && "${PUBLIC_IP}" != "None" ]]; then
    echo "task public IP: ${PUBLIC_IP}"
    echo "${PUBLIC_IP}" > "${REPO_DIR}/.task_ip"
    echo "${CLUSTER}" > "${REPO_DIR}/.cluster"
    echo "${SERVICE}" > "${REPO_DIR}/.service"
    exit 0
  fi
  sleep 5
done

echo "timed out waiting for task public IP" >&2
exit 1