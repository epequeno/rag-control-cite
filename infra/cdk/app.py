"""CDK app entry point.

Usage:
  AWS_PROFILE=arch cdk deploy    (via `just deploy`)
  AWS_PROFILE=arch cdk destroy   (via `just down`)

Secrets are created by ``scripts/deploy.sh`` before synth and passed here
by name — the stack never holds a secret value.
"""

from __future__ import annotations

import os

from aws_cdk import App, Environment
from cdk_stack import RagControlCiteStack

from rag_control_cite.config import settings

app = App()

RagControlCiteStack(
    app,
    "rag-control-cite",
    env=Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
    openrouter_secret_name=os.environ.get(
        "RCC_OPENROUTER_SECRET_NAME", "rag-control-cite/openrouter-api-key"
    ),
    tenant_tokens_secret_name=os.environ.get(
        "RCC_TENANT_TOKENS_SECRET_NAME", "rag-control-cite/tenant-tokens"
    ),
    embedding_model=settings.embedding_model,
    generator_model=settings.generator_model,
    judge_model=settings.judge_model,
)

app.synth()
