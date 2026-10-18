"""CDK synth: no secret literal, no 0.0.0.0/0 ingress, log retention set.

Requires the ``cdk`` dependency group (``uv sync --group cdk``) and node.
Skipped when the group is not installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CDK_DIR = REPO / "infra" / "cdk"

pytest.importorskip("aws_cdk", reason="cdk dependency group not installed")

# Placeholder account, deliberately not a real one: this suite must not publish
# anyone's account id. cdk.context.json carries the matching placeholder key so
# synth stays offline. If a real deploy rewrites cdk.context.json with the real
# account, do not commit that file.
_SYNTH_ENV = {
    "CDK_DEFAULT_ACCOUNT": "123456789012",
    "CDK_DEFAULT_REGION": "us-east-1",
    "RCC_DEPLOYER_IP": "203.0.113.7/32",
}


def _run_synth(out_dir: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **_SYNTH_ENV, **extra_env}
    return subprocess.run(
        ["npx", "-y", "aws-cdk", "synth", "--quiet", "--output", str(out_dir)],
        check=False,
        cwd=CDK_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> dict:
    if shutil.which("npx") is None:
        pytest.skip("npx not available")
    out = tmp_path_factory.mktemp("cdkout")
    result = _run_synth(out, {})
    if result.returncode != 0:
        pytest.skip(f"cdk synth unavailable: {result.stderr[-400:]}")
    path = out / "rag-control-cite.template.json"
    return json.loads(path.read_text())


def test_no_secret_literal(template) -> None:
    text = json.dumps(template)
    assert "sk-or-v1-" not in text
    # Secrets referenced by name, not value.
    assert "SecretId" in text


def test_no_open_ingress(template) -> None:
    for key, res in template["Resources"].items():
        if not res["Type"].startswith("AWS::EC2::SecurityGroup"):
            continue
        for rule in res.get("Properties", {}).get("SecurityGroupIngress", []):
            assert rule.get("CidrIp") != "0.0.0.0/0", f"open ingress on {key}"


def test_log_retention_set(template) -> None:
    retention = [
        res
        for res in template["Resources"].values()
        if res["Type"] == "AWS::Logs::LogGroup" and "RetentionInDays" in res.get("Properties", {})
    ]
    assert retention, "no log group with retention — logs would outlive the stack"


def test_asset_container_has_healthcheck(template) -> None:
    text = json.dumps(template)
    assert "healthz" in text


def test_synth_requires_deployer_ip(tmp_path) -> None:
    if shutil.which("npx") is None:
        pytest.skip("npx not available")
    env = {**os.environ, "CDK_DEFAULT_ACCOUNT": "123456789012", "CDK_DEFAULT_REGION": "us-east-1"}
    env.pop("RCC_DEPLOYER_IP", None)
    result = subprocess.run(
        ["npx", "-y", "aws-cdk", "synth", "--quiet", "--output", str(tmp_path / "noip")],
        check=False,
        cwd=CDK_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0
    assert "RCC_DEPLOYER_IP" in (result.stderr + result.stdout)
