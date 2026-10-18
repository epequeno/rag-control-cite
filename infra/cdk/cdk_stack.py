"""CDK stack for rag-control-cite: Fargate + RDS pgvector.

One idea per repo, and the repo's idea is the service: the stack deploys
exactly what `just up` runs locally — the FastAPI container and pgvector —
plus the secret wiring the container needs.

Secrets never enter a template: ``deploy.sh`` creates the OpenRouter key
and the tenant-token map out of band, and the stack takes them by
``from_secret_name_v2``. There is no ``unsafe_plain_text`` anywhere.

Teardown for cents: every resource is tagged ``project=rag-control-cite``
(so ``scripts/verify_teardown.sh`` can prove zero remain), log groups have
retention and ``DESTROY`` removal, and RDS takes no final snapshot — the
corpus is re-ingestable from OSCAL source in under a minute.
"""

from __future__ import annotations

import os

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

SERVICE_PORT = 8000
PROJECT_TAG = "rag-control-cite"

# No curl in the slim image — probe /healthz with the interpreter already there.
_HEALTHCHECK_CMD = (
    'python -c "import urllib.request,sys; '
    "sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=3).status == 200 else 1)\""
)


def _deployer_ip() -> str:
    """The deployer's /32. Required — there is no 0.0.0.0/0 fallback."""
    ip = os.environ.get("RCC_DEPLOYER_IP", "").strip()
    if not ip:
        raise ValueError(
            "RCC_DEPLOYER_IP is required (e.g. from scripts/deploy.sh); "
            "refusing to open ingress to 0.0.0.0/0"
        )
    return ip


class RagControlCiteStack(Stack):
    """Fargate + RDS pgvector + Secrets Manager."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        openrouter_secret_name: str,
        tenant_tokens_secret_name: str,
        embedding_model: str = "openai/text-embedding-3-large",
        generator_model: str = "openai/gpt-4o-mini",
        judge_model: str = "openai/gpt-4o",
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("project", PROJECT_TAG)

        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                )
            ],
        )

        db = rds.DatabaseInstance(
            self,
            "Postgres",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_17),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO),
            allocated_storage=20,
            max_allocated_storage=50,
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,  # corpus is re-ingestable
            credentials=rds.Credentials.from_generated_secret(username="rag"),
            database_name="rag",
            publicly_accessible=True,  # ingestion runs from the deployer
        )

        # Secrets created out of band by deploy.sh — referenced by name, so
        # the key never enters a template.
        openrouter_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "OpenRouterKey", openrouter_secret_name
        )
        tenant_tokens_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "TenantTokens", tenant_tokens_secret_name
        )

        log_group = logs.LogGroup(
            self,
            "AppLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        task_def = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            cpu=256,
            memory_limit_mib=1024,  # cross-encoder + bm25s index
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        container = task_def.add_container(
            "App",
            image=ecs.ContainerImage.from_asset(
                os.path.join("..", ".."),
                file="infra/docker/Dockerfile",
            ),
            environment={
                "EMBEDDING_MODEL": embedding_model,
                "GENERATOR_MODEL": generator_model,
                "JUDGE_MODEL": judge_model,
                "PORT": str(SERVICE_PORT),
                "RDS_HOST": db.instance_endpoint.hostname,
                "RDS_PORT": str(db.instance_endpoint.port),
                "RDS_DB_NAME": "rag",
            },
            secrets={
                "OPENROUTER_API_KEY": ecs.Secret.from_secrets_manager(openrouter_secret),
                "RDS_CREDENTIALS": ecs.Secret.from_secrets_manager(db.secret),
                # JSON {"tenant_id": "token"}; gates /query and /observability.
                "RCC_TENANT_TOKENS": ecs.Secret.from_secrets_manager(tenant_tokens_secret),
            },
            logging=ecs.LogDriver.aws_logs(stream_prefix="rcc", log_group=log_group),
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", _HEALTHCHECK_CMD],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60),
            ),
        )
        container.add_port_mappings(ecs.PortMapping(container_port=SERVICE_PORT))

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        service = ecs.FargateService(
            self,
            "Service",
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            assign_public_ip=True,
        )

        # Ingress: deployer IP only — required, never 0.0.0.0/0.
        my_ip = _deployer_ip()
        service.connections.allow_from(
            ec2.Peer.ipv4(my_ip),
            ec2.Port.tcp(SERVICE_PORT),
            description="HTTP from deployer",
        )
        db.connections.allow_default_port_from(service.connections, "App to Postgres")
        db.connections.allow_from(
            ec2.Peer.ipv4(my_ip),
            ec2.Port.tcp(5432),
            description="Seed from deployer",
        )

        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=service.service_name)
        CfnOutput(self, "DbSecretArn", value=db.secret.secret_arn or "")
