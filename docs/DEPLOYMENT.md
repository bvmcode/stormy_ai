# Deployment

Stormy AI runs as a CLI process locally or inside a Docker container on ECS Fargate. The container image is built for **linux/arm64** and expects AWS credentials (for S3 uploads) and an `HF_TOKEN` secret when using the Hugging Face LLM provider.

---

## Docker image

The [`Dockerfile`](../Dockerfile) uses a two-stage build:

1. **Builder** — installs system libraries for eccodes, PROJ, GEOS, HDF5, and compiles Python dependencies with `uv`.
2. **Runtime** — copies the virtualenv and application code; runs as non-root user `app`.

Default entrypoint:

```bash
python main.py [location]
```

When no location argument is passed, `config.yaml` `briefing.default_location` is used (`Atco, NJ 08004` by default).

### Makefile targets

| Target | Purpose |
|--------|---------|
| `make build` | Build and tag image for your AWS account ECR |
| `make auth` | `docker login` to ECR |
| `make push` | Push `:latest` to ECR |
| `make build_and_push` | Build, auth, push |
| `make local_run` | Run container with creds from `.env` and `aws configure` |
| `make shell` | Interactive bash in the image |
| `make lint` / `make format` | Code quality (not container-specific) |

Image URI pattern:

```text
<account-id>.dkr.ecr.us-east-1.amazonaws.com/wx_briefing_agent:latest
```

---

## AWS infrastructure

Terraform in [`infra/`](../infra/) provisions:

| Resource | Purpose |
|----------|---------|
| ECS cluster `stormy-ai` | Fargate cluster with Container Insights |
| Task definition `wx-briefing-agent` | 16 vCPU / 120 GiB ARM64 container |
| IAM execution role | Pull ECR image, write CloudWatch logs, read Secrets Manager |
| IAM task role | Read/write `stormy-ai-files` S3 bucket |
| Security group | Egress-only (outbound internet for APIs and HF router) |
| CloudWatch log group | `/ecs/wx-briefing-agent` (14-day retention) |

The task reads `HF_TOKEN` from Secrets Manager secret `stormy-ai/hf-token`.

### Bootstrap and apply

```bash
# Create the HF token secret (reads HF_TOKEN from .env)
make infra-bootstrap

# Plan and apply Terraform
make infra-plan
make infra-apply
```

`infra-apply` runs `infra-bootstrap` automatically so the secret exists before the task definition references it.

### Run a one-off briefing on Fargate

```bash
make infra-run-task
```

This reads cluster, task definition, subnets, and security group from Terraform outputs, verifies the task CPU/memory match Terraform variables, and calls `aws ecs run-task` with a public IP.

To pass a location, set `default_location` in `infra/variables.tf` (or via `-var`) and re-apply, or override the container command in a custom `run-task` invocation.

### Environment inside the task

| Variable | Source | Purpose |
|----------|--------|---------|
| `HF_TOKEN` | Secrets Manager | Hugging Face API access |
| `AWS_DEFAULT_REGION` / `AWS_REGION` | Task env | S3 client region |
| `BRIEFING_S3_BUCKET` | Task env | Briefing upload bucket |
| `RADAR_S3_BUCKET` / `GFS_S3_BUCKET` | Task env | Plot upload buckets |
| `MALLOC_ARENA_MAX` | Task env | Limit glibc arenas (memory) |
| `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | Task env | Cap BLAS threads (8) |

Task role credentials are provided automatically by ECS — no static AWS keys in the container.

### Logs

```bash
aws logs tail /ecs/wx-briefing-agent --follow --region us-east-1
```

---

## S3 outputs

Successful runs upload:

- Briefing markdown → `s3://stormy-ai-files/briefings/<date>/<zip>/<time>.md`
- Radar PNG → `s3://stormy-ai-files/radar/<date>/<time>.png`
- GFS charts → `s3://stormy-ai-files/models/gfs/<date>/<type>/<hour>.png`

Public embed URLs use `https://<bucket>.s3.amazonaws.com/<key>` unless `STORMY_S3_PUBLIC_BASE` is set (for CloudFront or custom domains). The bucket policy must grant `s3:GetObject` on these prefixes.

---

## Local vs cloud

| Concern | Local CLI | ECS Fargate |
|---------|-----------|-------------|
| LLM | Ollama or HF (your choice in `config.yaml`) | HF via `HF_TOKEN` secret (typical) |
| AWS creds | `aws configure` or env vars | Task IAM role |
| Image arch | Host Python or `make local_run` (arm64) | ARM64 Fargate |
| Config | `config.yaml` + `.env` | Baked into image; override via env if needed |

For local development with Ollama, run `python main.py` directly — no Docker or ECS required. S3 uploads still need valid AWS credentials when enabled.
