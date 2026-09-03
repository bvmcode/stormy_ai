AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)
AWS_ACCESS_KEY_ID := $(shell aws configure export-credentials | jq -r '.AccessKeyId')
AWS_SECRET_ACCESS_KEY := $(shell aws configure export-credentials | jq -r '.SecretAccessKey')
HF_TOKEN := $(shell grep '^HF_TOKEN=' .env 2>/dev/null | cut -d '=' -f 2-)
LANGSMITH_API_KEY := $(shell grep '^LANGSMITH_API_KEY=' .env 2>/dev/null | cut -d '=' -f 2-)
LANGSMITH_PROJECT := $(or $(shell grep '^LANGSMITH_PROJECT=' .env 2>/dev/null | cut -d '=' -f 2- | head -1),stormy-ai)
LANGSMITH_TRACING_ENABLED := $(shell grep -q '^LANGSMITH_TRACING=true' .env 2>/dev/null && echo true || echo false)
AWS_REGION := us-east-1
ECR_REPOSITORY_NAME := wx_briefing_agent
IMAGE := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY_NAME):latest

.PHONY: help build create auth push build_and_push create_repo local_run shell exec_shell test-creds test lint format infra-bootstrap infra-init infra-plan infra-apply infra-run-task

help: ## Show available make targets
	@echo "Stormy AI — make targets"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build linux/arm64 Docker image and tag for ECR
	docker buildx build --platform linux/arm64 -t $(IMAGE) .

create: ## Create the ECR repository (one-time)
	aws ecr create-repository --repository-name $(ECR_REPOSITORY_NAME) --region $(AWS_REGION)

auth: ## Log Docker in to ECR
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

push: ## Push the image to ECR (:latest)
	docker push $(IMAGE)

build_and_push: build auth push ## Build, authenticate, and push image to ECR

create_repo: auth create ## Authenticate to ECR and create repository

local_run: ## Run briefing container (HF_TOKEN + AWS creds from .env / aws configure)
	docker run --rm \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e AWS_DEFAULT_REGION=$(AWS_REGION) \
		-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
		$(IMAGE)

shell: ## Interactive bash shell in the image
	docker run --rm -it \
		--entrypoint /bin/bash \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e AWS_DEFAULT_REGION=$(AWS_REGION) \
		-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
		$(IMAGE)

exec_shell: ## Bash into a running container (set CONTAINER=<id> or auto-detect)
	@container=$${CONTAINER:-$$(docker ps -q --filter ancestor=$(IMAGE) | head -1)}; \
	if [ -z "$$container" ]; then \
		echo "No running container found for $(IMAGE)."; \
		echo "Pass CONTAINER=<id> or start one in the background, e.g.:"; \
		echo "  docker run -d --name wx_briefing_agent -e HF_TOKEN=... -e AWS_DEFAULT_REGION=$(AWS_REGION) -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... $(IMAGE) sleep infinity"; \
		exit 1; \
	fi; \
	docker exec -it $$container /bin/bash

test-creds: ## Print HF_TOKEN and AWS creds loaded by the Makefile (debug)
	@echo $(HF_TOKEN)
	@echo $(AWS_ACCESS_KEY_ID)
	@echo $(AWS_SECRET_ACCESS_KEY)

test: ## Run the test suite
	uv run pytest tests/

lint: ## Run flake8 and isort check on src/ and tests/
	uv run flake8 --config .flake8 src/ tests/
	uv run isort --check-only src/ tests/

format: ## Auto-format with black and isort
	uv run black -l 100 src/ tests/
	uv run isort --profile black src/ tests/

INFRA_DIR := infra
HF_TOKEN_SECRET_NAME := stormy-ai/hf-token
LANGSMITH_API_KEY_SECRET_NAME := stormy-ai/langsmith-api-key

infra-bootstrap: ## Create HF_TOKEN and LangSmith secrets in Secrets Manager if missing
	@bootstrap_secret() { \
		name="$$1"; value="$$2"; optional="$$3"; \
		if [ -z "$$value" ]; then \
			if [ "$$optional" = "optional" ]; then \
				echo "Skipping $$name (not set in .env)."; \
			else \
				echo "ERROR: $$name requires a value in .env."; \
				exit 1; \
			fi; \
			return 0; \
		fi; \
		if aws secretsmanager describe-secret --secret-id "$$name" --region $(AWS_REGION) >/dev/null 2>&1; then \
			echo "Secret $$name already exists."; \
		else \
			aws secretsmanager create-secret \
				--name "$$name" \
				--secret-string "$$value" \
				--region $(AWS_REGION); \
			echo "Created secret $$name."; \
		fi; \
	}; \
	bootstrap_secret "$(HF_TOKEN_SECRET_NAME)" "$(HF_TOKEN)"; \
	bootstrap_secret "$(LANGSMITH_API_KEY_SECRET_NAME)" "$(LANGSMITH_API_KEY)" optional

infra-init: ## terraform init in infra/
	terraform -chdir=$(INFRA_DIR) init

infra-plan: infra-init ## terraform plan
	terraform -chdir=$(INFRA_DIR) plan \
		-var="langsmith_tracing_enabled=$(LANGSMITH_TRACING_ENABLED)" \
		-var="langsmith_project=$(LANGSMITH_PROJECT)"

infra-apply: infra-bootstrap infra-init ## Bootstrap secrets and terraform apply
	terraform -chdir=$(INFRA_DIR) apply -auto-approve \
		-var="langsmith_tracing_enabled=$(LANGSMITH_TRACING_ENABLED)" \
		-var="langsmith_project=$(LANGSMITH_PROJECT)"

run-agent: ## Run one ECS Fargate briefing task (manual trigger)
	@cluster=$$(terraform -chdir=$(INFRA_DIR) output -raw ecs_cluster_name); \
	task_def=$$(terraform -chdir=$(INFRA_DIR) output -raw task_definition_arn); \
	expected_cpu=$$(terraform -chdir=$(INFRA_DIR) output -raw task_cpu); \
	expected_mem=$$(terraform -chdir=$(INFRA_DIR) output -raw task_memory_mib); \
	actual_cpu=$$(aws ecs describe-task-definition --task-definition "$$task_def" --region $(AWS_REGION) --query 'taskDefinition.cpu' --output text); \
	actual_mem=$$(aws ecs describe-task-definition --task-definition "$$task_def" --region $(AWS_REGION) --query 'taskDefinition.memory' --output text); \
	subnets=$$(terraform -chdir=$(INFRA_DIR) output -json subnet_ids | jq -r 'join(",")'); \
	sg=$$(terraform -chdir=$(INFRA_DIR) output -raw task_security_group_id); \
	echo "Task definition: $$task_def"; \
	echo "Task size: $$actual_cpu CPU units ($$((actual_cpu / 1024)) vCPU), $$actual_mem MiB ($$((actual_mem / 1024)) GiB)"; \
	if [ "$$actual_cpu" != "$$expected_cpu" ] || [ "$$actual_mem" != "$$expected_mem" ]; then \
		echo "ERROR: Task definition size mismatch. Run 'make infra-apply' first."; \
		exit 1; \
	fi; \
	aws ecs run-task \
		--cluster "$$cluster" \
		--task-definition "$$task_def" \
		--launch-type FARGATE \
		--network-configuration "awsvpcConfiguration={subnets=[$$subnets],securityGroups=[$$sg],assignPublicIp=ENABLED}" \
		--region $(AWS_REGION)