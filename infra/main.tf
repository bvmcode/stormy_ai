terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "stormy-ai-files"
    key    = "terraform/wx-briefing-agent/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      ManagedBy   = "terraform"
      Application = "wx-briefing-agent"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

data "aws_secretsmanager_secret" "hf_token" {
  name = var.hf_token_secret_name
}

data "aws_secretsmanager_secret" "langsmith_api_key" {
  count = var.langsmith_tracing_enabled ? 1 : 0
  name  = var.langsmith_api_key_secret_name
}
