variable "aws_region" {
  description = "AWS region for ECS and related resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project name used in resource naming and tags."
  type        = string
  default     = "stormy-ai"
}

variable "cluster_name" {
  description = "ECS cluster name."
  type        = string
  default     = "stormy-ai"
}

variable "task_family" {
  description = "ECS task definition family name."
  type        = string
  default     = "wx-briefing-agent"
}

variable "container_name" {
  description = "Name of the container in the task definition."
  type        = string
  default     = "wx-briefing-agent"
}

variable "ecr_image" {
  description = "Full ECR image URI (include tag) for the briefing agent."
  type        = string
  default     = "122887972227.dkr.ecr.us-east-1.amazonaws.com/wx_briefing_agent:latest"
}

variable "s3_bucket" {
  description = "S3 bucket the task can read from and write to."
  type        = string
  default     = "stormy-ai-files"
}

variable "hf_token_secret_name" {
  description = "Secrets Manager secret name that stores HF_TOKEN."
  type        = string
  default     = "stormy-ai/hf-token"
}

variable "task_cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)."
  type        = number
  default     = 16384
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 122880
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention for task output."
  type        = number
  default     = 14
}

variable "default_location" {
  description = "Default location argument passed to main.py when the task runs."
  type        = string
  default     = "Atco, NJ 08004"
}

variable "briefing_schedule_timezone" {
  description = "IANA timezone for the EventBridge briefing schedule."
  type        = string
  default     = "America/New_York"
}
