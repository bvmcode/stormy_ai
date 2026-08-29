resource "aws_cloudwatch_log_group" "task" {
  name              = "/ecs/${var.task_family}"
  retention_in_days = var.log_retention_days
}

resource "aws_ecs_cluster" "main" {
  name = var.cluster_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_security_group" "task" {
  name        = "${var.project_name}-fargate-task"
  description = "Egress-only security group for wx-briefing-agent Fargate tasks"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Allow outbound internet access"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

locals {
  container_environment = [
    {
      name  = "AWS_DEFAULT_REGION"
      value = var.aws_region
    },
    {
      name  = "AWS_REGION"
      value = var.aws_region
    },
    {
      name  = "BRIEFING_S3_BUCKET"
      value = var.s3_bucket
    },
    {
      name  = "RADAR_S3_BUCKET"
      value = var.s3_bucket
    },
    {
      name  = "GFS_S3_BUCKET"
      value = var.s3_bucket
    },
    {
      name  = "MALLOC_ARENA_MAX"
      value = "2"
    },
    {
      name  = "OMP_NUM_THREADS"
      value = "8"
    },
    {
      name  = "OPENBLAS_NUM_THREADS"
      value = "8"
    },
  ]

  container_command = var.default_location != "" ? [var.default_location] : []
}

resource "aws_ecs_task_definition" "wx_briefing_agent" {
  family                   = var.task_family
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }

  container_definitions = jsonencode([
    {
      name        = var.container_name
      image       = var.ecr_image
      essential   = true
      cpu         = var.task_cpu
      memory      = var.task_memory
      command     = length(local.container_command) > 0 ? local.container_command : null
      environment = local.container_environment
      secrets = [
        {
          name      = "HF_TOKEN"
          valueFrom = data.aws_secretsmanager_secret.hf_token.arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.task.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}
