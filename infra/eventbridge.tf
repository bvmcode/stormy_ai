resource "aws_iam_role" "scheduler_ecs" {
  name = "${var.project_name}-scheduler-ecs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "scheduler.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "scheduler_ecs_run_task" {
  name = "${var.project_name}-scheduler-ecs-run-task"
  role = aws_iam_role.scheduler_ecs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecs:RunTask"
        ]
        Resource = [
          aws_ecs_task_definition.wx_briefing_agent.arn
        ]
        Condition = {
          ArnLike = {
            "ecs:cluster" = aws_ecs_cluster.main.arn
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = [
          aws_iam_role.ecs_execution.arn,
          aws_iam_role.ecs_task.arn
        ]
      }
    ]
  })
}

resource "aws_scheduler_schedule" "briefing" {
  name       = "${var.project_name}-briefing-schedule"
  group_name = "default"

  description = "Run wx-briefing-agent at midnight, 6am, noon, and 6pm US Eastern"

  schedule_expression          = "cron(0 0,6,12,18 * * ? *)"
  schedule_expression_timezone = var.briefing_schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler_ecs.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.wx_briefing_agent.arn
      launch_type         = "FARGATE"
      platform_version    = "LATEST"
      task_count          = 1

      network_configuration {
        subnets          = data.aws_subnets.public.ids
        security_groups  = [aws_security_group.task.id]
        assign_public_ip = true
      }
    }
  }
}
