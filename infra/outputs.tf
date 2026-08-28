output "task_cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)."
  value       = var.task_cpu
}

output "task_memory_mib" {
  description = "Fargate task memory in MiB."
  value       = var.task_memory
}

output "aws_account_id" {
  description = "AWS account ID."
  value       = data.aws_caller_identity.current.account_id
}

output "ecs_cluster_name" {
  description = "ECS cluster name for run-task and future EventBridge targets."
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN."
  value       = aws_ecs_cluster.main.arn
}

output "task_definition_arn" {
  description = "Latest task definition ARN."
  value       = aws_ecs_task_definition.wx_briefing_agent.arn
}

output "task_definition_family" {
  description = "Task definition family (use with revision or :latest suffix)."
  value       = aws_ecs_task_definition.wx_briefing_agent.family
}

output "task_security_group_id" {
  description = "Security group ID for Fargate task networking."
  value       = aws_security_group.task.id
}

output "subnet_ids" {
  description = "Public subnet IDs in the default VPC for Fargate tasks."
  value       = data.aws_subnets.public.ids
}

output "task_execution_role_arn" {
  description = "IAM role ARN used by ECS to pull images and write logs."
  value       = aws_iam_role.ecs_execution.arn
}

output "task_role_arn" {
  description = "IAM role ARN assumed by the running container."
  value       = aws_iam_role.ecs_task.arn
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group for task stdout/stderr."
  value       = aws_cloudwatch_log_group.task.name
}

output "run_task_example" {
  description = "Example aws ecs run-task command with public IP for internet access."
  value       = <<-EOT
    aws ecs run-task \
      --cluster ${aws_ecs_cluster.main.name} \
      --task-definition ${aws_ecs_task_definition.wx_briefing_agent.arn} \
      --launch-type FARGATE \
      --network-configuration "awsvpcConfiguration={subnets=[${join(",", [for subnet_id in data.aws_subnets.public.ids : "\"${subnet_id}\""])}],securityGroups=[\"${aws_security_group.task.id}\"],assignPublicIp=ENABLED}"
  EOT
}
