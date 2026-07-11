variable "aws_region" {
  description = "The AWS region to deploy to"
  default     = "ap-southeast-2"
}

variable "project_name" {
  description = "Name of the project"
  default     = "tasco-maps"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  default     = "10.0.0.0/16"
}

variable "github_repo_owner" {
  description = "GitHub repository owner"
  type        = string
}

variable "github_repo_name" {
  description = "GitHub repository name"
  type        = string
}

variable "github_branch" {
  description = "GitHub branch to deploy"
  default     = "main"
}

variable "github_token" {
  description = "GitHub personal access token for CodePipeline"
  type        = string
  sensitive   = true
}

variable "container_port" {
  description = "Port exposed by the docker image"
  default     = 7860
}
