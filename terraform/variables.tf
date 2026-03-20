# =============================================================================
# terraform/variables.tf — Variables for Intel Gaudi 3 Infrastructure Template
# =============================================================================
#
# Configure these variables in terraform.tfvars or via environment variables
# (TF_VAR_<variable_name>) before running terraform apply.
# =============================================================================

variable "cluster_name" {
  description = "Unique name for this training cluster (used in resource naming)"
  type        = string
  default     = "gaudi3-scale"
}

variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be one of: dev, staging, production."
  }
}

variable "region" {
  description = "Cloud region to deploy the training cluster"
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "Availability zone for the training subnet"
  type        = string
  default     = "us-east-1a"
}

# =============================================================================
# Network Configuration
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for the training cluster VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block for the Gaudi 3 training subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into training nodes (restrict in production)"
  type        = string
  default     = "0.0.0.0/0"  # WARNING: Restrict this in production!
}

# =============================================================================
# Compute Configuration
# =============================================================================

variable "num_nodes" {
  description = "Number of Gaudi 3 training nodes in the cluster"
  type        = number
  default     = 1

  validation {
    condition     = var.num_nodes >= 1 && var.num_nodes <= 64
    error_message = "num_nodes must be between 1 and 64."
  }
}

variable "instance_type" {
  description = "Cloud instance type with Gaudi 3 HPU (e.g., dl1.24xlarge for Gaudi 1 on AWS)"
  type        = string
  default     = "dl1.24xlarge"  # Update when Gaudi 3 instances become available
}

variable "gaudi_ami_id" {
  description = "AMI ID for the Habana Deep Learning AMI (Gaudi-optimized)"
  type        = string
  default     = ""  # Set this to the appropriate Habana AMI for your region
}

variable "ssh_key_name" {
  description = "Name of the EC2 key pair for SSH access"
  type        = string
  default     = ""
}

variable "root_volume_size_gb" {
  description = "Size of the root EBS volume in GB (should be large enough for model checkpoints)"
  type        = number
  default     = 500
}

# =============================================================================
# Training Configuration
# =============================================================================

variable "max_epochs" {
  description = "Maximum number of training epochs"
  type        = number
  default     = 10
}

variable "batch_size" {
  description = "Per-device batch size for training"
  type        = number
  default     = 32
}

variable "accumulation_steps" {
  description = "Number of gradient accumulation steps"
  type        = number
  default     = 4
}

variable "base_lr" {
  description = "Base learning rate (will be scaled linearly with effective batch size)"
  type        = number
  default     = 0.001
}

# =============================================================================
# Storage Configuration
# =============================================================================

variable "checkpoint_bucket" {
  description = "S3 bucket name for storing model checkpoints"
  type        = string
  default     = ""
}

# =============================================================================
# Observability
# =============================================================================

variable "wandb_api_key" {
  description = "Weights & Biases API key for experiment tracking"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_monitoring" {
  description = "Enable CloudWatch/Prometheus monitoring for the cluster"
  type        = bool
  default     = true
}

# =============================================================================
# Security
# =============================================================================

variable "intel_api_key" {
  description = "Intel Developer Cloud API key (if using IDC)"
  type        = string
  default     = ""
  sensitive   = true
}
