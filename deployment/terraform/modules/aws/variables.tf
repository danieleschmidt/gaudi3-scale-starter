# AWS Infrastructure Variables

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "gaudi3-scale"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3
}

# Cluster Configuration
variable "cluster_endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# CPU Node Group Configuration
variable "cpu_instance_types" {
  description = "Instance types for CPU nodes"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "cpu_desired_capacity" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 3
}

variable "cpu_max_capacity" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_min_capacity" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_disk_size" {
  description = "Disk size for CPU nodes (GB)"
  type        = number
  default     = 100
}

variable "cpu_ami_id" {
  description = "AMI ID for CPU nodes (leave empty for EKS optimized)"
  type        = string
  default     = ""
}

variable "cpu_bootstrap_arguments" {
  description = "Additional bootstrap arguments for CPU nodes"
  type        = string
  default     = ""
}

# Gaudi HPU Node Group Configuration
variable "enable_gaudi_nodes" {
  description = "Enable Gaudi HPU node group"
  type        = bool
  default     = true
}

variable "gaudi_instance_types" {
  description = "Instance types for Gaudi HPU nodes"
  type        = list(string)
  default     = ["dl1.24xlarge"]
}

variable "gaudi_desired_capacity" {
  description = "Desired number of Gaudi HPU nodes"
  type        = number
  default     = 1
}

variable "gaudi_max_capacity" {
  description = "Maximum number of Gaudi HPU nodes"
  type        = number
  default     = 5
}

variable "gaudi_min_capacity" {
  description = "Minimum number of Gaudi HPU nodes"
  type        = number
  default     = 0
}

variable "gaudi_disk_size" {
  description = "Disk size for Gaudi HPU nodes (GB)"
  type        = number
  default     = 500
}

variable "gaudi_ami_id" {
  description = "AMI ID for Gaudi HPU nodes (should include Habana drivers)"
  type        = string
  default     = ""
}

variable "gaudi_bootstrap_arguments" {
  description = "Additional bootstrap arguments for Gaudi HPU nodes"
  type        = string
  default     = ""
}

# Storage Configuration
variable "enable_ebs_csi_driver" {
  description = "Enable EBS CSI driver"
  type        = bool
  default     = true
}

variable "enable_efs_csi_driver" {
  description = "Enable EFS CSI driver"
  type        = bool
  default     = true
}

variable "enable_fsx_csi_driver" {
  description = "Enable FSx CSI driver for high performance storage"
  type        = bool
  default     = false
}

# Monitoring and Logging
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "enable_container_insights" {
  description = "Enable Container Insights for monitoring"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts (IRSA)"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = false
}

variable "enable_network_policy" {
  description = "Enable network policies (requires Calico)"
  type        = bool
  default     = true
}

# Add-ons Configuration
variable "enable_aws_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "enable_cluster_autoscaler" {
  description = "Enable Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable Metrics Server"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Enable cert-manager"
  type        = bool
  default     = true
}

# Common Tags
variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "gaudi3-scale"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

# Application Configuration
variable "deploy_gaudi3_scale" {
  description = "Deploy Gaudi 3 Scale application using Helm"
  type        = bool
  default     = true
}

variable "gaudi3_scale_helm_chart_version" {
  description = "Helm chart version for Gaudi 3 Scale"
  type        = string
  default     = "0.4.0"
}

variable "gaudi3_scale_values" {
  description = "Values for Gaudi 3 Scale Helm chart"
  type        = map(any)
  default     = {}
}