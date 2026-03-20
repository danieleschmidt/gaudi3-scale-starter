# =============================================================================
# terraform/main.tf — Template for Intel Gaudi 3 HPU Infrastructure
# =============================================================================
#
# NOTE: This is a TEMPLATE configuration for reference purposes.
# It does not target a real provider and requires customization before use.
#
# To use with a real cloud provider (e.g., AWS, GCP, Azure):
#   1. Configure the provider block with your credentials
#   2. Replace placeholder resource types with actual Gaudi 3 instance types
#   3. Update network/security settings for your environment
#   4. Run: terraform init && terraform plan && terraform apply
#
# Intel Gaudi 3 is available via:
#   - AWS: dl2q instances (Qualcomm AI100), dl1 instances (Gaudi 1)
#     Note: Gaudi 3 availability depends on your cloud provider agreements
#   - On-premise: Intel Developer Cloud (IDC)
#   - Habana Labs partner ecosystem
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  # Uncomment and configure a real backend for production use:
  # backend "s3" {
  #   bucket = "my-terraform-state"
  #   key    = "gaudi3-scale/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

# =============================================================================
# Provider Configuration (placeholder — replace with your cloud provider)
# =============================================================================

# Example: AWS provider
# provider "aws" {
#   region = var.region
# }

# Example: For Intel Developer Cloud, use the appropriate provider
# provider "intel" {
#   api_key = var.intel_api_key
# }

# =============================================================================
# Networking
# =============================================================================

# Virtual Private Cloud (VPC) for the training cluster
# resource "aws_vpc" "gaudi_vpc" {
#   cidr_block           = var.vpc_cidr
#   enable_dns_hostnames = true
#   enable_dns_support   = true
#
#   tags = {
#     Name        = "${var.cluster_name}-vpc"
#     Environment = var.environment
#     Project     = "gaudi3-scale"
#   }
# }

# Subnet for Gaudi 3 instances
# resource "aws_subnet" "gaudi_subnet" {
#   vpc_id            = aws_vpc.gaudi_vpc.id
#   cidr_block        = var.subnet_cidr
#   availability_zone = var.availability_zone
#
#   tags = {
#     Name = "${var.cluster_name}-subnet"
#   }
# }

# =============================================================================
# Security Group
# =============================================================================

# resource "aws_security_group" "gaudi_sg" {
#   name_prefix = "${var.cluster_name}-sg"
#   vpc_id      = aws_vpc.gaudi_vpc.id
#
#   # Allow SSH access for debugging
#   ingress {
#     from_port   = 22
#     to_port     = 22
#     protocol    = "tcp"
#     cidr_blocks = [var.allowed_ssh_cidr]
#   }
#
#   # Allow inter-node communication (RDMA / HPU interconnect)
#   ingress {
#     from_port = 0
#     to_port   = 0
#     protocol  = "-1"
#     self      = true
#   }
#
#   egress {
#     from_port   = 0
#     to_port     = 0
#     protocol    = "-1"
#     cidr_blocks = ["0.0.0.0/0"]
#   }
# }

# =============================================================================
# Gaudi 3 Training Cluster
# =============================================================================

# resource "aws_instance" "gaudi_node" {
#   count         = var.num_nodes
#   ami           = var.gaudi_ami_id     # Habana Deep Learning AMI
#   instance_type = var.instance_type    # e.g., dl1.24xlarge (Gaudi 1) or future Gaudi 3 type
#   subnet_id     = aws_subnet.gaudi_subnet.id
#
#   vpc_security_group_ids = [aws_security_group.gaudi_sg.id]
#   key_name               = var.ssh_key_name
#
#   # Root volume — large enough for model checkpoints
#   root_block_device {
#     volume_size = var.root_volume_size_gb
#     volume_type = "gp3"
#     iops        = 3000
#     throughput  = 125
#   }
#
#   # User data script to set up the environment
#   user_data = templatefile("${path.module}/scripts/setup.sh.tpl", {
#     wandb_api_key = var.wandb_api_key
#     s3_bucket     = var.checkpoint_bucket
#   })
#
#   tags = {
#     Name        = "${var.cluster_name}-node-${count.index}"
#     Environment = var.environment
#     Project     = "gaudi3-scale"
#     Role        = count.index == 0 ? "master" : "worker"
#   }
# }

# =============================================================================
# Storage for Checkpoints
# =============================================================================

# resource "aws_s3_bucket" "checkpoints" {
#   bucket = "${var.cluster_name}-checkpoints-${var.environment}"
#
#   tags = {
#     Name    = "gaudi3-checkpoints"
#     Project = "gaudi3-scale"
#   }
# }

# resource "aws_s3_bucket_versioning" "checkpoints" {
#   bucket = aws_s3_bucket.checkpoints.id
#   versioning_configuration {
#     status = "Enabled"
#   }
# }

# =============================================================================
# Outputs
# =============================================================================

# output "master_node_ip" {
#   description = "Public IP of the master training node"
#   value       = aws_instance.gaudi_node[0].public_ip
# }
#
# output "worker_node_ips" {
#   description = "Public IPs of worker nodes"
#   value       = aws_instance.gaudi_node[*].public_ip
# }
#
# output "checkpoint_bucket" {
#   description = "S3 bucket for model checkpoints"
#   value       = aws_s3_bucket.checkpoints.bucket
# }
