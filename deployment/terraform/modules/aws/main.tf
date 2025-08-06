# AWS Infrastructure Module for Gaudi 3 Scale
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-vpc"
    Type = "networking"
  })
}

resource "aws_subnet" "private" {
  count             = var.availability_zones_count
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-private-${count.index + 1}"
    Type = "private"
    "kubernetes.io/role/internal-elb" = "1"
  })
}

resource "aws_subnet" "public" {
  count                   = var.availability_zones_count
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 100)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-public-${count.index + 1}"
    Type = "public"
    "kubernetes.io/role/elb" = "1"
  })
}

# Internet Gateway and NAT Gateways
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-igw"
  })
}

resource "aws_eip" "nat" {
  count  = var.availability_zones_count
  domain = "vpc"

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-nat-eip-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  count         = var.availability_zones_count
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-nat-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count  = var.availability_zones_count
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-private-rt-${count.index + 1}"
  })
}

resource "aws_route_table_association" "public" {
  count          = var.availability_zones_count
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = var.availability_zones_count
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "cluster" {
  name_prefix = "${var.cluster_name}-cluster-"
  vpc_id      = aws_vpc.main.id
  description = "EKS cluster security group"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-cluster-sg"
  })
}

resource "aws_security_group" "nodes" {
  name_prefix = "${var.cluster_name}-nodes-"
  vpc_id      = aws_vpc.main.id
  description = "EKS node group security group"

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.cluster.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-nodes-sg"
  })
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    security_group_ids      = [aws_security_group.cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = var.cluster_endpoint_public_access
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  encryption_config {
    provider {
      key_arn = aws_kms_key.cluster.arn
    }
    resources = ["secrets"]
  }

  tags = var.common_tags

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSVPCResourceController,
    aws_cloudwatch_log_group.cluster,
  ]
}

# EKS Node Groups
resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "cpu-nodes"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = var.cpu_instance_types

  scaling_config {
    desired_size = var.cpu_desired_capacity
    max_size     = var.cpu_max_capacity
    min_size     = var.cpu_min_capacity
  }

  update_config {
    max_unavailable = 1
  }

  launch_template {
    id      = aws_launch_template.cpu_nodes.id
    version = aws_launch_template.cpu_nodes.latest_version
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-cpu-nodes"
    Type = "cpu-compute"
  })

  depends_on = [
    aws_iam_role_policy_attachment.nodes_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodes_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodes_AmazonEC2ContainerRegistryReadOnly,
  ]
}

# Gaudi HPU Node Group (if enabled)
resource "aws_eks_node_group" "hpu_nodes" {
  count = var.enable_gaudi_nodes ? 1 : 0

  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gaudi-hpu-nodes"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = var.gaudi_instance_types

  scaling_config {
    desired_size = var.gaudi_desired_capacity
    max_size     = var.gaudi_max_capacity
    min_size     = var.gaudi_min_capacity
  }

  update_config {
    max_unavailable = 1
  }

  launch_template {
    id      = aws_launch_template.hpu_nodes[0].id
    version = aws_launch_template.hpu_nodes[0].latest_version
  }

  taint {
    key    = "gaudi.habana.ai/gaudi"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-gaudi-hpu-nodes"
    Type = "hpu-compute"
  })

  depends_on = [
    aws_iam_role_policy_attachment.nodes_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodes_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodes_AmazonEC2ContainerRegistryReadOnly,
  ]
}

# Launch Templates
resource "aws_launch_template" "cpu_nodes" {
  name_prefix   = "${var.cluster_name}-cpu-"
  image_id      = var.cpu_ami_id
  instance_type = var.cpu_instance_types[0]

  vpc_security_group_ids = [aws_security_group.nodes.id]

  user_data = base64encode(templatefile("${path.module}/user_data/cpu_nodes.sh", {
    cluster_name        = aws_eks_cluster.main.name
    cluster_endpoint    = aws_eks_cluster.main.endpoint
    cluster_ca_data     = aws_eks_cluster.main.certificate_authority[0].data
    bootstrap_arguments = var.cpu_bootstrap_arguments
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.common_tags, {
      Name = "${var.cluster_name}-cpu-node"
    })
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = var.cpu_disk_size
      volume_type = "gp3"
      encrypted   = true
      kms_key_id  = aws_kms_key.cluster.arn
    }
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
}

resource "aws_launch_template" "hpu_nodes" {
  count = var.enable_gaudi_nodes ? 1 : 0

  name_prefix   = "${var.cluster_name}-gaudi-"
  image_id      = var.gaudi_ami_id
  instance_type = var.gaudi_instance_types[0]

  vpc_security_group_ids = [aws_security_group.nodes.id]

  user_data = base64encode(templatefile("${path.module}/user_data/gaudi_nodes.sh", {
    cluster_name        = aws_eks_cluster.main.name
    cluster_endpoint    = aws_eks_cluster.main.endpoint
    cluster_ca_data     = aws_eks_cluster.main.certificate_authority[0].data
    bootstrap_arguments = var.gaudi_bootstrap_arguments
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.common_tags, {
      Name = "${var.cluster_name}-gaudi-node"
    })
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = var.gaudi_disk_size
      volume_type = "gp3"
      encrypted   = true
      kms_key_id  = aws_kms_key.cluster.arn
    }
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
}

# KMS Key for encryption
resource "aws_kms_key" "cluster" {
  description         = "EKS cluster encryption key"
  deletion_window_in_days = 7
  enable_key_rotation = true

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-encryption-key"
  })
}

resource "aws_kms_alias" "cluster" {
  name          = "alias/${var.cluster_name}-encryption-key"
  target_key_id = aws_kms_key.cluster.key_id
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.cluster.arn

  tags = var.common_tags
}