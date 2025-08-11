"""Cluster configuration models."""

from enum import Enum
from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for environments without pydantic
    from typing import Any
    
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ONPREM = "onprem"


class InstanceType(str, Enum):
    """Supported instance types for Gaudi 3 deployment."""
    # AWS
    AWS_DL2Q_24XLARGE = "dl2q.24xlarge"
    AWS_DL2Q_48XLARGE = "dl2q.48xlarge"
    
    # Azure
    AZURE_HX176RS = "Standard_HX176rs"
    AZURE_HX352RS = "Standard_HX352rs"
    
    # GCP
    GCP_A2_ULTRAGPU_8G = "a2-ultragpu-8g"
    
    # On-premises
    ONPREM_GAUDI3_NODE = "gaudi3-node"


class NodeConfig(BaseModel):
    """Configuration for a single node in the cluster."""
    
    node_id: str = Field(..., description="Unique node identifier")
    instance_type: InstanceType = Field(..., description="Instance type")
    hpu_count: int = Field(8, description="Number of HPUs per node")
    memory_gb: int = Field(512, description="Memory in GB")
    storage_gb: int = Field(1000, description="Storage in GB")
    network_bandwidth_gbps: int = Field(200, description="Network bandwidth in Gbps")
    availability_zone: Optional[str] = Field(None, description="Availability zone")
    
    @validator('hpu_count')
    def validate_hpu_count(cls, v):
        if v not in [8, 16]:
            raise ValueError('HPU count must be 8 or 16')
        return v


class NetworkConfig(BaseModel):
    """Network configuration for the cluster."""
    
    vpc_cidr: str = Field("10.0.0.0/16", description="VPC CIDR block")
    subnet_cidrs: List[str] = Field(
        default_factory=lambda: ["10.0.1.0/24", "10.0.2.0/24"],
        description="Subnet CIDR blocks"
    )
    enable_efa: bool = Field(True, description="Enable Elastic Fabric Adapter")
    enable_sr_iov: bool = Field(True, description="Enable SR-IOV")
    security_group_rules: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "ingress": ["22", "6443", "8080-8090"],
            "egress": ["all"]
        },
        description="Security group rules"
    )


class StorageConfig(BaseModel):
    """Storage configuration for the cluster."""
    
    root_volume_size_gb: int = Field(100, description="Root volume size in GB")
    data_volume_size_gb: int = Field(1000, description="Data volume size in GB")
    volume_type: str = Field("gp3", description="EBS volume type")
    iops: int = Field(3000, description="IOPS for volumes")
    throughput_mbps: int = Field(125, description="Throughput in MB/s")
    enable_encryption: bool = Field(True, description="Enable volume encryption")


class ClusterConfig(BaseModel):
    """Complete cluster configuration."""
    
    cluster_name: str = Field(..., description="Cluster name")
    provider: CloudProvider = Field(..., description="Cloud provider")
    region: str = Field(..., description="Deployment region")
    nodes: List[NodeConfig] = Field(..., description="Node configurations")
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Monitoring and logging
    enable_monitoring: bool = Field(True, description="Enable monitoring stack")
    enable_logging: bool = Field(True, description="Enable centralized logging")
    log_retention_days: int = Field(30, description="Log retention period")
    
    # Cost optimization
    enable_spot_instances: bool = Field(False, description="Use spot instances")
    auto_scaling_enabled: bool = Field(False, description="Enable auto-scaling")
    max_nodes: int = Field(16, description="Maximum number of nodes")
    min_nodes: int = Field(1, description="Minimum number of nodes")
    
    # Security
    enable_encryption_at_rest: bool = Field(True, description="Enable encryption at rest")
    enable_encryption_in_transit: bool = Field(True, description="Enable encryption in transit")
    ssh_key_name: Optional[str] = Field(None, description="SSH key name")
    
    # Tags
    tags: Dict[str, str] = Field(
        default_factory=lambda: {
            "Project": "Gaudi3Scale",
            "Environment": "production",
            "ManagedBy": "terraform"
        },
        description="Resource tags"
    )
    
    @validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            raise ValueError('At least one node must be configured')
        if len(v) > 64:
            raise ValueError('Maximum 64 nodes supported')
        return v
    
    @property
    def total_hpus(self) -> int:
        """Calculate total HPUs in the cluster."""
        return sum(node.hpu_count for node in self.nodes)
    
    @property
    def estimated_cost_per_hour(self) -> float:
        """Estimate hourly cost based on instance types."""
        cost_map = {
            InstanceType.AWS_DL2Q_24XLARGE: 32.77,
            InstanceType.AWS_DL2Q_48XLARGE: 65.54,
            InstanceType.AZURE_HX176RS: 45.50,
            InstanceType.AZURE_HX352RS: 91.00,
            InstanceType.GCP_A2_ULTRAGPU_8G: 38.20,
            InstanceType.ONPREM_GAUDI3_NODE: 0.00
        }
        
        total_cost = 0.0
        for node in self.nodes:
            base_cost = cost_map.get(node.instance_type, 30.0)
            if self.enable_spot_instances and self.provider != CloudProvider.ONPREM:
                base_cost *= 0.3  # Spot instance discount
            total_cost += base_cost
        
        return total_cost
    
    def to_terraform_vars(self) -> Dict[str, any]:
        """Convert to Terraform variables format."""
        return {
            "cluster_name": self.cluster_name,
            "region": self.region,
            "node_count": len(self.nodes),
            "instance_type": self.nodes[0].instance_type.value if self.nodes else "",
            "hpu_count": self.total_hpus,
            "enable_monitoring": self.enable_monitoring,
            "enable_spot_instances": self.enable_spot_instances,
            "vpc_cidr": self.network.vpc_cidr,
            "subnet_cidrs": self.network.subnet_cidrs,
            "tags": self.tags
        }