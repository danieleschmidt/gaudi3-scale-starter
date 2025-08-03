"""SQLAlchemy database models for Gaudi 3 Scale."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)


class ClusterModel(Base, TimestampMixin):
    """Database model for Gaudi 3 clusters."""
    
    __tablename__ = "clusters"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic cluster information
    name = Column(String(255), nullable=False, unique=True)
    provider = Column(String(50), nullable=False)  # aws, azure, gcp, onprem
    region = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="pending")  # pending, running, stopped, error
    
    # Configuration
    node_count = Column(Integer, nullable=False, default=1)
    instance_type = Column(String(100), nullable=False)
    total_hpus = Column(Integer, nullable=False)
    
    # Network configuration
    vpc_id = Column(String(255), nullable=True)
    subnet_ids = Column(JSON, nullable=True)
    security_group_ids = Column(JSON, nullable=True)
    
    # Cost tracking
    estimated_cost_per_hour = Column(Float, nullable=False, default=0.0)
    actual_cost_to_date = Column(Float, nullable=False, default=0.0)
    
    # Feature flags
    enable_monitoring = Column(Boolean, nullable=False, default=True)
    enable_spot_instances = Column(Boolean, nullable=False, default=False)
    enable_auto_scaling = Column(Boolean, nullable=False, default=False)
    
    # Metadata
    tags = Column(JSON, nullable=True)
    terraform_state_url = Column(String(500), nullable=True)
    
    # Health information
    last_health_check = Column(DateTime(timezone=True), nullable=True)
    health_status = Column(String(50), nullable=False, default="unknown")
    
    # Relationships
    training_jobs = relationship("TrainingJobModel", back_populates="cluster")
    metrics = relationship("MetricModel", back_populates="cluster")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('node_count > 0', name='check_positive_node_count'),
        CheckConstraint('total_hpus > 0', name='check_positive_hpu_count'),
        CheckConstraint('estimated_cost_per_hour >= 0', name='check_non_negative_cost'),
        Index('idx_cluster_provider_region', 'provider', 'region'),
        Index('idx_cluster_status', 'status'),
        Index('idx_cluster_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Cluster(name='{self.name}', provider='{self.provider}', status='{self.status}')>"


class TrainingJobModel(Base, TimestampMixin):
    """Database model for training jobs."""
    
    __tablename__ = "training_jobs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to cluster
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=False)
    
    # Job information
    name = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    model_size = Column(String(50), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    
    # Job status
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed, cancelled
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Training configuration
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    max_epochs = Column(Integer, nullable=False)
    precision = Column(String(20), nullable=False, default="bf16-mixed")
    
    # Resource usage
    devices_used = Column(Integer, nullable=False, default=8)
    peak_memory_usage_gb = Column(Float, nullable=True)
    peak_hpu_utilization = Column(Float, nullable=True)
    
    # Progress tracking
    current_epoch = Column(Integer, nullable=False, default=0)
    current_step = Column(Integer, nullable=False, default=0)
    total_steps = Column(Integer, nullable=True)
    
    # Performance metrics
    tokens_per_second = Column(Float, nullable=True)
    steps_per_second = Column(Float, nullable=True)
    current_loss = Column(Float, nullable=True)
    
    # Paths and outputs
    checkpoint_path = Column(String(500), nullable=True)
    output_path = Column(String(500), nullable=True)
    log_path = Column(String(500), nullable=True)
    
    # Experiment tracking
    wandb_run_id = Column(String(255), nullable=True)
    tensorboard_log_dir = Column(String(500), nullable=True)
    
    # Configuration as JSON
    training_config = Column(JSON, nullable=True)
    model_config = Column(JSON, nullable=True)
    dataset_config = Column(JSON, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)
    
    # Relationships
    cluster = relationship("ClusterModel", back_populates="training_jobs")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('batch_size > 0', name='check_positive_batch_size'),
        CheckConstraint('learning_rate > 0', name='check_positive_learning_rate'),
        CheckConstraint('max_epochs > 0', name='check_positive_epochs'),
        CheckConstraint('devices_used > 0', name='check_positive_devices'),
        CheckConstraint('current_epoch >= 0', name='check_non_negative_epoch'),
        CheckConstraint('current_step >= 0', name='check_non_negative_step'),
        Index('idx_training_job_cluster_id', 'cluster_id'),
        Index('idx_training_job_status', 'status'),
        Index('idx_training_job_model', 'model_name', 'model_size'),
        Index('idx_training_job_created_at', 'created_at'),
        UniqueConstraint('cluster_id', 'name', name='uq_cluster_job_name'),
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active."""
        return self.status in ("pending", "running")
    
    def __repr__(self):
        return f"<TrainingJob(name='{self.name}', model='{self.model_name}', status='{self.status}')>"


class MetricModel(Base, TimestampMixin):
    """Database model for storing metrics and telemetry data."""
    
    __tablename__ = "metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to cluster
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=False)
    
    # Optional foreign key to training job
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=True)
    
    # Metric identification
    metric_name = Column(String(255), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram, summary
    
    # Metric value and metadata
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    labels = Column(JSON, nullable=True)  # Additional labels as key-value pairs
    
    # Source information
    source_node = Column(String(255), nullable=True)
    source_hpu = Column(Integer, nullable=True)
    
    # Timestamp for the metric (different from created_at)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    cluster = relationship("ClusterModel", back_populates="metrics")
    training_job = relationship("TrainingJobModel", backref="metrics")
    
    # Constraints
    __table_args__ = (
        Index('idx_metric_cluster_name_timestamp', 'cluster_id', 'metric_name', 'timestamp'),
        Index('idx_metric_training_job', 'training_job_id'),
        Index('idx_metric_timestamp', 'timestamp'),
        Index('idx_metric_name_type', 'metric_name', 'metric_type'),
    )
    
    def __repr__(self):
        return f"<Metric(name='{self.metric_name}', value={self.value}, timestamp='{self.timestamp}')>"


class NodeModel(Base, TimestampMixin):
    """Database model for individual cluster nodes."""
    
    __tablename__ = "nodes"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to cluster
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=False)
    
    # Node identification
    node_name = Column(String(255), nullable=False)
    instance_id = Column(String(255), nullable=True)  # Cloud provider instance ID
    private_ip = Column(String(45), nullable=True)    # IPv4 or IPv6
    public_ip = Column(String(45), nullable=True)
    
    # Node specifications
    instance_type = Column(String(100), nullable=False)
    hpu_count = Column(Integer, nullable=False, default=8)
    memory_gb = Column(Integer, nullable=False)
    storage_gb = Column(Integer, nullable=False)
    
    # Node status
    status = Column(String(50), nullable=False, default="pending")  # pending, running, stopped, error
    availability_zone = Column(String(100), nullable=True)
    
    # Performance metrics
    cpu_utilization = Column(Float, nullable=True)
    memory_utilization = Column(Float, nullable=True)
    average_hpu_utilization = Column(Float, nullable=True)
    temperature_celsius = Column(Float, nullable=True)
    
    # Health information
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    health_status = Column(String(50), nullable=False, default="unknown")
    
    # Metadata
    labels = Column(JSON, nullable=True)
    
    # Relationships
    cluster = relationship("ClusterModel", backref="nodes")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('hpu_count > 0', name='check_positive_hpu_count_node'),
        CheckConstraint('memory_gb > 0', name='check_positive_memory'),
        CheckConstraint('storage_gb > 0', name='check_positive_storage'),
        Index('idx_node_cluster_id', 'cluster_id'),
        Index('idx_node_status', 'status'),
        Index('idx_node_health', 'health_status'),
        UniqueConstraint('cluster_id', 'node_name', name='uq_cluster_node_name'),
    )
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return self.health_status == "healthy" and self.status == "running"
    
    def __repr__(self):
        return f"<Node(name='{self.node_name}', cluster='{self.cluster_id}', status='{self.status}')>"


class AlertModel(Base, TimestampMixin):
    """Database model for alerts and notifications."""
    
    __tablename__ = "alerts"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Alert identification
    alert_name = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False)  # info, warning, critical, fatal
    
    # Alert source
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=True)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=True)
    source_node = Column(String(255), nullable=True)
    
    # Alert details
    metric_name = Column(String(255), nullable=True)
    threshold_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    condition = Column(String(255), nullable=True)
    
    # Alert status
    status = Column(String(50), nullable=False, default="firing")  # firing, resolved, silenced
    first_seen = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_seen = Column(DateTime(timezone=True), nullable=False, default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Alert content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    runbook_url = Column(String(500), nullable=True)
    
    # Notification tracking
    notification_sent = Column(Boolean, nullable=False, default=False)
    notification_channels = Column(JSON, nullable=True)
    
    # Metadata
    labels = Column(JSON, nullable=True)
    annotations = Column(JSON, nullable=True)
    
    # Relationships
    cluster = relationship("ClusterModel", backref="alerts")
    training_job = relationship("TrainingJobModel", backref="alerts")
    
    # Constraints
    __table_args__ = (
        Index('idx_alert_cluster_id', 'cluster_id'),
        Index('idx_alert_status_severity', 'status', 'severity'),
        Index('idx_alert_first_seen', 'first_seen'),
        Index('idx_alert_name', 'alert_name'),
    )
    
    @property
    def duration_seconds(self) -> float:
        """Calculate alert duration in seconds."""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.first_seen).total_seconds()
    
    def __repr__(self):
        return f"<Alert(name='{self.alert_name}', severity='{self.severity}', status='{self.status}')>"