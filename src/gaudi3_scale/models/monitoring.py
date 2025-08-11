"""Monitoring and metrics models."""

from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for environments without pydantic
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


class MetricType(str, Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricConfig(BaseModel):
    """Configuration for a single metric."""
    
    name: str = Field(..., description="Metric name")
    type: MetricType = Field(..., description="Metric type")
    description: str = Field(..., description="Metric description")
    labels: List[str] = Field(default_factory=list, description="Metric labels")
    unit: Optional[str] = Field(None, description="Metric unit")
    
    # Collection settings
    collection_interval: int = Field(15, description="Collection interval in seconds")
    retention_period: int = Field(7, description="Retention period in days")
    
    # Alerting thresholds
    warning_threshold: Optional[float] = Field(None, description="Warning threshold")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold")
    
    @validator('collection_interval')
    def validate_interval(cls, v):
        if v < 1 or v > 3600:
            raise ValueError('Collection interval must be between 1 and 3600 seconds')
        return v


class HPUMetrics(BaseModel):
    """HPU-specific metrics."""
    
    # Utilization metrics
    hpu_utilization: float = Field(..., description="HPU utilization percentage")
    memory_usage: float = Field(..., description="Memory usage in GB")
    memory_utilization: float = Field(..., description="Memory utilization percentage")
    
    # Performance metrics  
    throughput_tokens_per_sec: float = Field(..., description="Training throughput")
    effective_batch_size: int = Field(..., description="Effective batch size")
    steps_per_second: float = Field(..., description="Training steps per second")
    
    # Temperature and power
    temperature_celsius: float = Field(..., description="HPU temperature")
    power_consumption_watts: float = Field(..., description="Power consumption")
    
    # Graph compilation
    graph_compilation_time_ms: float = Field(..., description="Graph compilation time")
    lazy_mode_enabled: bool = Field(..., description="Lazy mode status")
    
    # Training metrics
    current_epoch: int = Field(..., description="Current training epoch")
    current_step: int = Field(..., description="Current training step")
    loss_value: Optional[float] = Field(None, description="Current loss value")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Metric timestamp")
    
    @validator('hpu_utilization', 'memory_utilization')
    def validate_percentage(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Utilization must be between 0 and 100')
        return v


class SystemMetrics(BaseModel):
    """System-level metrics."""
    
    # CPU metrics
    cpu_utilization: float = Field(..., description="CPU utilization percentage")
    cpu_load_avg: float = Field(..., description="CPU load average")
    
    # Memory metrics
    memory_total_gb: float = Field(..., description="Total system memory")
    memory_used_gb: float = Field(..., description="Used system memory")
    memory_available_gb: float = Field(..., description="Available system memory")
    
    # Disk metrics
    disk_usage_gb: float = Field(..., description="Disk usage in GB")
    disk_io_read_mbps: float = Field(..., description="Disk read speed")
    disk_io_write_mbps: float = Field(..., description="Disk write speed")
    
    # Network metrics
    network_rx_mbps: float = Field(..., description="Network receive speed")
    network_tx_mbps: float = Field(..., description="Network transmit speed")
    network_latency_ms: float = Field(..., description="Network latency")
    
    # Process metrics
    process_count: int = Field(..., description="Number of processes")
    open_file_descriptors: int = Field(..., description="Open file descriptors")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Metric timestamp")


class AlertRule(BaseModel):
    """Alert rule configuration."""
    
    name: str = Field(..., description="Alert rule name")
    severity: AlertSeverity = Field(..., description="Alert severity")
    metric_name: str = Field(..., description="Target metric name")
    
    # Condition
    condition: str = Field(..., description="Alert condition (e.g., '> 90')")
    duration: int = Field(300, description="Duration threshold in seconds")
    
    # Notification
    notification_channels: List[str] = Field(
        default_factory=list, 
        description="Notification channels"
    )
    message_template: Optional[str] = Field(None, description="Custom alert message")
    
    # Metadata
    enabled: bool = Field(True, description="Alert rule enabled")
    labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")


class HealthCheck(BaseModel):
    """Health check configuration and status."""
    
    service_name: str = Field(..., description="Service name")
    check_type: str = Field(..., description="Health check type")
    endpoint: Optional[str] = Field(None, description="Health check endpoint")
    
    # Check configuration
    interval: int = Field(30, description="Check interval in seconds")
    timeout: int = Field(10, description="Check timeout in seconds")
    retries: int = Field(3, description="Number of retries")
    
    # Status
    status: HealthStatus = Field(HealthStatus.UNKNOWN, description="Current status")
    last_check: Optional[datetime] = Field(None, description="Last check timestamp")
    last_success: Optional[datetime] = Field(None, description="Last successful check")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict, description="Health check tags")
    
    @validator('interval', 'timeout')
    def validate_timing(cls, v):
        if v < 1:
            raise ValueError('Timing values must be positive')
        return v


class MetricsConfig(BaseModel):
    """Complete metrics and monitoring configuration."""
    
    # Collection settings
    enabled: bool = Field(True, description="Enable metrics collection")
    collection_interval: int = Field(15, description="Default collection interval")
    retention_days: int = Field(30, description="Metrics retention period")
    
    # Storage
    storage_backend: str = Field("prometheus", description="Metrics storage backend")
    storage_path: str = Field("/var/lib/prometheus", description="Storage path")
    
    # HPU metrics
    hpu_metrics_enabled: bool = Field(True, description="Enable HPU metrics")
    hpu_collection_interval: int = Field(10, description="HPU metrics interval")
    
    # System metrics
    system_metrics_enabled: bool = Field(True, description="Enable system metrics")
    system_collection_interval: int = Field(30, description="System metrics interval")
    
    # Training metrics
    training_metrics_enabled: bool = Field(True, description="Enable training metrics")
    training_collection_interval: int = Field(5, description="Training metrics interval")
    
    # Custom metrics
    custom_metrics: List[MetricConfig] = Field(
        default_factory=list,
        description="Custom metric configurations"
    )
    
    # Alerting
    alerting_enabled: bool = Field(True, description="Enable alerting")
    alert_rules: List[AlertRule] = Field(
        default_factory=list,
        description="Alert rule configurations"
    )
    
    # Health checks
    health_checks: List[HealthCheck] = Field(
        default_factory=list,
        description="Health check configurations"
    )
    
    # Visualization
    grafana_enabled: bool = Field(True, description="Enable Grafana dashboards")
    grafana_port: int = Field(3000, description="Grafana port")
    
    # Notification channels
    notification_channels: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Notification channel configurations"
    )
    
    def get_default_alert_rules(self) -> List[AlertRule]:
        """Get default alert rules for Gaudi 3 infrastructure."""
        return [
            AlertRule(
                name="HighHPUUtilization",
                severity=AlertSeverity.WARNING,
                metric_name="hpu_utilization",
                condition="> 95",
                duration=300,
                notification_channels=["slack", "email"]
            ),
            AlertRule(
                name="HighMemoryUsage",
                severity=AlertSeverity.CRITICAL,
                metric_name="memory_utilization",
                condition="> 90",
                duration=180,
                notification_channels=["slack", "email", "pagerduty"]
            ),
            AlertRule(
                name="LowTrainingThroughput",
                severity=AlertSeverity.WARNING,
                metric_name="throughput_tokens_per_sec",
                condition="< 100",
                duration=600,
                notification_channels=["slack"]
            ),
            AlertRule(
                name="HighTemperature",
                severity=AlertSeverity.CRITICAL,
                metric_name="temperature_celsius",
                condition="> 85",
                duration=60,
                notification_channels=["slack", "email", "pagerduty"]
            )
        ]