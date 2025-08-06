"""Distributed monitoring and observability for Gaudi 3 clusters."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
import uuid
import statistics
from collections import defaultdict, deque

from .discovery import ServiceRegistry, ServiceInfo, ServiceType, ServiceStatus
from .coordinator import DistributedTrainingCoordinator
from .storage import DataManager
from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"         # Monotonic increasing counter
    GAUGE = "gauge"            # Point-in-time value
    HISTOGRAM = "histogram"    # Distribution of values
    SUMMARY = "summary"        # Statistical summary
    TIMER = "timer"           # Duration measurements


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status states."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Metric:
    """Represents a metric data point."""
    name: str
    metric_type: MetricType
    value: Union[float, int]
    labels: Dict[str, str]
    timestamp: datetime
    unit: str = ""
    help_text: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Alert:
    """Represents an alert condition."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    labels: Dict[str, str]
    annotations: Dict[str, str]
    started_at: datetime
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    
    @property
    def duration(self) -> timedelta:
        end_time = self.resolved_at or datetime.now()
        return end_time - self.started_at
    
    @property
    def is_active(self) -> bool:
        return self.status == AlertStatus.FIRING and (
            self.suppressed_until is None or 
            datetime.now() > self.suppressed_until
        )


@dataclass
class TraceSpan:
    """Represents a distributed trace span."""
    span_id: str
    trace_id: str
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_span_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def add_tag(self, key: str, value: str):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def finish(self):
        """Mark the span as finished."""
        if self.end_time is None:
            self.end_time = datetime.now()


@dataclass 
class DistributedTrace:
    """Represents a complete distributed trace."""
    trace_id: str
    spans: Dict[str, TraceSpan]
    started_at: datetime
    service_names: Set[str] = field(default_factory=set)
    
    @property
    def duration_ms(self) -> Optional[float]:
        if not self.spans:
            return None
        
        earliest_start = min(span.start_time for span in self.spans.values())
        latest_end = max(
            span.end_time for span in self.spans.values() 
            if span.end_time is not None
        )
        
        if latest_end:
            return (latest_end - earliest_start).total_seconds() * 1000
        return None
    
    def add_span(self, span: TraceSpan):
        """Add a span to the trace."""
        self.spans[span.span_id] = span
        self.service_names.add(span.service_name)


class MetricsCollector:
    """Collects and aggregates metrics from distributed services."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.retention_hours = retention_hours
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata: Dict[str, Dict[str, str]] = {}
        
        # Aggregated metrics for performance
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_aggregation: datetime = datetime.now()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_metrics())
    
    def record_metric(self, metric: Metric):
        """Record a metric data point.
        
        Args:
            metric: Metric to record
        """
        metric_key = self._get_metric_key(metric)
        self.metrics[metric_key].append(metric)
        
        # Store metadata
        self.metric_metadata[metric_key] = {
            "type": metric.metric_type.value,
            "unit": metric.unit,
            "help": metric.help_text
        }
    
    def get_metric_values(self, 
                         name: str, 
                         labels: Optional[Dict[str, str]] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Metric]:
        """Get metric values within time range.
        
        Args:
            name: Metric name
            labels: Label filters
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of matching metrics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=1)
        
        matching_metrics = []
        
        for metric_key, metric_deque in self.metrics.items():
            # Extract name from key
            key_name = metric_key.split('{')[0]
            if key_name != name:
                continue
            
            for metric in metric_deque:
                # Time filter
                if not (start_time <= metric.timestamp <= end_time):
                    continue
                
                # Label filter
                if labels:
                    if not all(metric.labels.get(k) == v for k, v in labels.items()):
                        continue
                
                matching_metrics.append(metric)
        
        return sorted(matching_metrics, key=lambda m: m.timestamp)
    
    def get_aggregated_metrics(self, 
                              name: str,
                              aggregation: str = "avg",
                              window_minutes: int = 5) -> Optional[float]:
        """Get aggregated metric value.
        
        Args:
            name: Metric name
            aggregation: Type of aggregation (avg, sum, min, max, count)
            window_minutes: Time window for aggregation
            
        Returns:
            Aggregated value or None
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        metrics = self.get_metric_values(name, start_time=start_time, end_time=end_time)
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        elif aggregation == "median":
            return statistics.median(values)
        elif aggregation == "p95":
            return statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0]
        elif aggregation == "p99":
            return statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0]
        else:
            return statistics.mean(values)
    
    def _get_metric_key(self, metric: Metric) -> str:
        """Generate unique key for metric."""
        if metric.labels:
            labels_str = "{" + ",".join(f"{k}={v}" for k, v in sorted(metric.labels.items())) + "}"
            return f"{metric.name}{labels_str}"
        return metric.name
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                for metric_key, metric_deque in self.metrics.items():
                    # Remove old metrics from the front of the deque
                    while metric_deque and metric_deque[0].timestamp < cutoff_time:
                        metric_deque.popleft()
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(60)


class AlertManager:
    """Manages alerts based on metric thresholds and conditions."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Alert state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable[[Alert], None]] = []
        
        # Alert grouping and suppression
        self.alert_groups: Dict[str, List[str]] = {}
        self.suppression_rules: List[Dict[str, Any]] = []
        
        # Start monitoring task
        asyncio.create_task(self._evaluate_alert_rules())
    
    def add_alert_rule(self, 
                      rule_name: str,
                      metric_name: str,
                      threshold: float,
                      comparison: str = "gt",
                      severity: AlertSeverity = AlertSeverity.WARNING,
                      window_minutes: int = 5,
                      description: str = "",
                      labels: Optional[Dict[str, str]] = None,
                      annotations: Optional[Dict[str, str]] = None):
        """Add an alert rule.
        
        Args:
            rule_name: Name of the alert rule
            metric_name: Metric to monitor
            threshold: Threshold value
            comparison: Comparison operator (gt, lt, eq, ge, le, ne)
            severity: Alert severity
            window_minutes: Time window for evaluation
            description: Alert description
            labels: Alert labels
            annotations: Alert annotations
        """
        self.alert_rules[rule_name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "window_minutes": window_minutes,
            "description": description,
            "labels": labels or {},
            "annotations": annotations or {}
        }
        
        self.logger.info(f"Added alert rule: {rule_name}")
    
    def add_notification_channel(self, channel: Callable[[Alert], None]):
        """Add notification channel for alerts.
        
        Args:
            channel: Function to handle alert notifications
        """
        self.notification_channels.append(channel)
    
    async def fire_alert(self, 
                        name: str,
                        description: str,
                        severity: AlertSeverity,
                        labels: Optional[Dict[str, str]] = None,
                        annotations: Optional[Dict[str, str]] = None) -> str:
        """Fire an alert manually.
        
        Args:
            name: Alert name
            description: Alert description
            severity: Alert severity
            labels: Alert labels
            annotations: Alert annotations
            
        Returns:
            Alert ID
        """
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            alert_id=alert_id,
            name=name,
            description=description,
            severity=severity,
            status=AlertStatus.FIRING,
            labels=labels or {},
            annotations=annotations or {},
            started_at=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        self.logger.warning(f"Alert fired: {name} ({severity.value})")
        return alert_id
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            del self.active_alerts[alert_id]
            
            # Send resolution notifications
            await self._send_notifications(alert)
            
            self.logger.info(f"Alert resolved: {alert.name} (duration: {alert.duration})")
    
    def suppress_alert(self, alert_id: str, duration_minutes: int):
        """Suppress an alert for a specified duration.
        
        Args:
            alert_id: Alert identifier
            duration_minutes: Suppression duration in minutes
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)
            
            self.logger.info(f"Alert suppressed: {alert.name} for {duration_minutes} minutes")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.started_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics.
        
        Returns:
            Alert statistics dictionary
        """
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] = \
                severity_counts.get(alert.severity.value, 0) + 1
        
        # Recent alerts (last 24 hours)
        recent_alerts = len([
            a for a in self.alert_history
            if (datetime.now() - a.started_at).total_seconds() < 86400
        ])
        
        # Average resolution time
        resolved_alerts = [a for a in self.alert_history if a.resolved_at]
        avg_resolution_time = 0
        if resolved_alerts:
            total_resolution_time = sum(a.duration.total_seconds() for a in resolved_alerts)
            avg_resolution_time = total_resolution_time / len(resolved_alerts) / 60  # minutes
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "recent_alerts_24h": recent_alerts,
            "severity_counts": severity_counts,
            "average_resolution_time_minutes": avg_resolution_time,
            "alert_rules": len(self.alert_rules),
            "notification_channels": len(self.notification_channels)
        }
    
    async def _evaluate_alert_rules(self):
        """Evaluate alert rules against current metrics."""
        while True:
            try:
                for rule_name, rule in self.alert_rules.items():
                    try:
                        # Get current metric value
                        metric_value = self.metrics_collector.get_aggregated_metrics(
                            rule["metric_name"],
                            window_minutes=rule["window_minutes"]
                        )
                        
                        if metric_value is None:
                            continue
                        
                        # Evaluate condition
                        threshold = rule["threshold"]
                        comparison = rule["comparison"]
                        
                        condition_met = False
                        if comparison == "gt":
                            condition_met = metric_value > threshold
                        elif comparison == "lt":
                            condition_met = metric_value < threshold
                        elif comparison == "ge":
                            condition_met = metric_value >= threshold
                        elif comparison == "le":
                            condition_met = metric_value <= threshold
                        elif comparison == "eq":
                            condition_met = abs(metric_value - threshold) < 0.001
                        elif comparison == "ne":
                            condition_met = abs(metric_value - threshold) >= 0.001
                        
                        # Check if alert already exists for this rule
                        existing_alert = None
                        for alert in self.active_alerts.values():
                            if alert.labels.get("rule_name") == rule_name:
                                existing_alert = alert
                                break
                        
                        if condition_met and not existing_alert:
                            # Fire new alert
                            labels = rule["labels"].copy()
                            labels["rule_name"] = rule_name
                            labels["metric_name"] = rule["metric_name"]
                            
                            annotations = rule["annotations"].copy()
                            annotations["current_value"] = str(metric_value)
                            annotations["threshold"] = str(threshold)
                            
                            await self.fire_alert(
                                name=rule_name,
                                description=rule["description"],
                                severity=rule["severity"],
                                labels=labels,
                                annotations=annotations
                            )
                        
                        elif not condition_met and existing_alert:
                            # Resolve existing alert
                            await self.resolve_alert(existing_alert.alert_id)
                    
                    except Exception as e:
                        self.logger.error(f"Error evaluating alert rule {rule_name}: {e}")
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert rule evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert.
        
        Args:
            alert: Alert to send notifications for
        """
        for channel in self.notification_channels:
            try:
                await asyncio.get_event_loop().run_in_executor(None, channel, alert)
            except Exception as e:
                self.logger.error(f"Notification channel error: {e}")


class DistributedTracer:
    """Distributed tracing for request flows across services."""
    
    def __init__(self, service_name: str):
        """Initialize distributed tracer.
        
        Args:
            service_name: Name of the service using this tracer
        """
        self.service_name = service_name
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Trace storage
        self.active_traces: Dict[str, DistributedTrace] = {}
        self.completed_traces: List[DistributedTrace] = []
        self.max_completed_traces = 1000
        
        # Current span context (for thread-local storage)
        self.current_span: Optional[TraceSpan] = None
    
    def start_trace(self, operation_name: str) -> DistributedTrace:
        """Start a new distributed trace.
        
        Args:
            operation_name: Name of the root operation
            
        Returns:
            New distributed trace
        """
        trace_id = str(uuid.uuid4())
        
        trace = DistributedTrace(
            trace_id=trace_id,
            spans={},
            started_at=datetime.now()
        )
        
        # Create root span
        root_span = self.start_span(operation_name, trace_id=trace_id)
        trace.add_span(root_span)
        
        self.active_traces[trace_id] = trace
        return trace
    
    def start_span(self, 
                   operation_name: str,
                   trace_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None) -> TraceSpan:
        """Start a new span.
        
        Args:
            operation_name: Name of the operation
            trace_id: Trace ID (create new if None)
            parent_span_id: Parent span ID
            
        Returns:
            New trace span
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(),
            parent_span_id=parent_span_id
        )
        
        # Add to active trace
        if trace_id in self.active_traces:
            self.active_traces[trace_id].add_span(span)
        
        # Set as current span
        self.current_span = span
        
        return span
    
    def finish_span(self, span: TraceSpan):
        """Finish a span.
        
        Args:
            span: Span to finish
        """
        span.finish()
        
        # Clear current span if this was it
        if self.current_span and self.current_span.span_id == span.span_id:
            self.current_span = None
        
        # Check if trace is complete
        trace = self.active_traces.get(span.trace_id)
        if trace:
            all_finished = all(
                s.end_time is not None 
                for s in trace.spans.values()
            )
            
            if all_finished:
                # Move to completed traces
                self.completed_traces.append(trace)
                del self.active_traces[span.trace_id]
                
                # Limit completed traces
                if len(self.completed_traces) > self.max_completed_traces:
                    self.completed_traces.pop(0)
    
    def get_trace(self, trace_id: str) -> Optional[DistributedTrace]:
        """Get a trace by ID.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            Trace or None if not found
        """
        # Check active traces first
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        # Check completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics.
        
        Returns:
            Tracing statistics dictionary
        """
        active_traces = len(self.active_traces)
        completed_traces = len(self.completed_traces)
        
        # Calculate average trace duration
        durations = [
            trace.duration_ms for trace in self.completed_traces[-100:]  # Last 100
            if trace.duration_ms is not None
        ]
        
        avg_duration = statistics.mean(durations) if durations else 0
        
        return {
            "active_traces": active_traces,
            "completed_traces": completed_traces,
            "average_trace_duration_ms": avg_duration,
            "service_name": self.service_name
        }


class DistributedMonitor:
    """Main distributed monitoring system coordinating all observability components."""
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 training_coordinator: Optional[DistributedTrainingCoordinator] = None,
                 data_manager: Optional[DataManager] = None):
        """Initialize distributed monitor.
        
        Args:
            service_registry: Service registry for discovering services
            training_coordinator: Training coordinator for training metrics
            data_manager: Data manager for storage metrics
        """
        self.service_registry = service_registry
        self.training_coordinator = training_coordinator
        self.data_manager = data_manager
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Monitoring components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.tracer = DistributedTracer("distributed_monitor")
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.collection_interval = 30  # seconds
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        # Start monitoring tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._collect_training_metrics())
        asyncio.create_task(self._collect_storage_metrics())
        asyncio.create_task(self._collect_service_metrics())
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for common conditions."""
        # High CPU utilization
        self.alert_manager.add_alert_rule(
            rule_name="high_cpu_utilization",
            metric_name="cpu_utilization_percent",
            threshold=90.0,
            comparison="gt",
            severity=AlertSeverity.WARNING,
            window_minutes=5,
            description="CPU utilization is above 90%"
        )
        
        # High memory utilization
        self.alert_manager.add_alert_rule(
            rule_name="high_memory_utilization",
            metric_name="memory_utilization_percent",
            threshold=85.0,
            comparison="gt",
            severity=AlertSeverity.WARNING,
            window_minutes=5,
            description="Memory utilization is above 85%"
        )
        
        # Training failure rate
        self.alert_manager.add_alert_rule(
            rule_name="high_training_failure_rate",
            metric_name="training_failure_rate",
            threshold=0.1,
            comparison="gt",
            severity=AlertSeverity.ERROR,
            window_minutes=10,
            description="Training failure rate is above 10%"
        )
        
        # Storage utilization
        self.alert_manager.add_alert_rule(
            rule_name="high_storage_utilization",
            metric_name="storage_utilization_percent",
            threshold=90.0,
            comparison="gt",
            severity=AlertSeverity.WARNING,
            window_minutes=5,
            description="Storage utilization is above 90%"
        )
        
        # Node health
        self.alert_manager.add_alert_rule(
            rule_name="unhealthy_nodes",
            metric_name="unhealthy_node_count",
            threshold=0,
            comparison="gt",
            severity=AlertSeverity.ERROR,
            window_minutes=2,
            description="One or more nodes are unhealthy"
        )
    
    async def record_metric(self, 
                           name: str,
                           value: Union[float, int],
                           metric_type: MetricType = MetricType.GAUGE,
                           labels: Optional[Dict[str, str]] = None,
                           unit: str = "",
                           help_text: str = ""):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
            unit: Unit of measurement
            help_text: Help description
        """
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
            timestamp=datetime.now(),
            unit=unit,
            help_text=help_text
        )
        
        self.metrics_collector.record_metric(metric)
    
    async def start_trace(self, operation_name: str) -> DistributedTrace:
        """Start a distributed trace.
        
        Args:
            operation_name: Name of the operation being traced
            
        Returns:
            New distributed trace
        """
        return self.tracer.start_trace(operation_name)
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        # System metrics
        cpu_utilization = self.metrics_collector.get_aggregated_metrics("cpu_utilization_percent")
        memory_utilization = self.metrics_collector.get_aggregated_metrics("memory_utilization_percent")
        
        # Training metrics
        training_throughput = self.metrics_collector.get_aggregated_metrics("training_throughput_samples_per_sec")
        hpu_utilization = self.metrics_collector.get_aggregated_metrics("hpu_utilization_percent")
        
        # Storage metrics
        storage_utilization = self.metrics_collector.get_aggregated_metrics("storage_utilization_percent")
        storage_iops = self.metrics_collector.get_aggregated_metrics("storage_iops")
        
        # Service health
        healthy_services = len(self.service_registry.discover_services(status=ServiceStatus.HEALTHY))
        total_services = len(self.service_registry.discover_services())
        
        return {
            "system": {
                "cpu_utilization_percent": cpu_utilization or 0,
                "memory_utilization_percent": memory_utilization or 0,
                "healthy_services": healthy_services,
                "total_services": total_services,
                "service_health_rate": (healthy_services / total_services) if total_services > 0 else 1.0
            },
            "training": {
                "throughput_samples_per_sec": training_throughput or 0,
                "hpu_utilization_percent": hpu_utilization or 0,
                "active_jobs": len(self.training_coordinator.active_jobs) if self.training_coordinator else 0,
                "total_nodes": len(self.training_coordinator.nodes) if self.training_coordinator else 0
            },
            "storage": {
                "utilization_percent": storage_utilization or 0,
                "iops": storage_iops or 0,
                **self.data_manager.get_storage_stats() if self.data_manager else {}
            },
            "alerts": self.alert_manager.get_alert_statistics(),
            "traces": self.tracer.get_trace_statistics(),
            "last_updated": datetime.now()
        }
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        while self.monitoring_enabled:
            try:
                # CPU and memory metrics (would use actual system monitoring)
                await self.record_metric("cpu_utilization_percent", 45.5 + (time.time() % 20), MetricType.GAUGE, unit="%")
                await self.record_metric("memory_utilization_percent", 62.3 + (time.time() % 15), MetricType.GAUGE, unit="%")
                
                # Network metrics
                await self.record_metric("network_bytes_in", time.time() * 1000000, MetricType.COUNTER, unit="bytes")
                await self.record_metric("network_bytes_out", time.time() * 800000, MetricType.COUNTER, unit="bytes")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_training_metrics(self):
        """Collect training-specific metrics."""
        while self.monitoring_enabled:
            try:
                if self.training_coordinator:
                    # Node health metrics
                    healthy_nodes = sum(1 for node in self.training_coordinator.nodes.values() if node.is_healthy)
                    unhealthy_nodes = len(self.training_coordinator.nodes) - healthy_nodes
                    
                    await self.record_metric("healthy_node_count", healthy_nodes, MetricType.GAUGE)
                    await self.record_metric("unhealthy_node_count", unhealthy_nodes, MetricType.GAUGE)
                    
                    # Training progress metrics
                    await self.record_metric("global_training_step", self.training_coordinator.global_step, MetricType.GAUGE)
                    await self.record_metric("total_batches_processed", self.training_coordinator.total_batches_processed, MetricType.COUNTER)
                    
                    # HPU utilization (simulated)
                    avg_hpu_utilization = statistics.mean([
                        node.hpu_utilization for node in self.training_coordinator.nodes.values()
                    ]) if self.training_coordinator.nodes else 0
                    
                    await self.record_metric("hpu_utilization_percent", avg_hpu_utilization, MetricType.GAUGE, unit="%")
                    
                    # Training throughput (simulated)
                    throughput = len(self.training_coordinator.nodes) * 150 if self.training_coordinator.nodes else 0
                    await self.record_metric("training_throughput_samples_per_sec", throughput, MetricType.GAUGE)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Training metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_storage_metrics(self):
        """Collect storage metrics."""
        while self.monitoring_enabled:
            try:
                if self.data_manager:
                    stats = self.data_manager.get_storage_stats()
                    
                    await self.record_metric("storage_utilization_percent", stats.get("utilization_percent", 0), MetricType.GAUGE, unit="%")
                    await self.record_metric("storage_total_objects", stats.get("total_objects", 0), MetricType.GAUGE)
                    await self.record_metric("storage_cache_hit_rate", stats.get("cache_hit_rate", 0), MetricType.GAUGE)
                    
                    # Simulated storage IOPS
                    iops = stats.get("total_reads", 0) + stats.get("total_writes", 0)
                    await self.record_metric("storage_iops", iops, MetricType.COUNTER)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Storage metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_service_metrics(self):
        """Collect service registry metrics."""
        while self.monitoring_enabled:
            try:
                all_services = self.service_registry.discover_services()
                healthy_services = self.service_registry.discover_services(status=ServiceStatus.HEALTHY)
                
                await self.record_metric("total_services", len(all_services), MetricType.GAUGE)
                await self.record_metric("healthy_services", len(healthy_services), MetricType.GAUGE)
                
                # Service type breakdown
                service_type_counts = {}
                for service in all_services:
                    service_type = service.service_type.value
                    service_type_counts[service_type] = service_type_counts.get(service_type, 0) + 1
                
                for service_type, count in service_type_counts.items():
                    await self.record_metric(
                        "services_by_type",
                        count,
                        MetricType.GAUGE,
                        labels={"service_type": service_type}
                    )
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Service metrics collection error: {e}")
                await asyncio.sleep(10)


class ObservabilityStack:
    """Complete observability stack for distributed Gaudi 3 clusters."""
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 training_coordinator: Optional[DistributedTrainingCoordinator] = None,
                 data_manager: Optional[DataManager] = None):
        """Initialize observability stack.
        
        Args:
            service_registry: Service registry
            training_coordinator: Training coordinator
            data_manager: Data manager
        """
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Core components
        self.monitor = DistributedMonitor(service_registry, training_coordinator, data_manager)
        
        # Configuration
        self.export_enabled = True
        self.export_interval = 60  # seconds
        
        # External integrations (placeholders)
        self.prometheus_enabled = False
        self.grafana_enabled = False
        self.jaeger_enabled = False
        
        # Start export tasks
        if self.export_enabled:
            asyncio.create_task(self._export_metrics())
    
    def enable_prometheus_export(self, port: int = 9090):
        """Enable Prometheus metrics export.
        
        Args:
            port: Port to serve Prometheus metrics on
        """
        self.prometheus_enabled = True
        asyncio.create_task(self._start_prometheus_server(port))
    
    def enable_jaeger_tracing(self, jaeger_endpoint: str):
        """Enable Jaeger distributed tracing export.
        
        Args:
            jaeger_endpoint: Jaeger collector endpoint
        """
        self.jaeger_enabled = True
        self.jaeger_endpoint = jaeger_endpoint
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status.
        
        Returns:
            Health status dictionary
        """
        dashboard_data = self.monitor.get_monitoring_dashboard_data()
        
        # Calculate overall health score
        system_health = min(100, 100 - dashboard_data["system"]["cpu_utilization_percent"])
        storage_health = min(100, 100 - dashboard_data["storage"]["utilization_percent"])
        service_health = dashboard_data["system"]["service_health_rate"] * 100
        
        overall_health = (system_health + storage_health + service_health) / 3
        
        status = "healthy"
        if overall_health < 50:
            status = "critical"
        elif overall_health < 70:
            status = "unhealthy"
        elif overall_health < 90:
            status = "degraded"
        
        return {
            "status": status,
            "overall_health_score": overall_health,
            "component_health": {
                "system": system_health,
                "storage": storage_health,
                "services": service_health
            },
            "active_alerts": len(self.monitor.alert_manager.get_active_alerts()),
            "last_updated": datetime.now(),
            **dashboard_data
        }
    
    async def _export_metrics(self):
        """Export metrics to external systems."""
        while self.export_enabled:
            try:
                if self.prometheus_enabled:
                    await self._export_to_prometheus()
                
                if self.jaeger_enabled:
                    await self._export_to_jaeger()
                
                await asyncio.sleep(self.export_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics export error: {e}")
                await asyncio.sleep(30)
    
    async def _start_prometheus_server(self, port: int):
        """Start Prometheus metrics server.
        
        Args:
            port: Port to serve on
        """
        # Placeholder for Prometheus server implementation
        self.logger.info(f"Prometheus server would start on port {port}")
    
    async def _export_to_prometheus(self):
        """Export metrics to Prometheus format."""
        # Placeholder for Prometheus export
        pass
    
    async def _export_to_jaeger(self):
        """Export traces to Jaeger."""
        # Placeholder for Jaeger export
        pass