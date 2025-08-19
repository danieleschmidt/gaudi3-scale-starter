"""Enhanced monitoring system for Generation 2 robustness.

This module provides comprehensive monitoring capabilities including:
- Real-time metrics collection and aggregation
- Performance monitoring with alerting
- Health checks and system diagnostics
- Resource usage tracking
- Training progress monitoring with anomaly detection
- Custom metrics and alerts
"""

import time
import threading
import queue
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import statistics

from .logging_utils import get_logger
from .exceptions import create_validation_error

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """A metric data point."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """An alert notification."""
    id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class MetricCollector:
    """Thread-safe metric collection system."""
    
    def __init__(self, max_metrics: int = 10000):
        """Initialize metric collector.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics
        self.metrics: Dict[str, List[Metric]] = {}
        self.lock = threading.Lock()
        
    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metric_type=metric_type,
            labels=labels or {},
            unit=unit
        )
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
                
            self.metrics[name].append(metric)
            
            # Trim old metrics to maintain memory limit
            if len(self.metrics[name]) > self.max_metrics:
                self.metrics[name] = self.metrics[name][-self.max_metrics:]
                
    def get_latest(self, name: str) -> Optional[Metric]:
        """Get the latest metric value."""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
        return None
        
    def get_history(self, name: str, limit: int = 100) -> List[Metric]:
        """Get metric history."""
        with self.lock:
            if name in self.metrics:
                return self.metrics[name][-limit:]
        return []
        
    def get_stats(self, name: str, window_minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get statistical summary of a metric over a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            if name not in self.metrics:
                return None
                
            recent_values = [
                m.value for m in self.metrics[name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_values:
                return None
                
            return {
                'count': len(recent_values),
                'min': min(recent_values),
                'max': max(recent_values),
                'mean': statistics.mean(recent_values),
                'median': statistics.median(recent_values),
                'std': statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
                'latest': recent_values[-1]
            }
            
    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all metrics (thread-safe copy)."""
        with self.lock:
            return {name: metrics.copy() for name, metrics in self.metrics.items()}


class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.lock = threading.Lock()
        
    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        condition: str = "greater_than",
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: Optional[str] = None
    ):
        """Add an alert rule.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold: Threshold value for the alert
            condition: Condition type ('greater_than', 'less_than', 'equals')
            severity: Alert severity level
            message_template: Custom message template
        """
        rule = {
            'threshold': threshold,
            'condition': condition,
            'severity': severity,
            'message_template': message_template or f"{metric_name} {condition} {threshold}"
        }
        
        with self.lock:
            self.alert_rules[metric_name] = rule
            
        logger.info(f"Added alert rule for {metric_name}: {condition} {threshold}")
        
    def check_alerts(self, metric: Metric):
        """Check if a metric triggers any alerts."""
        with self.lock:
            if metric.name not in self.alert_rules:
                return
                
            rule = self.alert_rules[metric.name]
            threshold = rule['threshold']
            condition = rule['condition']
            
            should_alert = False
            if condition == "greater_than" and metric.value > threshold:
                should_alert = True
            elif condition == "less_than" and metric.value < threshold:
                should_alert = True
            elif condition == "equals" and metric.value == threshold:
                should_alert = True
                
            if should_alert:
                alert_id = f"{metric.name}_{metric.timestamp.timestamp()}"
                alert = Alert(
                    id=alert_id,
                    severity=rule['severity'],
                    message=rule['message_template'].format(
                        metric_name=metric.name,
                        value=metric.value,
                        threshold=threshold
                    ),
                    metric_name=metric.name,
                    threshold=threshold,
                    actual_value=metric.value,
                    timestamp=metric.timestamp
                )
                
                self.alerts.append(alert)
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
                        
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
            
    def resolve_alert(self, alert_id: str):
        """Resolve an alert by ID."""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_timestamp = datetime.now()
                    break


class PerformanceMonitor:
    """Performance monitoring with timing and resource tracking."""
    
    def __init__(self, collector: MetricCollector):
        """Initialize performance monitor.
        
        Args:
            collector: Metric collector instance
        """
        self.collector = collector
        self.active_timers: Dict[str, float] = {}
        
    def start_timer(self, name: str):
        """Start a performance timer."""
        self.active_timers[name] = time.time()
        
    def end_timer(self, name: str) -> float:
        """End a performance timer and record the duration."""
        if name not in self.active_timers:
            logger.warning(f"Timer {name} was not started")
            return 0.0
            
        duration = time.time() - self.active_timers[name]
        del self.active_timers[name]
        
        self.collector.record(
            f"timer.{name}",
            duration,
            MetricType.TIMER,
            unit="seconds"
        )
        
        return duration
        
    def record_memory_usage(self):
        """Record current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.collector.record(
                "memory.rss",
                memory_info.rss / (1024 * 1024),  # MB
                MetricType.GAUGE,
                unit="MB"
            )
            
            self.collector.record(
                "memory.vms", 
                memory_info.vms / (1024 * 1024),  # MB
                MetricType.GAUGE,
                unit="MB"
            )
            
            # System memory
            sys_memory = psutil.virtual_memory()
            self.collector.record(
                "system.memory.percent",
                sys_memory.percent,
                MetricType.GAUGE,
                unit="percent"
            )
            
        except ImportError:
            logger.debug("psutil not available, skipping memory monitoring")
        except Exception as e:
            logger.error(f"Failed to record memory usage: {e}")
            
    def record_cpu_usage(self):
        """Record current CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            self.collector.record(
                "cpu.percent",
                cpu_percent,
                MetricType.GAUGE,
                unit="percent"
            )
            
        except ImportError:
            logger.debug("psutil not available, skipping CPU monitoring")
        except Exception as e:
            logger.error(f"Failed to record CPU usage: {e}")
            
    def record_training_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        batch_time: float
    ):
        """Record training-specific metrics."""
        metrics = [
            ("training.epoch", epoch, MetricType.COUNTER),
            ("training.loss", loss, MetricType.GAUGE),
            ("training.accuracy", accuracy, MetricType.GAUGE, "percent"),
            ("training.learning_rate", learning_rate, MetricType.GAUGE),
            ("training.batch_time", batch_time, MetricType.TIMER, "seconds")
        ]
        
        for name, value, metric_type, unit in [(m[0], m[1], m[2], m[3] if len(m) > 3 else None) for m in metrics]:
            self.collector.record(name, value, metric_type, unit=unit)


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, collector: MetricCollector):
        """Initialize health checker.
        
        Args:
            collector: Metric collector instance
        """
        self.collector = collector
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        
    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy, False otherwise
        """
        self.health_checks[name] = check_func
        
    def run_health_checks(self) -> Dict[str, bool]:
        """Run all health checks and record results."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                results[name] = is_healthy
                
                self.collector.record(
                    f"health.{name}",
                    1.0 if is_healthy else 0.0,
                    MetricType.GAUGE
                )
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = False
                
                self.collector.record(
                    f"health.{name}",
                    0.0,
                    MetricType.GAUGE
                )
                
        return results


class EnhancedMonitor:
    """Comprehensive monitoring system for robust training."""
    
    def __init__(
        self,
        enable_performance_monitoring: bool = True,
        enable_health_checks: bool = True,
        enable_alerts: bool = True,
        metrics_retention_limit: int = 10000
    ):
        """Initialize enhanced monitor.
        
        Args:
            enable_performance_monitoring: Enable performance metrics collection
            enable_health_checks: Enable health monitoring
            enable_alerts: Enable alert system
            metrics_retention_limit: Maximum metrics to keep in memory
        """
        self.collector = MetricCollector(metrics_retention_limit)
        self.alert_manager = AlertManager() if enable_alerts else None
        self.performance_monitor = PerformanceMonitor(self.collector) if enable_performance_monitoring else None
        self.health_checker = HealthChecker(self.collector) if enable_health_checks else None
        
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        
        # Setup default health checks
        if self.health_checker:
            self._setup_default_health_checks()
            
        # Setup default alerts
        if self.alert_manager:
            self._setup_default_alerts()
            
        logger.info("EnhancedMonitor initialized")
        
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        def memory_health_check():
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage
            except:
                return True  # Assume healthy if can't check
                
        def disk_health_check():
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return disk.percent < 90  # Less than 90% disk usage
            except:
                return True  # Assume healthy if can't check
                
        self.health_checker.add_health_check("memory", memory_health_check)
        self.health_checker.add_health_check("disk", disk_health_check)
        
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # Memory usage alert
        self.alert_manager.add_alert_rule(
            "system.memory.percent",
            85.0,
            "greater_than",
            AlertSeverity.WARNING,
            "High memory usage: {value}% > {threshold}%"
        )
        
        # Training loss divergence alert
        self.alert_manager.add_alert_rule(
            "training.loss",
            10.0,
            "greater_than", 
            AlertSeverity.ERROR,
            "Training loss is diverging: {value} > {threshold}"
        )
        
        # Batch time performance alert
        self.alert_manager.add_alert_rule(
            "training.batch_time",
            30.0,
            "greater_than",
            AlertSeverity.WARNING,
            "Slow batch processing: {value}s > {threshold}s"
        )
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Background monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        logger.info("Background monitoring stopped")
        
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Record system metrics
                if self.performance_monitor:
                    self.performance_monitor.record_memory_usage()
                    self.performance_monitor.record_cpu_usage()
                    
                # Run health checks
                if self.health_checker:
                    self.health_checker.run_health_checks()
                    
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)  # Brief pause on error
                
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a custom metric."""
        self.collector.record(name, value, metric_type, labels, unit)
        
        # Check alerts if enabled
        if self.alert_manager:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metric_type=metric_type,
                labels=labels or {},
                unit=unit
            )
            self.alert_manager.check_alerts(metric)
            
    def get_metric_stats(self, name: str, window_minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get statistical summary of a metric."""
        return self.collector.get_stats(name, window_minutes)
        
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest value of a metric."""
        return self.collector.get_latest(name)
        
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        if self.alert_manager:
            return self.alert_manager.get_active_alerts()
        return []
        
    def add_alert_rule(self, metric_name: str, threshold: float, **kwargs):
        """Add a custom alert rule."""
        if self.alert_manager:
            self.alert_manager.add_alert_rule(metric_name, threshold, **kwargs)
            
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add an alert notification callback."""
        if self.alert_manager:
            self.alert_manager.add_alert_callback(callback)
            
    def export_metrics(self, filepath: Union[str, Path]) -> bool:
        """Export all metrics to a JSON file."""
        try:
            metrics_data = {}
            all_metrics = self.collector.get_all_metrics()
            
            for name, metrics_list in all_metrics.items():
                metrics_data[name] = [
                    {
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'type': m.metric_type.value,
                        'labels': m.labels,
                        'unit': m.unit
                    }
                    for m in metrics_list
                ]
                
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
            
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of the monitoring system status."""
        summary = {
            'monitoring_active': self.monitoring_active,
            'total_metrics': len(self.collector.metrics),
            'active_alerts': len(self.get_active_alerts()),
            'components': {
                'performance_monitor': self.performance_monitor is not None,
                'health_checker': self.health_checker is not None,
                'alert_manager': self.alert_manager is not None
            }
        }
        
        # Add recent health check results
        if self.health_checker:
            latest_health = {}
            for check_name in self.health_checker.health_checks:
                latest_metric = self.collector.get_latest(f"health.{check_name}")
                if latest_metric:
                    latest_health[check_name] = latest_metric.value == 1.0
            summary['health_status'] = latest_health
            
        return summary


# Context manager for timing operations
class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: EnhancedMonitor, metric_name: str):
        self.monitor = monitor
        self.metric_name = metric_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_metric(
                self.metric_name,
                duration,
                MetricType.TIMER,
                unit="seconds"
            )


# Helper function to create a basic monitoring setup
def create_basic_monitor() -> EnhancedMonitor:
    """Create a basic monitoring setup with sensible defaults."""
    monitor = EnhancedMonitor(
        enable_performance_monitoring=True,
        enable_health_checks=True,
        enable_alerts=True
    )
    
    # Add a simple alert callback that logs alerts
    def log_alert(alert: Alert):
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
    monitor.add_alert_callback(log_alert)
    
    return monitor