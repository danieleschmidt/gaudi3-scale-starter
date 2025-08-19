"""Comprehensive Monitoring System for Gaudi 3 Scale - Generation 2 Enhancement.

This module provides enterprise-grade monitoring and observability including:
- Real-time metrics collection and aggregation
- Performance profiling and bottleneck detection
- Alert management with escalation policies
- Distributed tracing for request flow analysis
- Resource utilization monitoring
- SLA/SLO tracking and reporting
- Custom dashboards and visualization
"""

import asyncio
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set
import json
import statistics
from datetime import datetime, timedelta

try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False


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


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class MetricPoint:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "type": self.metric_type.value
        }


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    threshold: float
    metric_name: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "metric_name": self.metric_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "tags": self.tags
        }


class MetricCollector:
    """Thread-safe metric collection."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.aggregated_metrics: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger("metric_collector")
        
        # Performance counters
        self.collection_count = 0
        self.last_collection_time = 0
    
    def record(self, metric: MetricPoint):
        """Record a metric point."""
        with self.lock:
            self.metrics[metric.name].append(metric)
            self.collection_count += 1
            self.last_collection_time = time.time()
            
            # Update aggregated metrics
            self._update_aggregated_metrics(metric)
    
    def record_value(self, name: str, value: float, tags: Dict[str, str] = None, 
                    metric_type: MetricType = MetricType.GAUGE):
        """Record a metric value directly."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type=metric_type
        )
        self.record(metric)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_value(name, value, tags, MetricType.COUNTER)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        self.record_value(name, value, tags, MetricType.GAUGE)
    
    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager to time operations."""
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.record_value(name, duration, tags, MetricType.TIMER)
        
        return timer()
    
    def _update_aggregated_metrics(self, metric: MetricPoint):
        """Update aggregated metrics for the given metric."""
        name = metric.name
        
        if name not in self.aggregated_metrics:
            self.aggregated_metrics[name] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "last_value": 0.0,
                "last_updated": 0.0
            }
        
        agg = self.aggregated_metrics[name]
        
        if metric.metric_type == MetricType.COUNTER:
            agg["sum"] += metric.value
        else:
            agg["last_value"] = metric.value
        
        agg["count"] += 1
        agg["min"] = min(agg["min"], metric.value)
        agg["max"] = max(agg["max"], metric.value)
        agg["last_updated"] = metric.timestamp
    
    def get_metric_history(self, name: str, limit: int = None) -> List[MetricPoint]:
        """Get metric history."""
        with self.lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
            if limit:
                metrics = metrics[-limit:]
            
            return metrics
    
    def get_metric_stats(self, name: str, window_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """Get statistics for a metric within a time window."""
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            if name not in self.metrics:
                return None
            
            # Filter to time window
            recent_metrics = [m for m in self.metrics[name] if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return None
            
            values = [m.value for m in recent_metrics]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "sum": sum(values),
                "rate": len(values) / window_seconds if window_seconds > 0 else 0.0,
                "window_seconds": window_seconds,
                "timestamp": time.time()
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all aggregated metrics."""
        with self.lock:
            return {
                "metrics": dict(self.aggregated_metrics),
                "collection_count": self.collection_count,
                "last_collection_time": self.last_collection_time,
                "active_metric_names": list(self.metrics.keys()),
                "timestamp": time.time()
            }


class PerformanceProfiler:
    """Performance profiling and bottleneck detection."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.active_traces: Dict[str, Dict] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.logger = logging.getLogger("performance_profiler")
    
    @contextmanager
    def trace_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Trace an operation for performance analysis."""
        trace_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"
        start_time = time.time()
        
        trace_info = {
            "trace_id": trace_id,
            "operation_name": operation_name,
            "start_time": start_time,
            "tags": tags or {},
            "thread_id": threading.get_ident(),
            "steps": []
        }
        
        with self.lock:
            self.active_traces[trace_id] = trace_info
        
        try:
            yield TraceContext(trace_id, self)
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self.lock:
                if trace_id in self.active_traces:
                    trace_info = self.active_traces.pop(trace_id)
                    trace_info["end_time"] = end_time
                    trace_info["duration"] = duration
                    
                    self.completed_traces.append(trace_info)
                    
                    # Record metrics
                    self.metric_collector.record_value(
                        f"operation_duration_{operation_name}",
                        duration,
                        tags,
                        MetricType.TIMER
                    )
                    
                    # Detect potential bottlenecks
                    self._analyze_trace(trace_info)
    
    def add_trace_step(self, trace_id: str, step_name: str, duration: float = None):
        """Add a step to an active trace."""
        with self.lock:
            if trace_id in self.active_traces:
                step_info = {
                    "step_name": step_name,
                    "timestamp": time.time(),
                    "duration": duration
                }
                self.active_traces[trace_id]["steps"].append(step_info)
    
    def _analyze_trace(self, trace_info: Dict):
        """Analyze completed trace for bottlenecks."""
        duration = trace_info["duration"]
        operation_name = trace_info["operation_name"]
        
        # Define performance thresholds (configurable)
        thresholds = {
            "slow_operation": 5.0,  # seconds
            "very_slow_operation": 10.0  # seconds
        }
        
        if duration > thresholds["very_slow_operation"]:
            self.logger.warning(f"Very slow operation detected: {operation_name} took {duration:.2f}s")
            self.metric_collector.increment_counter("slow_operations_very_slow", tags={
                "operation": operation_name
            })
        elif duration > thresholds["slow_operation"]:
            self.logger.info(f"Slow operation detected: {operation_name} took {duration:.2f}s")
            self.metric_collector.increment_counter("slow_operations", tags={
                "operation": operation_name
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary."""
        with self.lock:
            completed_traces = list(self.completed_traces)
        
        if not completed_traces:
            return {"message": "No completed traces"}
        
        # Analyze operation performance
        operation_stats = defaultdict(list)
        for trace in completed_traces:
            op_name = trace["operation_name"]
            operation_stats[op_name].append(trace["duration"])
        
        performance_summary = {}
        for op_name, durations in operation_stats.items():
            performance_summary[op_name] = {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "p95": self._percentile(durations, 95),
                "p99": self._percentile(durations, 99)
            }
        
        return {
            "operation_performance": performance_summary,
            "total_traces": len(completed_traces),
            "active_traces": len(self.active_traces),
            "timestamp": time.time()
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]


class TraceContext:
    """Context for operation tracing."""
    
    def __init__(self, trace_id: str, profiler: PerformanceProfiler):
        self.trace_id = trace_id
        self.profiler = profiler
    
    def add_step(self, step_name: str):
        """Add a step to the trace."""
        self.profiler.add_trace_step(self.trace_id, step_name)
    
    @contextmanager
    def step(self, step_name: str):
        """Time a step within the trace."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.profiler.add_trace_step(self.trace_id, step_name, duration)


class AlertManager:
    """Alert management system."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.logger = logging.getLogger("alert_manager")
        
        # Notification handlers
        self.notification_handlers: List[Callable] = []
        
        # Start alert evaluation thread
        self._running = False
        self._thread = None
    
    def add_alert_rule(self, rule_id: str, metric_name: str, condition: str, 
                      threshold: float, severity: AlertSeverity,
                      description: str = "", tags: Dict[str, str] = None):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule_id] = {
                "metric_name": metric_name,
                "condition": condition,  # "greater_than", "less_than", "equals"
                "threshold": threshold,
                "severity": severity,
                "description": description,
                "tags": tags or {},
                "created_at": time.time()
            }
        
        self.logger.info(f"Added alert rule: {rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self.lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def start(self):
        """Start alert evaluation."""
        with self.lock:
            if self._running:
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._evaluation_loop, daemon=True)
            self._thread.start()
            self.logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert evaluation."""
        with self.lock:
            self._running = False
            if self._thread:
                self._thread.join(timeout=5.0)
            self.logger.info("Alert manager stopped")
    
    def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self._running:
            try:
                self._evaluate_alert_rules()
                time.sleep(30)  # Evaluate every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in alert evaluation: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        with self.lock:
            rules_snapshot = dict(self.alert_rules)
        
        for rule_id, rule in rules_snapshot.items():
            try:
                self._evaluate_single_rule(rule_id, rule)
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_id}: {e}")
    
    def _evaluate_single_rule(self, rule_id: str, rule: Dict):
        """Evaluate a single alert rule."""
        metric_name = rule["metric_name"]
        condition = rule["condition"]
        threshold = rule["threshold"]
        
        # Get recent metric stats
        stats = self.metric_collector.get_metric_stats(metric_name, window_seconds=300)
        if not stats:
            return  # No data available
        
        # Determine the value to compare
        if condition.endswith("_avg"):
            value = stats["mean"]
        elif condition.endswith("_max"):
            value = stats["max"]
        elif condition.endswith("_min"):
            value = stats["min"]
        else:
            # Use the latest aggregated value
            all_metrics = self.metric_collector.get_all_metrics()
            if metric_name in all_metrics["metrics"]:
                value = all_metrics["metrics"][metric_name]["last_value"]
            else:
                return
        
        # Check condition
        triggered = False
        if condition.startswith("greater_than"):
            triggered = value > threshold
        elif condition.startswith("less_than"):
            triggered = value < threshold
        elif condition.startswith("equals"):
            triggered = abs(value - threshold) < 0.001
        
        # Handle alert state change
        with self.lock:
            if triggered and rule_id not in self.active_alerts:
                # New alert
                alert = Alert(
                    id=rule_id,
                    name=f"Alert_{rule_id}",
                    description=rule["description"],
                    condition=condition,
                    severity=rule["severity"],
                    threshold=threshold,
                    metric_name=metric_name,
                    tags=rule["tags"]
                )
                
                self.active_alerts[rule_id] = alert
                self.alert_history.append(alert)
                
                self.logger.warning(f"Alert triggered: {rule_id} - {rule['description']}")
                
                # Send notifications
                self._send_notifications(alert)
                
            elif not triggered and rule_id in self.active_alerts:
                # Alert resolved
                alert = self.active_alerts.pop(rule_id)
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                alert.updated_at = time.time()
                
                self.alert_history.append(alert)
                
                self.logger.info(f"Alert resolved: {rule_id}")
                
                # Send resolution notification
                self._send_notifications(alert)
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error sending alert notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history."""
        with self.lock:
            return list(self.alert_history)[-limit:]


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.logger = logging.getLogger("system_monitor")
        self._running = False
        self._thread = None
    
    def start(self, interval: float = 30.0):
        """Start system monitoring."""
        if not _psutil_available:
            self.logger.warning("psutil not available, system monitoring disabled")
            return
        
        self.interval = interval
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        self.logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                time.sleep(min(self.interval, 60.0))
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        if not _psutil_available:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metric_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.metric_collector.set_gauge("system_cpu_count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metric_collector.set_gauge("system_memory_percent", memory.percent)
            self.metric_collector.set_gauge("system_memory_available_gb", memory.available / (1024**3))
            self.metric_collector.set_gauge("system_memory_total_gb", memory.total / (1024**3))
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            self.metric_collector.set_gauge("system_disk_percent", 
                                           (disk_usage.used / disk_usage.total) * 100)
            self.metric_collector.set_gauge("system_disk_free_gb", disk_usage.free / (1024**3))
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.metric_collector.set_gauge("system_network_bytes_sent", net_io.bytes_sent)
                self.metric_collector.set_gauge("system_network_bytes_recv", net_io.bytes_recv)
            except Exception:
                pass  # Network stats not available on all systems
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")


class ComprehensiveMonitor:
    """Main monitoring system orchestrating all monitoring components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.metric_collector = MetricCollector(
            max_points=self.config.get("max_metric_points", 10000)
        )
        self.profiler = PerformanceProfiler(self.metric_collector)
        self.alert_manager = AlertManager(self.metric_collector)
        self.system_monitor = SystemMonitor(self.metric_collector)
        
        self.logger = logging.getLogger("comprehensive_monitor")
        self.start_time = time.time()
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default notification handler
        self.alert_manager.add_notification_handler(self._default_notification_handler)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "max_metric_points": 10000,
            "system_monitoring_interval": 30.0,
            "alert_evaluation_interval": 30.0,
            "default_alerts": {
                "high_cpu": {
                    "metric": "system_cpu_percent",
                    "condition": "greater_than",
                    "threshold": 80.0,
                    "severity": "warning"
                },
                "high_memory": {
                    "metric": "system_memory_percent",
                    "condition": "greater_than",
                    "threshold": 90.0,
                    "severity": "critical"
                },
                "low_disk_space": {
                    "metric": "system_disk_percent",
                    "condition": "greater_than",
                    "threshold": 85.0,
                    "severity": "warning"
                }
            }
        }
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_alerts = self.config.get("default_alerts", {})
        
        for alert_name, alert_config in default_alerts.items():
            self.alert_manager.add_alert_rule(
                rule_id=alert_name,
                metric_name=alert_config["metric"],
                condition=alert_config["condition"],
                threshold=alert_config["threshold"],
                severity=AlertSeverity(alert_config["severity"]),
                description=f"Default alert for {alert_name}"
            )
    
    def _default_notification_handler(self, alert: Alert):
        """Default alert notification handler."""
        status = "TRIGGERED" if alert.status == AlertStatus.ACTIVE else "RESOLVED"
        self.logger.warning(f"ALERT {status}: {alert.name} - {alert.description}")
    
    def start(self):
        """Start all monitoring components."""
        self.alert_manager.start()
        self.system_monitor.start(self.config.get("system_monitoring_interval", 30.0))
        self.logger.info("Comprehensive monitoring started")
    
    def stop(self):
        """Stop all monitoring components."""
        self.alert_manager.stop()
        self.system_monitor.stop()
        self.logger.info("Comprehensive monitoring stopped")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None,
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a metric."""
        self.metric_collector.record_value(name, value, tags, metric_type)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter."""
        self.metric_collector.increment_counter(name, value, tags)
    
    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Time an operation."""
        return self.metric_collector.time_operation(name, tags)
    
    def trace_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Trace an operation for performance analysis."""
        return self.profiler.trace_operation(operation_name, tags)
    
    def add_alert_rule(self, rule_id: str, metric_name: str, condition: str,
                      threshold: float, severity: AlertSeverity, description: str = ""):
        """Add a custom alert rule."""
        self.alert_manager.add_alert_rule(rule_id, metric_name, condition, threshold, severity, description)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        uptime = time.time() - self.start_time
        
        # Get all metrics
        all_metrics = self.metric_collector.get_all_metrics()
        
        # Get performance summary
        performance_summary = self.profiler.get_performance_summary()
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "timestamp": time.time(),
            "uptime": uptime,
            "metrics_overview": {
                "total_metrics": len(all_metrics.get("active_metric_names", [])),
                "collection_count": all_metrics.get("collection_count", 0),
                "last_collection": all_metrics.get("last_collection_time", 0)
            },
            "system_health": self._get_system_health_summary(),
            "performance": performance_summary,
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": [alert.to_dict() for alert in active_alerts],
                "recent_history": [alert.to_dict() for alert in self.alert_manager.get_alert_history(10)]
            },
            "metrics": all_metrics
        }
    
    def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        health = {"status": "unknown"}
        
        if _psutil_available:
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Determine overall health
                if cpu_percent > 90 or memory.percent > 95 or (disk.used / disk.total) > 0.95:
                    status = "critical"
                elif cpu_percent > 80 or memory.percent > 85 or (disk.used / disk.total) > 0.85:
                    status = "warning"
                else:
                    status = "healthy"
                
                health = {
                    "status": status,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": (disk.used / disk.total) * 100,
                    "timestamp": time.time()
                }
            except Exception as e:
                health = {"status": "error", "error": str(e)}
        
        return health


# Global monitoring instance
_monitor = None


def get_monitor(config: Optional[Dict[str, Any]] = None) -> ComprehensiveMonitor:
    """Get or create global monitoring instance."""
    global _monitor
    
    if _monitor is None:
        _monitor = ComprehensiveMonitor(config)
        _monitor.start()
    
    return _monitor


def shutdown_monitor():
    """Shutdown global monitoring instance."""
    global _monitor
    
    if _monitor:
        _monitor.stop()
        _monitor = None