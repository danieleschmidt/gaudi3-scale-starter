"""Metrics collection and reporting for Gaudi 3 training."""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
    import psutil
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    model_name: str
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0  # tokens/samples per second
    hpu_utilization: float = 0.0
    memory_usage: float = 0.0
    power_consumption: float = 0.0
    temperature: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    load_average: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Base metrics collector for training and system metrics.
    
    Collects and stores various metrics related to training performance,
    hardware utilization, and system health.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.training_metrics: List[TrainingMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self._start_time = time.time()
    
    def collect_training_metrics(self, model_name: str, **kwargs) -> TrainingMetrics:
        """Collect training metrics.
        
        Args:
            model_name: Name of the model being trained
            **kwargs: Additional metric values
            
        Returns:
            Training metrics object
        """
        metrics = TrainingMetrics(
            model_name=model_name,
            epoch=kwargs.get('epoch', 0),
            step=kwargs.get('step', 0),
            loss=kwargs.get('loss', 0.0),
            accuracy=kwargs.get('accuracy', 0.0),
            learning_rate=kwargs.get('learning_rate', 0.0),
            throughput=kwargs.get('throughput', 0.0),
            hpu_utilization=self._get_hpu_utilization(),
            memory_usage=self._get_hpu_memory_usage(),
            power_consumption=self._get_power_consumption(),
            temperature=self._get_temperature()
        )
        
        self.training_metrics.append(metrics)
        return metrics
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics.
        
        Returns:
            System metrics object
        """
        if not psutil:
            logger.warning("psutil not available, returning empty system metrics")
            return SystemMetrics()
        
        # Network statistics
        net_io = psutil.net_io_counters()
        
        metrics = SystemMetrics(
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_bytes_sent=net_io.bytes_sent,
            network_bytes_recv=net_io.bytes_recv,
            load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
        )
        
        self.system_metrics.append(metrics)
        return metrics
    
    def _get_hpu_utilization(self) -> float:
        """Get HPU utilization percentage.
        
        Returns:
            HPU utilization percentage
        """
        try:
            import habana_frameworks.torch as htorch
            # This is a placeholder - actual implementation would depend on Habana's monitoring API
            # For now, return a mock value
            return 85.0
        except ImportError:
            return 0.0
    
    def _get_hpu_memory_usage(self) -> float:
        """Get HPU memory usage in GB.
        
        Returns:
            HPU memory usage in GB
        """
        try:
            import habana_frameworks.torch as htorch
            # This is a placeholder - actual implementation would use htorch.hpu.memory_allocated()
            return 24.5
        except ImportError:
            return 0.0
    
    def _get_power_consumption(self) -> float:
        """Get power consumption in watts.
        
        Returns:
            Power consumption in watts
        """
        # Placeholder for actual power monitoring
        return 350.0
    
    def _get_temperature(self) -> float:
        """Get device temperature in Celsius.
        
        Returns:
            Temperature in Celsius
        """
        # Placeholder for actual temperature monitoring
        return 65.0
    
    def get_training_summary(self, model_name: str) -> Dict[str, Any]:
        """Get training summary for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Training summary statistics
        """
        model_metrics = [m for m in self.training_metrics if m.model_name == model_name]
        
        if not model_metrics:
            return {}
        
        latest = model_metrics[-1]
        
        return {
            "model_name": model_name,
            "total_epochs": latest.epoch,
            "total_steps": latest.step,
            "latest_loss": latest.loss,
            "latest_accuracy": latest.accuracy,
            "average_throughput": sum(m.throughput for m in model_metrics) / len(model_metrics),
            "average_hpu_utilization": sum(m.hpu_utilization for m in model_metrics) / len(model_metrics),
            "training_duration": (latest.timestamp - model_metrics[0].timestamp).total_seconds(),
            "start_time": model_metrics[0].timestamp.isoformat(),
            "last_update": latest.timestamp.isoformat()
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary statistics.
        
        Returns:
            System summary statistics
        """
        if not self.system_metrics:
            return {}
        
        latest = self.system_metrics[-1]
        recent_metrics = [m for m in self.system_metrics 
                         if m.timestamp > datetime.now() - timedelta(minutes=10)]
        
        if not recent_metrics:
            recent_metrics = [latest]
        
        return {
            "current_cpu_usage": latest.cpu_usage,
            "current_memory_usage": latest.memory_usage,
            "current_disk_usage": latest.disk_usage,
            "average_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "average_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "network_bytes_sent": latest.network_bytes_sent,
            "network_bytes_recv": latest.network_bytes_recv,
            "load_average": latest.load_average,
            "uptime": time.time() - self._start_time,
            "last_update": latest.timestamp.isoformat()
        }


class PrometheusMetrics:
    """Prometheus metrics exporter for Gaudi 3 training.
    
    Exports training and system metrics in Prometheus format
    for integration with monitoring dashboards.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics.
        
        Args:
            registry: Prometheus registry to use
            
        Raises:
            ImportError: If prometheus_client is not available
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not available")
        
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # Training metrics
        self.training_loss = Gauge(
            'gaudi_training_loss',
            'Current training loss',
            ['model_name'],
            registry=self.registry
        )
        
        self.training_accuracy = Gauge(
            'gaudi_training_accuracy',
            'Current training accuracy',
            ['model_name'],
            registry=self.registry
        )
        
        self.training_throughput = Gauge(
            'gaudi_training_throughput',
            'Training throughput (samples/tokens per second)',
            ['model_name'],
            registry=self.registry
        )
        
        self.training_epochs = Counter(
            'gaudi_training_epochs_total',
            'Total number of training epochs',
            ['model_name'],
            registry=self.registry
        )
        
        self.training_steps = Counter(
            'gaudi_training_steps_total',
            'Total number of training steps',
            ['model_name'],
            registry=self.registry
        )
        
        # HPU metrics
        self.hpu_utilization = Gauge(
            'gaudi_hpu_utilization_percent',
            'HPU utilization percentage',
            ['device_id'],
            registry=self.registry
        )
        
        self.hpu_memory_usage = Gauge(
            'gaudi_hpu_memory_usage_bytes',
            'HPU memory usage in bytes',
            ['device_id'],
            registry=self.registry
        )
        
        self.hpu_power_consumption = Gauge(
            'gaudi_hpu_power_watts',
            'HPU power consumption in watts',
            ['device_id'],
            registry=self.registry
        )
        
        self.hpu_temperature = Gauge(
            'gaudi_hpu_temperature_celsius',
            'HPU temperature in Celsius',
            ['device_id'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'gaudi_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'gaudi_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'gaudi_system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Training latency histogram
        self.training_step_duration = Histogram(
            'gaudi_training_step_duration_seconds',
            'Training step duration in seconds',
            ['model_name'],
            registry=self.registry
        )
        
        # Cost metrics
        self.training_cost = Counter(
            'gaudi_training_cost_usd',
            'Training cost in USD',
            ['model_name', 'instance_type'],
            registry=self.registry
        )
    
    def update_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Update training metrics.
        
        Args:
            metrics: Training metrics to update
        """
        model_name = metrics.model_name
        
        self.training_loss.labels(model_name=model_name).set(metrics.loss)
        self.training_accuracy.labels(model_name=model_name).set(metrics.accuracy)
        self.training_throughput.labels(model_name=model_name).set(metrics.throughput)
        
        # Update HPU metrics (assuming single device for now)
        device_id = "0"
        self.hpu_utilization.labels(device_id=device_id).set(metrics.hpu_utilization)
        self.hpu_memory_usage.labels(device_id=device_id).set(metrics.memory_usage * 1024**3)  # Convert GB to bytes
        self.hpu_power_consumption.labels(device_id=device_id).set(metrics.power_consumption)
        self.hpu_temperature.labels(device_id=device_id).set(metrics.temperature)
    
    def update_system_metrics(self, metrics: SystemMetrics) -> None:
        """Update system metrics.
        
        Args:
            metrics: System metrics to update
        """
        self.system_cpu_usage.set(metrics.cpu_usage)
        self.system_memory_usage.set(metrics.memory_usage)
        self.system_disk_usage.set(metrics.disk_usage)
    
    def increment_training_step(self, model_name: str, duration: float) -> None:
        """Increment training step counter and record duration.
        
        Args:
            model_name: Name of the model
            duration: Step duration in seconds
        """
        self.training_steps.labels(model_name=model_name).inc()
        self.training_step_duration.labels(model_name=model_name).observe(duration)
    
    def increment_training_epoch(self, model_name: str) -> None:
        """Increment training epoch counter.
        
        Args:
            model_name: Name of the model
        """
        self.training_epochs.labels(model_name=model_name).inc()
    
    def add_training_cost(self, model_name: str, instance_type: str, cost: float) -> None:
        """Add training cost.
        
        Args:
            model_name: Name of the model
            instance_type: Type of instance used
            cost: Cost in USD
        """
        self.training_cost.labels(model_name=model_name, instance_type=instance_type).inc(cost)
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry).decode('utf-8')


class MetricsAggregator:
    """Aggregates metrics over time windows for analysis."""
    
    def __init__(self, collector: MetricsCollector):
        """Initialize metrics aggregator.
        
        Args:
            collector: Metrics collector instance
        """
        self.collector = collector
    
    def get_training_trends(self, model_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get training trends over time window.
        
        Args:
            model_name: Name of the model
            window_minutes: Time window in minutes
            
        Returns:
            Training trends data
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.collector.training_metrics
            if m.model_name == model_name and m.timestamp > cutoff
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate trends
        losses = [m.loss for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics]
        
        return {
            "model_name": model_name,
            "window_minutes": window_minutes,
            "data_points": len(recent_metrics),
            "loss_trend": {
                "current": losses[-1] if losses else 0,
                "average": sum(losses) / len(losses) if losses else 0,
                "min": min(losses) if losses else 0,
                "max": max(losses) if losses else 0,
                "change": (losses[-1] - losses[0]) if len(losses) > 1 else 0
            },
            "accuracy_trend": {
                "current": accuracies[-1] if accuracies else 0,
                "average": sum(accuracies) / len(accuracies) if accuracies else 0,
                "min": min(accuracies) if accuracies else 0,
                "max": max(accuracies) if accuracies else 0,
                "change": (accuracies[-1] - accuracies[0]) if len(accuracies) > 1 else 0
            },
            "throughput_trend": {
                "current": throughputs[-1] if throughputs else 0,
                "average": sum(throughputs) / len(throughputs) if throughputs else 0,
                "min": min(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0
            }
        }
    
    def get_efficiency_metrics(self, model_name: str) -> Dict[str, float]:
        """Calculate training efficiency metrics.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Efficiency metrics
        """
        model_metrics = [m for m in self.collector.training_metrics if m.model_name == model_name]
        
        if not model_metrics:
            return {}
        
        # Calculate efficiency metrics
        avg_hpu_utilization = sum(m.hpu_utilization for m in model_metrics) / len(model_metrics)
        avg_throughput = sum(m.throughput for m in model_metrics) / len(model_metrics)
        avg_power = sum(m.power_consumption for m in model_metrics) / len(model_metrics)
        
        # Efficiency scores (0-100)
        hpu_efficiency = min(avg_hpu_utilization, 100.0)
        power_efficiency = max(0, 100 - (avg_power - 300) / 5)  # Penalty above 300W
        thermal_efficiency = max(0, 100 - max(0, max(m.temperature for m in model_metrics) - 70))  # Penalty above 70°C
        
        return {
            "hpu_efficiency": hpu_efficiency,
            "power_efficiency": power_efficiency,
            "thermal_efficiency": thermal_efficiency,
            "overall_efficiency": (hpu_efficiency + power_efficiency + thermal_efficiency) / 3,
            "avg_hpu_utilization": avg_hpu_utilization,
            "avg_throughput": avg_throughput,
            "avg_power_consumption": avg_power
        }