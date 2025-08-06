"""Advanced performance monitoring and profiling system."""

import time
import asyncio
import threading
import functools
import logging
import psutil
import gc
import sys
import tracemalloc
from typing import Any, Dict, List, Optional, Union, Callable, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import weakref
import resource
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class PerformanceAlert:
    """Performance alert configuration."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    threshold: Union[int, float]
    level: AlertLevel = AlertLevel.WARNING
    cooldown: float = 300.0  # seconds
    last_triggered: float = 0.0
    callback: Optional[Callable] = None

@dataclass
class ProfilingConfig:
    """Configuration for performance profiling."""
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_io_profiling: bool = True
    memory_trace_limit: int = 10000
    profiling_interval: float = 60.0
    auto_gc_stats: bool = True
    track_top_functions: int = 20
    export_interval: float = 300.0
    export_path: Optional[Path] = None

class PerformanceCounter:
    """Thread-safe performance counter."""
    
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self, value: Union[int, float] = 1) -> None:
        """Increment counter."""
        with self._lock:
            self._value += value
    
    def decrement(self, value: Union[int, float] = 1) -> None:
        """Decrement counter."""
        with self._lock:
            self._value -= value
    
    def set(self, value: Union[int, float]) -> None:
        """Set counter value."""
        with self._lock:
            self._value = value
    
    def get(self) -> Union[int, float]:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0

class PerformanceGauge:
    """Thread-safe performance gauge."""
    
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: Union[int, float]) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = float(value)
    
    def get(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value
    
    def add(self, value: Union[int, float]) -> None:
        """Add to gauge value."""
        with self._lock:
            self._value += float(value)
    
    def subtract(self, value: Union[int, float]) -> None:
        """Subtract from gauge value."""
        with self._lock:
            self._value -= float(value)

class PerformanceHistogram:
    """Performance histogram for tracking value distributions."""
    
    def __init__(self, name: str, buckets: Optional[List[float]] = None, 
                 labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        
        self._bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._bucket_counts[float('inf')] = 0  # +Inf bucket
        self._count = 0
        self._sum = 0.0
        self._lock = threading.Lock()
    
    def observe(self, value: float) -> None:
        """Observe a value."""
        with self._lock:
            self._count += 1
            self._sum += value
            
            # Increment appropriate bucket
            for bucket in sorted(self.buckets):
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
                    break
            else:
                self._bucket_counts[float('inf')] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            return {
                'count': self._count,
                'sum': self._sum,
                'avg': self._sum / self._count if self._count > 0 else 0.0,
                'buckets': self._bucket_counts.copy()
            }

class PerformanceTimer:
    """High-precision performance timer."""
    
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self._measurements = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def time(self, func: Optional[Callable] = None):
        """Timer decorator or context manager."""
        if func is not None:
            # Used as decorator
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure():
                    return func(*args, **kwargs)
            return wrapper
        else:
            # Used as context manager
            return self.measure()
    
    @contextmanager
    def measure(self):
        """Context manager for measuring execution time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record(duration)
    
    @asynccontextmanager
    async def measure_async(self):
        """Async context manager for measuring execution time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record(duration)
    
    def record(self, duration: float) -> None:
        """Record a timing measurement."""
        with self._lock:
            self._measurements.append(duration)
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        with self._lock:
            if not self._measurements:
                return {
                    'count': 0,
                    'total': 0.0,
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
            
            measurements = sorted(self._measurements)
            count = len(measurements)
            total = sum(measurements)
            
            return {
                'count': count,
                'total': total,
                'avg': total / count,
                'min': measurements[0],
                'max': measurements[-1],
                'p50': measurements[int(count * 0.5)],
                'p95': measurements[int(count * 0.95)],
                'p99': measurements[int(count * 0.99)]
            }

class SystemMetrics:
    """System-level performance metrics collector."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics."""
        try:
            return {
                'cpu_percent': self.process.cpu_percent(),
                'cpu_times_user': self.process.cpu_times().user,
                'cpu_times_system': self.process.cpu_times().system,
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                'system_cpu_percent': psutil.cpu_percent(),
                'system_cpu_count': psutil.cpu_count(),
                'load_avg_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
                'load_avg_5m': psutil.getloadavg()[1] if hasattr(psutil, 'getloadavg') else 0.0,
                'load_avg_15m': psutil.getloadavg()[2] if hasattr(psutil, 'getloadavg') else 0.0,
            }
        except Exception as e:
            self.logger.warning(f"Error collecting CPU metrics: {e}")
            return {}
    
    def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics."""
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'memory_rss': memory_info.rss,
                'memory_vms': memory_info.vms,
                'memory_percent': self.process.memory_percent(),
                'memory_available': system_memory.available,
                'memory_used': system_memory.used,
                'memory_total': system_memory.total,
                'memory_cached': getattr(system_memory, 'cached', 0),
                'memory_buffers': getattr(system_memory, 'buffers', 0),
                'swap_used': psutil.swap_memory().used,
                'swap_total': psutil.swap_memory().total,
            }
        except Exception as e:
            self.logger.warning(f"Error collecting memory metrics: {e}")
            return {}
    
    def collect_io_metrics(self) -> Dict[str, float]:
        """Collect I/O metrics."""
        try:
            io_counters = self.process.io_counters()
            disk_io = psutil.disk_io_counters()
            
            metrics = {
                'io_read_count': io_counters.read_count,
                'io_write_count': io_counters.write_count,
                'io_read_bytes': io_counters.read_bytes,
                'io_write_bytes': io_counters.write_bytes,
            }
            
            if disk_io:
                metrics.update({
                    'disk_read_count': disk_io.read_count,
                    'disk_write_count': disk_io.write_count,
                    'disk_read_bytes': disk_io.read_bytes,
                    'disk_write_bytes': disk_io.write_bytes,
                    'disk_read_time': disk_io.read_time,
                    'disk_write_time': disk_io.write_time,
                })
            
            return metrics
        except Exception as e:
            self.logger.warning(f"Error collecting I/O metrics: {e}")
            return {}
    
    def collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics."""
        try:
            net_io = psutil.net_io_counters()
            
            if net_io:
                return {
                    'network_bytes_sent': net_io.bytes_sent,
                    'network_bytes_recv': net_io.bytes_recv,
                    'network_packets_sent': net_io.packets_sent,
                    'network_packets_recv': net_io.packets_recv,
                    'network_errin': net_io.errin,
                    'network_errout': net_io.errout,
                    'network_dropin': net_io.dropin,
                    'network_dropout': net_io.dropout,
                }
            return {}
        except Exception as e:
            self.logger.warning(f"Error collecting network metrics: {e}")
            return {}
    
    def collect_gc_metrics(self) -> Dict[str, float]:
        """Collect garbage collection metrics."""
        try:
            gc_stats = gc.get_stats()
            gc_counts = gc.get_count()
            
            metrics = {
                'gc_collections_gen0': gc_counts[0],
                'gc_collections_gen1': gc_counts[1],
                'gc_collections_gen2': gc_counts[2],
                'gc_objects': len(gc.get_objects()),
            }
            
            for i, stats in enumerate(gc_stats):
                metrics.update({
                    f'gc_gen{i}_collections': stats['collections'],
                    f'gc_gen{i}_collected': stats['collected'],
                    f'gc_gen{i}_uncollectable': stats['uncollectable'],
                })
            
            return metrics
        except Exception as e:
            self.logger.warning(f"Error collecting GC metrics: {e}")
            return {}
    
    def collect_all_metrics(self) -> Dict[str, float]:
        """Collect all system metrics."""
        all_metrics = {}
        all_metrics.update(self.collect_cpu_metrics())
        all_metrics.update(self.collect_memory_metrics())
        all_metrics.update(self.collect_io_metrics())
        all_metrics.update(self.collect_network_metrics())
        all_metrics.update(self.collect_gc_metrics())
        all_metrics['timestamp'] = time.time()
        return all_metrics

class MemoryProfiler:
    """Memory profiling and tracking."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self._snapshots = deque(maxlen=100)
        self._top_stats = deque(maxlen=50)
        
        if config.enable_memory_profiling:
            tracemalloc.start(config.memory_trace_limit)
    
    def take_snapshot(self) -> None:
        """Take memory snapshot."""
        if not self.config.enable_memory_profiling:
            return
        
        try:
            snapshot = tracemalloc.take_snapshot()
            self._snapshots.append({
                'timestamp': time.time(),
                'snapshot': snapshot,
                'stats': self._analyze_snapshot(snapshot)
            })
            
            self.logger.debug(f"Memory snapshot taken, total snapshots: {len(self._snapshots)}")
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
    
    def _analyze_snapshot(self, snapshot) -> Dict[str, Any]:
        """Analyze memory snapshot."""
        top_stats = snapshot.statistics('lineno')[:self.config.track_top_functions]
        
        total_size = sum(stat.size for stat in top_stats)
        total_count = sum(stat.count for stat in top_stats)
        
        return {
            'total_size': total_size,
            'total_count': total_count,
            'top_functions': [
                {
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size': stat.size,
                    'count': stat.count,
                    'size_mb': stat.size / (1024 * 1024)
                }
                for stat in top_stats[:10]
            ]
        }
    
    def compare_snapshots(self, snapshot1_idx: int = -2, snapshot2_idx: int = -1) -> Optional[Dict[str, Any]]:
        """Compare two memory snapshots."""
        if len(self._snapshots) < 2:
            return None
        
        try:
            snap1 = self._snapshots[snapshot1_idx]['snapshot']
            snap2 = self._snapshots[snapshot2_idx]['snapshot']
            
            top_stats = snap2.compare_to(snap1, 'lineno')[:self.config.track_top_functions]
            
            return {
                'timestamp': time.time(),
                'comparison': [
                    {
                        'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                        'size_diff': stat.size_diff,
                        'count_diff': stat.count_diff,
                        'size_diff_mb': stat.size_diff / (1024 * 1024)
                    }
                    for stat in top_stats[:10]
                ]
            }
        except Exception as e:
            self.logger.error(f"Error comparing snapshots: {e}")
            return None
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        if not self.config.enable_memory_profiling:
            return {}
        
        try:
            current, peak = tracemalloc.get_traced_memory()
            return {
                'current_mb': current / (1024 * 1024),
                'peak_mb': peak / (1024 * 1024),
                'snapshots_count': len(self._snapshots)
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def stop_profiling(self) -> None:
        """Stop memory profiling."""
        if self.config.enable_memory_profiling:
            tracemalloc.stop()

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        self.config = config or ProfilingConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Metric storage
        self._counters: Dict[str, PerformanceCounter] = {}
        self._gauges: Dict[str, PerformanceGauge] = {}
        self._histograms: Dict[str, PerformanceHistogram] = {}
        self._timers: Dict[str, PerformanceTimer] = {}
        
        # System metrics
        self._system_metrics = SystemMetrics()
        self._memory_profiler = MemoryProfiler(self.config)
        
        # Alerts
        self._alerts: List[PerformanceAlert] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._export_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Metric history
        self._metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        self._lock = threading.RLock()
    
    def counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> PerformanceCounter:
        """Get or create a counter."""
        key = f"{name}:{hash(frozenset((labels or {}).items()))}"
        
        with self._lock:
            if key not in self._counters:
                self._counters[key] = PerformanceCounter(name, labels)
            return self._counters[key]
    
    def gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> PerformanceGauge:
        """Get or create a gauge."""
        key = f"{name}:{hash(frozenset((labels or {}).items()))}"
        
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = PerformanceGauge(name, labels)
            return self._gauges[key]
    
    def histogram(self, name: str, buckets: Optional[List[float]] = None, 
                 labels: Optional[Dict[str, str]] = None) -> PerformanceHistogram:
        """Get or create a histogram."""
        key = f"{name}:{hash(frozenset((labels or {}).items()))}"
        
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = PerformanceHistogram(name, buckets, labels)
            return self._histograms[key]
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> PerformanceTimer:
        """Get or create a timer."""
        key = f"{name}:{hash(frozenset((labels or {}).items()))}"
        
        with self._lock:
            if key not in self._timers:
                self._timers[key] = PerformanceTimer(name, labels)
            return self._timers[key]
    
    def add_alert(self, alert: PerformanceAlert) -> None:
        """Add performance alert."""
        with self._lock:
            self._alerts.append(alert)
            self.logger.info(f"Added performance alert: {alert.name}")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric point."""
        point = MetricPoint(time.time(), value, labels or {})
        
        with self._lock:
            self._metric_history[name].append(point)
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self.config.profiling_interval > 0:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.config.export_interval > 0 and self.config.export_path:
            self._export_task = asyncio.create_task(self._export_loop())
        
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self._shutdown_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        
        self._memory_profiler.stop_profiling()
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                system_metrics = self._system_metrics.collect_all_metrics()
                for name, value in system_metrics.items():
                    self.record_metric(f"system.{name}", value)
                
                # Take memory snapshot
                if self.config.enable_memory_profiling:
                    self._memory_profiler.take_snapshot()
                
                # Check alerts
                await self._check_alerts()
                
                # Force garbage collection if enabled
                if self.config.auto_gc_stats:
                    gc.collect()
                
                await asyncio.sleep(self.config.profiling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.profiling_interval)
    
    async def _export_loop(self) -> None:
        """Background export loop."""
        while not self._shutdown_event.is_set():
            try:
                await self.export_metrics()
                await asyncio.sleep(self.config.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in export loop: {e}")
                await asyncio.sleep(self.config.export_interval)
    
    async def _check_alerts(self) -> None:
        """Check performance alerts."""
        current_time = time.time()
        
        with self._lock:
            for alert in self._alerts:
                # Check cooldown
                if current_time - alert.last_triggered < alert.cooldown:
                    continue
                
                # Get latest metric value
                if alert.metric_name in self._metric_history:
                    history = self._metric_history[alert.metric_name]
                    if history:
                        latest_value = history[-1].value
                        
                        # Check condition
                        triggered = self._check_alert_condition(latest_value, alert)
                        
                        if triggered:
                            alert.last_triggered = current_time
                            await self._trigger_alert(alert, latest_value)
    
    def _check_alert_condition(self, value: Union[int, float], alert: PerformanceAlert) -> bool:
        """Check if alert condition is met."""
        if alert.condition == "gt":
            return value > alert.threshold
        elif alert.condition == "lt":
            return value < alert.threshold
        elif alert.condition == "eq":
            return value == alert.threshold
        elif alert.condition == "ne":
            return value != alert.threshold
        elif alert.condition == "gte":
            return value >= alert.threshold
        elif alert.condition == "lte":
            return value <= alert.threshold
        return False
    
    async def _trigger_alert(self, alert: PerformanceAlert, value: Union[int, float]) -> None:
        """Trigger performance alert."""
        self.logger.log(
            logging.WARNING if alert.level == AlertLevel.WARNING else logging.ERROR,
            f"Performance alert '{alert.name}': {alert.metric_name} = {value} {alert.condition} {alert.threshold}"
        )
        
        if alert.callback:
            try:
                if asyncio.iscoroutinefunction(alert.callback):
                    await alert.callback(alert, value)
                else:
                    alert.callback(alert, value)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    async def export_metrics(self) -> None:
        """Export metrics to file."""
        if not self.config.export_path:
            return
        
        try:
            export_data = {
                'timestamp': time.time(),
                'counters': {name: counter.get() for name, counter in self._counters.items()},
                'gauges': {name: gauge.get() for name, gauge in self._gauges.items()},
                'histograms': {name: hist.get_stats() for name, hist in self._histograms.items()},
                'timers': {name: timer.get_stats() for name, timer in self._timers.items()},
                'system_metrics': self._system_metrics.collect_all_metrics(),
                'memory_usage': self._memory_profiler.get_current_memory_usage()
            }
            
            export_file = self.config.export_path / f"metrics_{int(time.time())}.json"
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.debug(f"Exported metrics to {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                'counters': {name: counter.get() for name, counter in self._counters.items()},
                'gauges': {name: gauge.get() for name, gauge in self._gauges.items()},
                'histograms': {name: hist.get_stats() for name, hist in self._histograms.items()},
                'timers': {name: timer.get_stats() for name, timer in self._timers.items()},
                'system_metrics': self._system_metrics.collect_all_metrics(),
                'memory_usage': self._memory_profiler.get_current_memory_usage(),
                'alerts': len(self._alerts)
            }
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[MetricPoint]:
        """Get metric history."""
        with self._lock:
            history = self._metric_history.get(name, deque())
            return list(history)[-limit:]
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            
            self._metric_history.clear()
            self.logger.info("All metrics reset")


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance.
    
    Returns:
        Performance monitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Decorators for easy performance monitoring
def monitor_performance(metric_name: Optional[str] = None, 
                       labels: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
        labels: Additional labels for the metric
    """
    def decorator(func):
        name = metric_name or f"function.{func.__name__}"
        timer = get_performance_monitor().timer(name, labels)
        counter = get_performance_monitor().counter(f"{name}.calls", labels)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counter.increment()
            with timer.measure():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_async_performance(metric_name: Optional[str] = None,
                             labels: Optional[Dict[str, str]] = None):
    """Decorator to monitor async function performance.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
        labels: Additional labels for the metric
    """
    def decorator(func):
        name = metric_name or f"async_function.{func.__name__}"
        timer = get_performance_monitor().timer(name, labels)
        counter = get_performance_monitor().counter(f"{name}.calls", labels)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            counter.increment()
            async with timer.measure_async():
                return await func(*args, **kwargs)
        return wrapper
    return decorator